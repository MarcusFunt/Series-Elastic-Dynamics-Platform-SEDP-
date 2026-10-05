"""Run and summarize the predeclared Phase 5 PPO ablation matrix."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


TRAINING_SEEDS = (173, 271, 389)
EVALUATION_SEEDS = (1001, 1002, 1003)
ABLATIONS = (
    {'name': 'baseline_estimated_history8_preview', 'history': 8,
     'oracle_state': False, 'preview_enabled': True},
    {'name': 'oracle_state_upper_bound', 'history': 8,
     'oracle_state': True, 'preview_enabled': True},
    {'name': 'estimated_history1', 'history': 1,
     'oracle_state': False, 'preview_enabled': True},
    {'name': 'estimated_history8_no_preview', 'history': 8,
     'oracle_state': False, 'preview_enabled': False},
)
METRICS = (
    'position_rmse_mm', 'angle_rms_deg', 'peak_angle_deg', 'energy_integral_mJs',
    'peak_rail_fraction', 'saturation_fraction', 'stop_contact_fraction',
    'max_soft_rail_violation_fraction', 'max_carriage_speed_violation_m_s',
    'max_motor_speed_violation_rad_s', 'estimator_position_rmse_mm', 'completion_rate',
)


def _git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def _git_revision(root):
    return _git(root, 'rev-parse', 'HEAD')


def _source_is_clean(root):
    return not _git(root, 'status', '--porcelain', '--untracked-files=normal')


def _run_key(config_name, seed):
    return f'{config_name}/seed_{seed}'


def _case_key(row):
    return (row['scenario'], int(row['seed']), bool(row['randomized']))


def _metrics(row):
    values = {name: float(row[name]) for name in METRICS if name != 'completion_rate'}
    values['completion_rate'] = float(not row['terminated'])
    return values


def _mean(values):
    return float(statistics.mean(values)) if values else None


def _std(values):
    return float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)


def _summarize_run(evaluation):
    policies = {}
    lqr = {}
    for row in evaluation['rows']:
        if row['controller'] == 'policy':
            policies[_case_key(row)] = _metrics(row)
        elif row['controller'] == 'lqr':
            lqr[_case_key(row)] = _metrics(row)
    keys = sorted(set(policies) & set(lqr))
    if len(keys) != len(policies) or len(keys) != len(lqr):
        raise ValueError('Evaluation policy and LQR rows do not form identical paired cases')
    aggregate = {'policy': {}, 'lqr': {}, 'policy_minus_lqr': {}}
    for metric in METRICS:
        aggregate['policy'][metric] = _mean([policies[key][metric] for key in keys])
        aggregate['lqr'][metric] = _mean([lqr[key][metric] for key in keys])
        aggregate['policy_minus_lqr'][metric] = _mean(
            [policies[key][metric] - lqr[key][metric] for key in keys]
        )
    return {'paired_cases': len(keys), 'policy': policies, 'lqr': lqr,
            'aggregate': aggregate}


def _summarize_configs(ablations, run_records, training_seeds):
    by_config = {}
    for ablation in ablations:
        name = ablation['name']
        records = [run_records[_run_key(name, seed)] for seed in training_seeds]
        per_seed = {}
        for seed, record in zip(training_seeds, records):
            eval_summary = record['evaluation_summary']
            per_seed[str(seed)] = {
                'training_wall_seconds': record['training_metadata']['wall_seconds'],
                'final_training_mean_reward': record['training'][-1]['mean_reward'],
                'promotion': record['promotion'],
                'aggregate_metrics': eval_summary['aggregate'],
                'evaluation_case_count': eval_summary['paired_cases'],
            }
        summary = {}
        for cohort in ('policy', 'lqr', 'policy_minus_lqr'):
            summary[cohort] = {}
            for metric in METRICS:
                values = [per_seed[str(seed)]['aggregate_metrics'][cohort][metric]
                          for seed in training_seeds]
                summary[cohort][metric] = {'mean': _mean(values), 'std': _std(values)}
        summary['final_training_mean_reward'] = {
            'mean': _mean([per_seed[str(seed)]['final_training_mean_reward']
                           for seed in training_seeds]),
            'std': _std([per_seed[str(seed)]['final_training_mean_reward']
                         for seed in training_seeds]),
        }
        by_config[name] = {'settings': ablation, 'per_training_seed': per_seed,
                           'summary': summary}

    baseline_name = ablations[0]['name']
    baseline_records = {seed: run_records[_run_key(baseline_name, seed)]
                        for seed in training_seeds}
    for ablation in ablations[1:]:
        name = ablation['name']
        seed_deltas = {metric: [] for metric in METRICS}
        for seed in training_seeds:
            base = baseline_records[seed]['evaluation_summary']['policy']
            candidate = run_records[_run_key(name, seed)]['evaluation_summary']['policy']
            common = sorted(set(base) & set(candidate))
            if set(base) != set(candidate):
                raise ValueError('Ablation and baseline evaluations do not share identical cases')
            for metric in METRICS:
                seed_deltas[metric].append(_mean([
                    candidate[key][metric] - base[key][metric] for key in common
                ]))
        by_config[name]['change_vs_baseline'] = {
            metric: {'mean_across_training_seeds': _mean(values),
                     'std_across_training_seeds': _std(values)}
            for metric, values in seed_deltas.items()
        }
    by_config[baseline_name]['change_vs_baseline'] = {
        metric: {'mean_across_training_seeds': 0.0,
                 'std_across_training_seeds': 0.0}
        for metric in METRICS
    }
    return by_config


def _write_report(path, report):
    configs = report['configurations']
    lines = [
        '# Phase 5 PPO Ablation Results', '',
        f"Source revision: `{report['source_revision']}`  ",
        f"Training steps per seed: {report['budget']['steps']:,}; "
        f"training seeds: {', '.join(map(str, report['training_seeds']))}; "
        f"shared held-out seeds: {', '.join(map(str, report['evaluation_seeds']))}.", '',
        'Metrics below average each policy over the same three fixed motion scenarios and '
        'three randomized held-out references. The uncertainty column is the sample standard '
        'deviation across the three independent training seeds.', '',
        '| Configuration | Position RMSE (mm) | Angle RMS (deg) | Energy integral (mJ·s) | Completion | Δ energy vs baseline |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for ablation in ABLATIONS:
        name = ablation['name']
        item = configs[name]
        summary = item['summary']
        def cell(cohort, metric, delta=False):
            if delta:
                obj = item['change_vs_baseline'][metric]
                mean, sd = obj['mean_across_training_seeds'], obj['std_across_training_seeds']
            else:
                obj = summary[cohort][metric]
                mean, sd = obj['mean'], obj['std']
            return f'{mean:.3f} ± {sd:.3f}'
        lines.append(
            f"| {name} | {cell('policy', 'position_rmse_mm')} "
            f"| {cell('policy', 'angle_rms_deg')} "
            f"| {cell('policy', 'energy_integral_mJs')} "
            f"| {cell('policy', 'completion_rate')} "
            f"| {cell('policy', 'energy_integral_mJs', delta=True)} |"
        )
    lines.extend([
        '',
        'Lower values are better for position error, angle, energy, rail/speed violations, '
        'and saturation. Higher completion is better. Oracle-state results are an information '
        'upper bound and are not eligible for deployment. No ablation is promoted unless every '
        'existing `evaluate_v4.promotion` safety and paired-performance gate accepts it.', '',
        f"Raw evaluation rows and per-run training logs are embedded in `{report['evidence_path']}`. "
        f"Checkpoint and full per-run artifacts remain under `{report['run_root']}` (ignored by Git).",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--output', type=Path, default=Path('runs/phase5/ppo_v5'))
    parser.add_argument('--evidence', type=Path,
                        default=Path('docs/control/evidence/ppo_ablation_v5.json'))
    parser.add_argument('--report', type=Path, default=Path('docs/control/PPO_PHASE5.md'))
    parser.add_argument('--training-seeds', nargs='+', type=int, default=list(TRAINING_SEEDS))
    parser.add_argument('--evaluation-seeds', nargs='+', type=int, default=list(EVALUATION_SEEDS))
    args = parser.parse_args()
    if min(args.steps, args.envs, args.rollout) <= 0:
        parser.error('steps, envs, and rollout must be positive')

    repo_root = Path(__file__).resolve().parents[2]
    if not _source_is_clean(repo_root):
        parser.error('commit the experiment code and predeclared plan before launching PPO runs')
    revision = _git_revision(repo_root)
    output = args.output if args.output.is_absolute() else repo_root / args.output
    if output.exists():
        parser.error(f'output directory already exists: {output}; choose a fresh --output path')
    output.mkdir(parents=True)
    budget = {'steps': args.steps, 'n_envs': args.envs, 'rollout': args.rollout,
              'learning_rate': 3e-5, 'anchor_kl': .005, 'ppo_epochs': 3,
              'torch_threads': 1, 'actuator_mode': 'torque',
              'teacher_initialization': False,
              'note': 'PPO starts from a seeded residual actor around constrained LQR; no v4 teacher checkpoint is present in the checkout.'}
    manifest = {
        'experiment': 'SEDP Phase 5 PPO estimator/history/preview ablation',
        'source_revision': revision, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'budget': budget,
        'training_seeds': args.training_seeds, 'evaluation_seeds': args.evaluation_seeds,
        'evaluation_cases': 'all fixed SCENARIOS plus randomized references for shared evaluation seeds',
        'configurations': list(ABLATIONS),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    run_records = {}
    trainer = Path(__file__).with_name('train_residual_v4.py')
    started_all = time.perf_counter()
    for ablation in ABLATIONS:
        for seed in args.training_seeds:
            name = ablation['name']
            key = _run_key(name, seed)
            run_dir = output / name / f'seed_{seed}'
            run_dir.mkdir(parents=True)
            command = [sys.executable, str(trainer), 'ppo', '--seed', str(seed),
                       '--steps', str(args.steps), '--envs', str(args.envs),
                       '--rollout', str(args.rollout), '--eval-seeds',
                       *map(str, args.evaluation_seeds), '--history', str(ablation['history']),
                       '--outdir', str(run_dir)]
            if ablation['oracle_state']:
                command.append('--oracle-state')
            if not ablation['preview_enabled']:
                command.append('--no-preview')
            print(f'\n=== {key} ===\n{json.dumps(command)}', flush=True)
            run_started = time.perf_counter()
            log_path = run_dir / 'stdout.log'
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=repo_root, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding='utf-8', errors='replace',
                                           env={**os.environ, 'OMP_NUM_THREADS': '1',
                                                'MKL_NUM_THREADS': '1'})
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
            return_code = process.wait()
            run_elapsed = time.perf_counter() - run_started
            if return_code:
                raise RuntimeError(f'{key} failed with exit code {return_code}; see {log_path}')
            evaluation_path = run_dir / 'evaluation.json'
            training_path = run_dir / 'training.json'
            metadata_path = run_dir / 'run_metadata.json'
            if not all(path.is_file() for path in (evaluation_path, training_path,
                                                   metadata_path, run_dir / 'policy_candidate.pt')):
                raise RuntimeError(f'{key} completed without all expected artifacts')
            if json.loads(metadata_path.read_text(encoding='utf-8'))['source_revision'] != revision:
                raise RuntimeError(f'{key} source revision does not match the predeclared revision')
            evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
            training = json.loads(training_path.read_text(encoding='utf-8'))
            training_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            run_records[key] = {
                'configuration': ablation, 'training_seed': seed,
                'run_wall_seconds_including_evaluation': run_elapsed,
                'training_metadata': training_metadata, 'training': training,
                'evaluation_summary': _summarize_run(evaluation),
                'promotion': evaluation['promotion'],
                'evaluation_rows': evaluation['rows'],
            }
            print(f'Completed {key}: {run_elapsed:.1f}s; '
                  f"promotion accepted={evaluation['promotion']['accepted']}", flush=True)

    configurations = _summarize_configs(ABLATIONS, run_records, args.training_seeds)
    for record in run_records.values():
        paired = record['evaluation_summary']
        raw_policy, raw_lqr = paired.pop('policy'), paired.pop('lqr')
        paired['case_metrics'] = [
            {'scenario': key[0], 'evaluation_seed': key[1], 'randomized': key[2],
             'policy': raw_policy[key], 'lqr': raw_lqr[key]}
            for key in sorted(raw_policy)
        ]
    report = {
        **manifest,
        'run_root': str(output.relative_to(repo_root)),
        'evidence_path': str(args.evidence),
        'total_wall_seconds': time.perf_counter() - started_all,
        'configurations': configurations,
        'runs': run_records,
    }
    evidence = args.evidence if args.evidence.is_absolute() else repo_root / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report_path = args.report if args.report.is_absolute() else repo_root / args.report
    _write_report(report_path, report)
    print(f'\nWrote {evidence}\nWrote {report_path}\n', flush=True)


if __name__ == '__main__':
    main()

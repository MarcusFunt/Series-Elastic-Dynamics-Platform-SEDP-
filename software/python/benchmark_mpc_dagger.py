"""Collect student-state MPC labels, fine-tune with DAgger, and evaluate fresh seeds."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from active_vibration_rig_2d import PlantParams
from benchmark_mpc_imitation import compare_imitation_to_mpc
from constrained_mpc import MPCConfig
from evaluate_goal_hold_v4 import run_case
from evaluate_v4 import load_policy
from rig_rl_env_v4 import RLEnvConfigV4


def validate_dagger_seed_separation(prior_seed_groups, dagger_seed,
                                    dagger_episodes, evaluation_seeds):
    """Validate all historical, DAgger-rollout, and fresh evaluation seeds."""
    if dagger_episodes < 1:
        raise ValueError('DAgger episode count must be positive')
    dagger = [int(dagger_seed) + 1009 * i for i in range(int(dagger_episodes))]
    evaluation = [int(seed) for seed in evaluation_seeds]
    groups = {str(name): [int(seed) for seed in seeds]
              for name, seeds in prior_seed_groups.items()}
    groups['dagger'] = dagger
    groups['fresh evaluation'] = evaluation
    for name, seeds in groups.items():
        if not seeds:
            raise ValueError(f'{name} seed set must be nonempty')
        if any(seed < 0 for seed in seeds):
            raise ValueError(f'{name} seeds must be nonnegative')
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'{name} seeds must be unique')
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if set(groups[left]) & set(groups[right]):
                raise ValueError(f'{left} and {right} seed sets must be disjoint')
    return dagger, evaluation


def compare_student_versions(original_rows, dagger_rows):
    """Summarize paired changes from the original student to its DAgger update."""
    original = {int(row['seed']): row for row in original_rows}
    dagger = {int(row['seed']): row for row in dagger_rows}
    if not original or set(original) != set(dagger):
        raise ValueError('Student versions must have identical nonempty seed coverage')
    pairs = []
    for seed in sorted(original):
        before, after = original[seed], dagger[seed]
        pairs.append({
            'seed': seed,
            'energy_ratio_dagger_over_original': (
                after['integrated_resonator_energy_mJs'] /
                max(before['integrated_resonator_energy_mJs'], 1e-12)),
            'position_rmse_change_mm': after['position_rmse_mm'] - before['position_rmse_mm'],
            'angle_rms_change_deg': after['angle_rms_deg'] - before['angle_rms_deg'],
            'peak_angle_change_deg': after['peak_angle_deg'] - before['peak_angle_deg'],
            'peak_rail_fraction_change': after['peak_rail_fraction'] - before['peak_rail_fraction'],
            'original_action_p95_ms': before['policy_action_p95_ms'],
            'dagger_action_p95_ms': after['policy_action_p95_ms'],
            'original_completed_holds': before['completed_two_second_holds'],
            'dagger_completed_holds': after['completed_two_second_holds'],
        })
    ratios = [pair['energy_ratio_dagger_over_original'] for pair in pairs]
    return {
        'mean_energy_ratio_dagger_over_original': sum(ratios) / len(ratios),
        'seeds_lower_energy': sum(ratio < 1. for ratio in ratios),
        'seeds_energy_within_2_percent': sum(abs(ratio - 1.) <= .02 for ratio in ratios),
        'pairs': pairs,
    }


def _source_fingerprint(repo_root):
    paths = ('software/python/constrained_mpc.py',
             'software/python/evaluate_goal_hold_v4.py',
             'software/python/train_residual_v4.py',
             'software/python/benchmark_mpc_dagger.py')
    dependencies = (
        'software/python/active_vibration_rig_2d.py',
        'software/python/benchmark_mpc_imitation.py',
        'software/python/benchmark_suite.py',
        'software/python/evaluate_v4.py',
        'software/python/legacy_constrained_mpc.py',
        'software/python/mpc_kernels.py',
        'software/python/ppo_agent_v2.py',
        'software/python/residual_control_v4.py',
        'software/python/rig_rl_env_v3.py',
        'software/python/rig_rl_env_v4.py',
        'software/python/state_estimator.py',
        'software/python/timing_models.py',
    )
    def digest_files(files):
        digest = hashlib.sha256()
        for relative in files:
            digest.update(relative.encode())
            digest.update((repo_root / relative).read_bytes())
        return digest.hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    return {'git_revision': revision, 'source_sha256': digest_files(paths),
            'source_files': list(paths),
            'dependency_sha256': digest_files(dependencies),
            'dependency_files': list(dependencies)}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _write_report(path, data):
    train = data['training']
    original = data['original_student_vs_mpc']
    dagger = data['dagger_student_vs_mpc']
    delta = data['dagger_vs_original_student']
    lines = [
        '# MPC DAgger Evaluation', '',
        f"- Status: {data['status']}",
        f"- Source study: `{data['source_run']}`",
        f"- Runtime dependency SHA256: `{data['source']['dependency_sha256']}` "
        f"({len(data['source']['dependency_files'])} dependency files).",
        f"- Task: random goal hold, {data['config']['goal_edge_margin_m'] * 1000:g} mm edge exclusion, "
        f"{data['config']['goal_hold_min_seconds']:g} s minimum target hold.",
        f"- MPC: horizon {data['selected_mpc_config']['horizon']}, linearization stride "
        f"{data['selected_mpc_config']['linearization_stride']}.",
        f"- DAgger rollout seeds: `{data['dagger_episode_seeds']}`",
        f"- Fresh evaluation seeds: `{data['evaluation_seeds']}`", '',
        '## Training', '',
        f"Collected {train['dagger_samples']} student-rollout states; "
        f"{train['dagger_successful_labels']} MPC labels were successful and "
        f"{train['dagger_fallback_labels']} fallback labels were excluded.",
        f"Combined dataset: {train['combined_samples']} labels ({train['original_samples']} original, "
        f"{train['dagger_samples']} DAgger); {train['dagger_fit']['excluded_fallback_labels']} "
        f"unsuccessful MPC labels were excluded from the combined fit. Best episode-held-out action MAE: "
        f"{train['best_heldout_action_mae']:.6f}.",
        f"Warm-start checkpoint: `{train['initial_student']}`.", '',
        '## Paired comparison against MPC', '',
        '| Student | Passed behavior gates | p95 speedup vs MPC | Mean energy ratio vs MPC | Failed paired gates |',
        '|---|---:|---:|---:|---:|',
        f"| Original | {original['performance_preserved']} | {original['p95_latency_speedup']:.2f}× | "
        f"{original['mean_energy_ratio']:.4f} | {len(original['reasons'])} |",
        f"| DAgger | {dagger['performance_preserved']} | {dagger['p95_latency_speedup']:.2f}× | "
        f"{dagger['mean_energy_ratio']:.4f} | {len(dagger['reasons'])} |", '',
        '## DAgger change on paired fresh seeds', '',
        f"DAgger/original mean resonator-energy ratio: {delta['mean_energy_ratio_dagger_over_original']:.4f}; "
        f"lower energy on {delta['seeds_lower_energy']}/{len(delta['pairs'])} seeds and within ±2% "
        f"on {delta['seeds_energy_within_2_percent']}/{len(delta['pairs'])}. "
        f"Policy action p95 means: {dagger['mean_imitation_action_p95_ms']:.4f} ms (DAgger), "
        f"{original['mean_imitation_action_p95_ms']:.4f} ms (original).", '',
        '## Per-seed DAgger vs MPC', '',
        '| Seed | Energy ratio | Position RMSE change (mm) | Angle RMS change (deg) | Holds | Gate failures |',
        '|---:|---:|---:|---:|---:|---|',
    ]
    dagger_eval = {int(row['seed']): row for row in data['dagger_student_rows']}
    for pair in dagger['pairs']:
        row = dagger_eval[pair['seed']]
        lines.append(f"| {pair['seed']} | {pair['energy_ratio']:.4f} | "
                     f"{pair['position_rmse_change_mm']:+.4f} | "
                     f"{pair['angle_rms_change_deg']:+.4f} | "
                     f"{row['completed_two_second_holds']}/{row['goal_count']} | "
                     f"{'; '.join(pair['reasons']) or 'none'} |")
    lines.extend(['', f"JSON evidence: `{data['evidence_json']}`", ''])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path,
                        default=Path('runs/phase6/mpc_runtime_imitation_20261007'))
    parser.add_argument('--outdir', type=Path,
                        default=Path('runs/phase6/mpc_dagger_20261007'))
    parser.add_argument('--evidence-json', type=Path,
                        default=Path('docs/control/evidence/mpc_dagger_20261007.json'))
    parser.add_argument('--report', type=Path,
                        default=Path('docs/control/evidence/mpc_dagger_20261007.md'))
    parser.add_argument('--dagger-seed', type=int, default=42420)
    parser.add_argument('--dagger-samples', type=int, default=9600)
    parser.add_argument('--dagger-episodes', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--evaluation-seeds', nargs='*', type=int,
                        default=list(range(12011, 12021)))
    parser.add_argument('--tolerance-mm', type=float, default=5.)
    args = parser.parse_args()
    if args.dagger_samples < args.dagger_episodes or args.dagger_episodes < 4 or args.epochs < 1:
        raise ValueError('Use at least four DAgger episodes, samples >= episodes, and positive epochs')

    repo_root = Path(__file__).resolve().parents[2]
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    source_run = resolve(args.source_run)
    run_dir = resolve(args.outdir)
    evidence_json = resolve(args.evidence_json)
    report = resolve(args.report)
    source_summary = json.loads((source_run / 'summary.json').read_text(encoding='utf-8'))
    teacher_config_path = source_run / 'teacher' / 'teacher_config.json'
    original_dataset = source_run / 'teacher' / 'teacher_runs.npz'
    original_checkpoint_path = source_run / 'teacher' / 'policy_candidate.pt'
    teacher_config = json.loads(teacher_config_path.read_text(encoding='utf-8'))
    if teacher_config.get('schema_version') != 1:
        raise ValueError('Unsupported source teacher config schema')
    selected_mpc = MPCConfig(**teacher_config['mpc_config'])
    if asdict(selected_mpc) != source_summary['selected_mpc_config']:
        raise ValueError('Source summary and teacher MPC configurations do not match')
    cfg = RLEnvConfigV4(**teacher_config['env_config'])
    if cfg.reference_mode != 'random_goal_hold' or cfg.goal_edge_margin_m < .015 - 1e-12 or cfg.goal_hold_min_seconds < 2.5:
        raise ValueError('Source teacher config does not preserve the required goal-hold task')
    base_params = PlantParams(**teacher_config['plant_parameters'])

    training = source_summary['training']['teacher_config']
    historical = {
        'optimization': [row['seed'] for row in source_summary['optimization_baseline_rows']],
        'confirmation': source_summary['confirmation_seeds'],
        'teacher': [int(training['seed']) + 1009 * i
                    for i in range(int(training['teacher_episodes']))],
        'previous evaluation': source_summary['evaluation_seeds'],
    }
    dagger_seeds, evaluation_seeds = validate_dagger_seed_separation(
        historical, args.dagger_seed, args.dagger_episodes, args.evaluation_seeds)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty DAgger run directory: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source = _source_fingerprint(repo_root)
    original_model, original_cfg, original_checkpoint = load_policy(original_checkpoint_path)
    if asdict(original_cfg) != asdict(cfg):
        raise ValueError('Original student config differs from the resolved task configuration')
    if original_checkpoint.get('extra', {}).get('mpc_config') != asdict(selected_mpc):
        raise ValueError('Original student checkpoint MPC config differs from teacher config')

    train_script = Path(__file__).resolve().parent / 'train_residual_v4.py'
    dagger_checkpoint_path = run_dir / 'dagger_student' / 'policy_candidate.pt'
    train_command = [
        sys.executable, str(train_script), 'dagger', '--dataset', str(original_dataset),
        '--resolved-teacher-config', str(teacher_config_path), '--init', str(original_checkpoint_path),
        '--outdir', str(run_dir / 'dagger_student'), '--seed', str(args.dagger_seed),
        '--samples', str(args.dagger_samples), '--episodes', str(args.dagger_episodes),
        '--epochs', str(args.epochs), '--skip-evaluation',
    ]
    manifest = {
        'status': 'running', 'source': source, 'source_run': str(source_run),
        'teacher_config': str(teacher_config_path), 'original_dataset': str(original_dataset),
        'original_student': str(original_checkpoint_path), 'training_command': train_command,
        'historical_seed_groups': historical, 'dagger_episode_seeds': dagger_seeds,
        'evaluation_seeds': evaluation_seeds, 'selected_mpc_config': asdict(selected_mpc),
        'environment_config': asdict(cfg), 'plant_parameters': asdict(base_params),
        'requested_dagger_samples': args.dagger_samples,
        'dagger_episodes': args.dagger_episodes, 'epochs': args.epochs,
        'tolerance_mm': args.tolerance_mm,
    }
    _write_json(run_dir / 'run_manifest.json', manifest)
    print('EVENT ' + json.dumps({'phase': 'dagger_training_start',
          'samples': args.dagger_samples, 'episodes': args.dagger_episodes,
          'episode_seeds': dagger_seeds}), flush=True)
    subprocess.run(train_command, cwd=repo_root, check=True)

    dagger_fit = json.loads((run_dir / 'dagger_student' / 'dagger_fit.json').read_text(encoding='utf-8'))
    dagger_config = json.loads((run_dir / 'dagger_student' / 'dagger_config.json').read_text(encoding='utf-8'))
    if dagger_config['episode_seeds'] != dagger_seeds:
        raise RuntimeError('Collected DAgger episode seed list differs from the validated protocol')
    dagger_model, dagger_cfg, dagger_checkpoint = load_policy(dagger_checkpoint_path)
    if asdict(dagger_cfg) != asdict(cfg):
        raise RuntimeError('DAgger student config differs from the resolved task configuration')
    if dagger_checkpoint.get('extra', {}).get('mpc_config') != asdict(selected_mpc):
        raise RuntimeError('DAgger checkpoint MPC config differs from the resolved teacher config')

    mpc_rows=[];original_rows=[];dagger_rows=[]
    print('EVENT ' + json.dumps({'phase': 'fresh_seed_evaluation_start',
          'seeds': evaluation_seeds, 'controllers': ['mpc', 'original_student', 'dagger_student']}), flush=True)
    for seed in evaluation_seeds:
        mpc_row=run_case('mpc',seed,cfg,horizon=selected_mpc.horizon,
                         tolerance_mm=args.tolerance_mm,base_params=base_params,
                         mpc_config=selected_mpc)
        original_row=run_case('policy',seed,cfg,model=original_model,
                              horizon=selected_mpc.horizon,tolerance_mm=args.tolerance_mm,
                              base_params=base_params)
        original_row['algorithm']='original_student'
        dagger_row=run_case('policy',seed,cfg,model=dagger_model,
                            horizon=selected_mpc.horizon,tolerance_mm=args.tolerance_mm,
                            base_params=base_params)
        dagger_row['algorithm']='dagger_student'
        mpc_rows.append(mpc_row);original_rows.append(original_row);dagger_rows.append(dagger_row)
        print('EVENT ' + json.dumps({'phase':'fresh_seed_evaluation','seed':seed,
              'mpc_p95_ms':mpc_row['mpc_solve_p95_ms'],
              'original_action_p95_ms':original_row['policy_action_p95_ms'],
              'dagger_action_p95_ms':dagger_row['policy_action_p95_ms'],
              'mpc_energy_mjs':mpc_row['integrated_resonator_energy_mJs'],
              'dagger_energy_mjs':dagger_row['integrated_resonator_energy_mJs'],
              'holds':dagger_row['completed_two_second_holds'],
              'goals':dagger_row['goal_count']}),flush=True)

    original_decision=compare_imitation_to_mpc(mpc_rows,original_rows)
    dagger_decision=compare_imitation_to_mpc(mpc_rows,dagger_rows)
    delta=compare_student_versions(original_rows,dagger_rows)
    data={
        'status':'completed','source':source,'source_run':str(source_run),
        'run_directory':str(run_dir),'evidence_json':str(evidence_json),
        'config':asdict(cfg),'plant_parameters':asdict(base_params),
        'selected_mpc_config':asdict(selected_mpc),
        'historical_seed_groups':historical,'dagger_episode_seeds':dagger_seeds,
        'evaluation_seeds':evaluation_seeds,
        'training':{
            'initial_student':str(original_checkpoint_path),
            'dagger_checkpoint':str(dagger_checkpoint_path),
            'dagger_fit':dagger_fit,'dagger_config':dagger_config,
            'dagger_successful_labels':dagger_fit['dagger_successful_labels'],
            'dagger_fallback_labels':dagger_fit['dagger_fallback_labels'],
            'original_samples':dagger_fit['original_samples'],
            'dagger_samples':dagger_fit['dagger_samples'],
            'combined_samples':dagger_fit['combined_samples'],
            'best_heldout_action_mae':dagger_fit['best_heldout_action_mae'],
            'training_command':train_command,
            'original_checkpoint_source_revision':original_checkpoint.get('extra',{}).get('source_revision'),
            'dagger_checkpoint_source_revision':dagger_checkpoint.get('extra',{}).get('source_revision'),
        },
        'mpc_rows':mpc_rows,'original_student_rows':original_rows,
        'dagger_student_rows':dagger_rows,
        'original_student_vs_mpc':original_decision,
        'dagger_student_vs_mpc':dagger_decision,
        'dagger_vs_original_student':delta,
        'interpretation':{
            'energy_metric':'integrated resonator mechanical energy in mJ; not motor electrical consumption',
            'latency_scope':'Python process controller-call timing on the simulation host; not deployment-hardware timing',
            'dagger_state_distribution':'observations collected while applying the original student deterministic actions',
            'promotion':'DAgger is considered behavior-preserving only when every paired MPC behavior gate passes; useful also requires the existing latency gates',
        },
    }
    _write_json(run_dir/'summary.json',data)
    _write_json(evidence_json,data)
    _write_report(run_dir/'report.md',data)
    _write_report(report,data)
    manifest.update({'status':'completed','dagger_checkpoint':str(dagger_checkpoint_path),
                     'evidence_json':str(evidence_json),'report':str(report),
                     'dagger_behavior_preserved':dagger_decision['performance_preserved'],
                     'dagger_useful':dagger_decision['useful']})
    _write_json(run_dir/'run_manifest.json',manifest)
    print('RESULT '+json.dumps({
        'dagger_behavior_preserved':dagger_decision['performance_preserved'],
        'dagger_useful':dagger_decision['useful'],
        'dagger_p95_speedup':dagger_decision['p95_latency_speedup'],
        'mean_energy_ratio':dagger_decision['mean_energy_ratio'],
        'dagger_vs_original_energy_ratio':delta['mean_energy_ratio_dagger_over_original'],
        'evidence_json':str(evidence_json),'report':str(report),'run_directory':str(run_dir)}),flush=True)


if __name__ == '__main__':
    main()

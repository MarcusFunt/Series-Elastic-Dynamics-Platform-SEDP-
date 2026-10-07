"""Benchmark a faster MPC linearization setting, then distill and evaluate it.

This finite experiment first compares full per-stage torque linearization with
reuse every two stages on paired random-goal episodes. It only trains an MPC
student after the faster setting clears the paired behavior gate (or selects
the unchanged setting if it does not).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from active_vibration_rig_2d import PlantParams
from constrained_mpc import MPCConfig
from evaluate_goal_hold_v4 import goal_hold_mpc_config, run_case, summarize
from evaluate_v4 import load_policy


def judge_mpc_optimization(baseline_rows, candidate_rows):
    """Accept a stride only when paired behavior holds and p95 improves >=10%."""
    baseline = {int(row['seed']): row for row in baseline_rows}
    candidate = {int(row['seed']): row for row in candidate_rows}
    reasons = []
    if set(baseline) != set(candidate) or not baseline:
        reasons.append('baseline and candidate seed coverage differs or is empty')

    pairs = []
    for seed in sorted(set(baseline) & set(candidate)):
        ref, row = baseline[seed], candidate[seed]
        pair_reasons = []
        for metric, allowance in (
                ('position_rmse_mm', .05),
                ('angle_rms_deg', .01),
                ('peak_angle_deg', .01)):
            if row[metric] > ref[metric] * 1.02 + allowance:
                pair_reasons.append(f'{metric} regressed beyond the paired bound')
        if row['peak_rail_fraction'] > ref['peak_rail_fraction'] + .01:
            pair_reasons.append('peak rail fraction regressed by more than 0.01')
        if row['integrated_resonator_energy_mJs'] > ref['integrated_resonator_energy_mJs'] * 1.02:
            pair_reasons.append('integrated resonator energy regressed by more than 2%')
        if row['terminated']:
            pair_reasons.append('candidate terminated')
        if row['exclusion_zone_violations']:
            pair_reasons.append('candidate entered the 15 mm exclusion zone')
        if not row['goal_targets_respect_margin']:
            pair_reasons.append('candidate generated a target inside the exclusion margin')
        if row['goal_count'] <= 0 or row['completed_two_second_holds'] != row['goal_count']:
            pair_reasons.append('candidate did not complete every two-second goal hold')
        if row['saturation_fraction'] > ref['saturation_fraction'] + .02:
            pair_reasons.append('saturation fraction regressed by more than 0.02')
        if row['mpc_fallback_fraction'] > ref['mpc_fallback_fraction'] + .02:
            pair_reasons.append('fallback fraction regressed by more than 0.02')
        if row['mpc_deadline_miss_fraction'] > ref['mpc_deadline_miss_fraction'] + .01:
            pair_reasons.append('10 ms deadline-miss fraction regressed by more than 0.01')
        pairs.append({
            'seed': seed,
            'baseline_solve_p95_ms': ref['mpc_solve_p95_ms'],
            'candidate_solve_p95_ms': row['mpc_solve_p95_ms'],
            'baseline_deadline_miss_fraction': ref['mpc_deadline_miss_fraction'],
            'candidate_deadline_miss_fraction': row['mpc_deadline_miss_fraction'],
            'position_rmse_change_mm': row['position_rmse_mm'] - ref['position_rmse_mm'],
            'angle_rms_change_deg': row['angle_rms_deg'] - ref['angle_rms_deg'],
            'peak_rail_fraction_change': row['peak_rail_fraction'] - ref['peak_rail_fraction'],
            'energy_ratio': row['integrated_resonator_energy_mJs'] /
                            max(ref['integrated_resonator_energy_mJs'], 1e-12),
            'reasons': pair_reasons,
        })
        reasons.extend(f'seed {seed}: {reason}' for reason in pair_reasons)

    baseline_p95 = (sum(row['mpc_solve_p95_ms'] for row in baseline.values()) /
                    len(baseline)) if baseline else 0.
    candidate_p95 = (sum(row['mpc_solve_p95_ms'] for row in candidate.values()) /
                     len(candidate)) if candidate else 0.
    speedup = 1. - candidate_p95 / max(baseline_p95, 1e-12)
    if speedup < .10:
        reasons.append('mean per-seed p95 solve latency improved by less than 10%')
    accepted = not reasons
    return {
        'accepted': accepted,
        'selected_stride': 2 if accepted else 1,
        'baseline_mean_seed_p95_ms': baseline_p95,
        'candidate_mean_seed_p95_ms': candidate_p95,
        'p95_speedup_fraction': speedup,
        'pairs': pairs,
        'reasons': reasons,
    }


def compare_imitation_to_mpc(mpc_rows, imitation_rows):
    """Check whether the deterministic student preserves MPC behavior and speed."""
    mpc = {int(row['seed']): row for row in mpc_rows}
    imitation = {int(row['seed']): row for row in imitation_rows}
    performance_reasons = []
    speed_reasons = []
    if set(mpc) != set(imitation) or not mpc:
        performance_reasons.append('MPC and imitation seed coverage differs or is empty')
    pairs = []
    for seed in sorted(set(mpc) & set(imitation)):
        ref, row = mpc[seed], imitation[seed]
        pair_reasons = []
        if row['position_rmse_mm'] > ref['position_rmse_mm'] * 1.02 + .05:
            pair_reasons.append('position tracking exceeded paired MPC bound')
        if row['angle_rms_deg'] > ref['angle_rms_deg'] * 1.02 + .01:
            pair_reasons.append('angle RMS exceeded paired MPC bound')
        if row['peak_angle_deg'] > ref['peak_angle_deg'] * 1.02 + .01:
            pair_reasons.append('peak angle exceeded paired MPC bound')
        if row['peak_rail_fraction'] > ref['peak_rail_fraction'] + .01:
            pair_reasons.append('peak rail fraction exceeded paired MPC bound')
        energy_ratio = row['integrated_resonator_energy_mJs'] / max(
            ref['integrated_resonator_energy_mJs'], 1e-12)
        if energy_ratio > 1.02:
            pair_reasons.append('integrated resonator energy exceeded MPC by more than 2%')
        if (row['terminated'] or row['exclusion_zone_violations'] or
                not row['goal_targets_respect_margin']):
            pair_reasons.append('safety or exclusion-zone gate failed')
        if row['goal_count'] <= 0 or row['completed_two_second_holds'] != row['goal_count']:
            pair_reasons.append('not every two-second goal hold completed')
        if row['saturation_fraction'] > ref['saturation_fraction'] + .02:
            pair_reasons.append('saturation fraction exceeded paired MPC bound')
        pairs.append({
            'seed': seed,
            'mpc_solve_p95_ms': ref['mpc_solve_p95_ms'],
            'mpc_solve_p99_ms': ref['mpc_solve_p99_ms'],
            'mpc_solve_p50_ms': ref['mpc_solve_p50_ms'],
            'mpc_deadline_miss_fraction': ref['mpc_deadline_miss_fraction'],
            'imitation_action_p95_ms': row['policy_action_p95_ms'],
            'imitation_action_p99_ms': row['policy_action_p99_ms'],
            'imitation_action_p50_ms': row['policy_action_p50_ms'],
            'imitation_deadline_miss_fraction': row['policy_deadline_miss_fraction'],
            'position_rmse_change_mm': row['position_rmse_mm'] - ref['position_rmse_mm'],
            'angle_rms_change_deg': row['angle_rms_deg'] - ref['angle_rms_deg'],
            'peak_rail_fraction_change': row['peak_rail_fraction'] - ref['peak_rail_fraction'],
            'energy_ratio': energy_ratio,
            'reasons': pair_reasons,
        })
        performance_reasons.extend(f'seed {seed}: {reason}' for reason in pair_reasons)
    mpc_p95 = (sum(r['mpc_solve_p95_ms'] for r in mpc.values()) / len(mpc)) if mpc else 0.
    imitation_p95 = (sum(r['policy_action_p95_ms'] for r in imitation.values()) /
                     len(imitation)) if imitation else 0.
    speedup = mpc_p95 / max(imitation_p95, 1e-12)
    mpc_p50 = (sum(r['mpc_solve_p50_ms'] for r in mpc.values()) / len(mpc)) if mpc else 0.
    imitation_p50 = (sum(r['policy_action_p50_ms'] for r in imitation.values()) /
                     len(imitation)) if imitation else 0.
    mpc_p99 = (sum(r['mpc_solve_p99_ms'] for r in mpc.values()) / len(mpc)) if mpc else 0.
    imitation_p99 = (sum(r['policy_action_p99_ms'] for r in imitation.values()) /
                     len(imitation)) if imitation else 0.
    mpc_deadlines = (sum(r['mpc_deadline_miss_fraction'] for r in mpc.values()) /
                     len(mpc)) if mpc else 0.
    imitation_deadlines = (sum(r['policy_deadline_miss_fraction'] for r in imitation.values()) /
                           len(imitation)) if imitation else 0.
    if speedup < 2.:
        speed_reasons.append('imitation p95 action latency was not at least 2x faster than MPC p95 solve latency')
    if imitation_p99 >= mpc_p99:
        speed_reasons.append('imitation p99 action latency was not lower than MPC p99 solve latency')
    if imitation_deadlines > mpc_deadlines + .005:
        speed_reasons.append('imitation 10 ms deadline-miss fraction exceeded MPC by more than 0.005')
    return {
        'performance_preserved': not performance_reasons,
        'useful': not performance_reasons and not speed_reasons,
        'mean_mpc_solve_p95_ms': mpc_p95,
        'mean_imitation_action_p95_ms': imitation_p95,
        'p95_latency_speedup': speedup,
        'mean_mpc_solve_p50_ms': mpc_p50,
        'mean_imitation_action_p50_ms': imitation_p50,
        'mean_mpc_solve_p99_ms': mpc_p99,
        'mean_imitation_action_p99_ms': imitation_p99,
        'p99_latency_speedup': mpc_p99 / max(imitation_p99, 1e-12),
        'mean_mpc_deadline_miss_fraction': mpc_deadlines,
        'mean_imitation_deadline_miss_fraction': imitation_deadlines,
        'mean_energy_ratio': (sum(p['energy_ratio'] for p in pairs) / len(pairs)) if pairs else None,
        'max_energy_ratio': max((p['energy_ratio'] for p in pairs), default=None),
        'pairs': pairs,
        'performance_reasons': performance_reasons,
        'speed_reasons': speed_reasons,
        'reasons': performance_reasons + speed_reasons,
    }


def validate_seed_separation(optimization_seeds, evaluation_seeds,
                             teacher_seed, teacher_episodes,
                             confirmation_seeds=None):
    optimization_seeds = [int(seed) for seed in optimization_seeds]
    evaluation_seeds = [int(seed) for seed in evaluation_seeds]
    teacher_seeds = [int(teacher_seed) + 1009 * i for i in range(teacher_episodes)]
    groups = {
        'optimization': optimization_seeds,
        'evaluation': evaluation_seeds,
        'teacher episode': teacher_seeds,
    }
    if confirmation_seeds is not None:
        groups['confirmation'] = [int(seed) for seed in confirmation_seeds]
    for name, seeds in groups.items():
        if not seeds:
            raise ValueError(f'{name} seeds must be nonempty')
        if any(seed < 0 for seed in seeds):
            raise ValueError(f'{name} seeds must be nonnegative')
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'{name} seeds must be unique')
    names = list(groups)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            if set(groups[left_name]) & set(groups[right_name]):
                raise ValueError(f'{left_name} and {right_name} seeds must be disjoint')
    return teacher_seeds


def select_confirmed_stride(initial_decision, confirmation_decision):
    return (2 if initial_decision.get('accepted') and confirmation_decision and
            confirmation_decision.get('accepted') else 1)


def _source_fingerprint(repo_root):
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    paths = ('software/python/constrained_mpc.py',
             'software/python/evaluate_goal_hold_v4.py',
             'software/python/train_residual_v4.py',
             'software/python/benchmark_mpc_imitation.py')
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update((repo_root / relative).read_bytes())
    return {'git_revision': revision, 'source_sha256': digest.hexdigest(),
            'source_files': list(paths)}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _write_teacher_config(outdir, cfg, mpc_config, base_params):
    path = outdir / 'resolved_teacher_config.json'
    _write_json(path, {
        'schema_version': 1,
        'env_config': asdict(cfg),
        'mpc_config': asdict(mpc_config),
        'plant_parameters': asdict(base_params),
    })
    return path


def _json_ready(value):
    return json.loads(json.dumps(value))


def _assert_teacher_config_matches(teacher_config, cfg, mpc_config, base_params):
    expected = {
        'env_config': _json_ready(asdict(cfg)),
        'mpc_config': _json_ready(asdict(mpc_config)),
        'plant_parameters': _json_ready(asdict(base_params)),
    }
    mismatches = [key for key, value in expected.items()
                  if teacher_config.get(key) != value]
    if mismatches:
        raise RuntimeError('Teacher config differs from selected benchmark settings: ' +
                           ', '.join(mismatches))


def _write_markdown(path, data):
    optimization = data['optimization']
    imitation = data.get('imitation')
    lines = [
        '# MPC Runtime and Imitation Evaluation', '',
        f"- Source revision: `{data['source']['git_revision']}`",
        f"- Source fingerprint: `{data['source']['source_sha256']}`",
        f"- Task: random target holds, 15 mm edge exclusion, at least 2.5 s training holds and 2 s evaluated in-tolerance holds.",
        f"- Control period: {data['config']['control_dt'] * 1000:.1f} ms; actuator: `{data['config']['actuator_mode']}`.",
        '', '## MPC optimization', '',
        f"- Decision: {'selected stride 2' if optimization['accepted'] else 'kept stride 1'}.",
        f"- Mean per-seed p95 solve time: {optimization['baseline_mean_seed_p95_ms']:.3f} ms at stride 1, {optimization['candidate_mean_seed_p95_ms']:.3f} ms at stride 2.",
        f"- p95 improvement: {optimization['p95_speedup_fraction'] * 100:.1f}%.",
        f"- Paired behavior gate: {'passed' if optimization['accepted'] else 'failed'}.",
        '', '| Seed | stride 1 p95 (ms) | stride 2 p95 (ms) | position Δ (mm) | angle RMS Δ (°) | peak rail Δ | energy ratio | deadline miss Δ |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    candidate_rows = {r['seed']: r for r in data['optimization_candidate_rows']}
    baseline_rows = {r['seed']: r for r in data['optimization_baseline_rows']}
    for pair in optimization['pairs']:
        ref = baseline_rows[pair['seed']]
        row = candidate_rows[pair['seed']]
        lines.append(
            f"| {pair['seed']} | {ref['mpc_solve_p95_ms']:.3f} | {row['mpc_solve_p95_ms']:.3f} | "
            f"{pair['position_rmse_change_mm']:+.4f} | {pair['angle_rms_change_deg']:+.5f} | "
            f"{pair['peak_rail_fraction_change']:+.4f} | {pair['energy_ratio']:.4f} | "
            f"{row['mpc_deadline_miss_fraction'] - ref['mpc_deadline_miss_fraction']:+.4f} |")
    confirmation = data.get('confirmation')
    if confirmation:
        decision = confirmation.get('decision')
        if decision is not None:
            lines.extend(['', f"- Fresh paired confirmation on seeds {', '.join(map(str, confirmation['seeds']))}: "
                          f"{'passed' if decision['accepted'] else 'failed'}; "
                          f"p95 improvement {decision['p95_speedup_fraction'] * 100:.1f}%."])
        else:
            rows = confirmation.get('selected_rows', [])
            mean_p95 = (sum(row['mpc_solve_p95_ms'] for row in rows) / len(rows)) if rows else 0.
            lines.extend(['', f"- Fresh rerun of selected stride 1 on seeds {', '.join(map(str, confirmation['seeds']))}: "
                          f"mean p95 {mean_p95:.3f} ms; stride 2 did not pass the initial paired gate."])
    if optimization['reasons']:
        lines.extend(['', 'Reasons for retaining stride 1:', ''])
        lines.extend(f'- {reason}' for reason in optimization['reasons'])
    if imitation is not None:
        fit = data['training']['teacher_fit']
        lines.extend([
            '', '## MPC imitation', '',
            f"- Best held-out action MAE: {fit['best_heldout_action_mae']:.6f} normalized action units.",
            f"- Successful labels used: {fit['train_samples'] + fit['validation_samples']}; fallback labels excluded: {fit['excluded_fallback_labels']}.",
            f"- Mean MPC p50/p95/p99 solve time: {imitation['mean_mpc_solve_p50_ms']:.3f}/{imitation['mean_mpc_solve_p95_ms']:.3f}/{imitation['mean_mpc_solve_p99_ms']:.3f} ms.",
            f"- Mean imitation p50/p95/p99 action time: {imitation['mean_imitation_action_p50_ms']:.4f}/{imitation['mean_imitation_action_p95_ms']:.4f}/{imitation['mean_imitation_action_p99_ms']:.4f} ms; p95/p99 speedup: {imitation['p95_latency_speedup']:.1f}×/{imitation['p99_latency_speedup']:.1f}×.",
            f"- Mean 10 ms deadline misses: MPC {imitation['mean_mpc_deadline_miss_fraction']:.4f}, imitation {imitation['mean_imitation_deadline_miss_fraction']:.4f}.",
            f"- Closed-loop performance gates: {'passed' if imitation['performance_preserved'] else 'not all passed'}.",
            f"- Speed/performance usefulness gate: {'passed' if imitation['useful'] else 'not passed'}.",
            f"- Mean integrated resonator energy ratio vs MPC: {imitation['mean_energy_ratio']:.4f}.",
            '', '| Seed | MPC p50/p95/p99 (ms) | imitation p50/p95/p99 (ms) | position Δ (mm) | angle RMS Δ (°) | rail Δ | energy ratio |',
            '|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for pair in imitation['pairs']:
            lines.append(
                f"| {pair['seed']} | {pair['mpc_solve_p50_ms']:.3f}/{pair['mpc_solve_p95_ms']:.3f}/{pair['mpc_solve_p99_ms']:.3f} | "
                f"{pair['imitation_action_p50_ms']:.4f}/{pair['imitation_action_p95_ms']:.4f}/{pair['imitation_action_p99_ms']:.4f} | "
                f"{pair['position_rmse_change_mm']:+.4f} | {pair['angle_rms_change_deg']:+.5f} | "
                f"{pair['peak_rail_fraction_change']:+.4f} | {pair['energy_ratio']:.4f} |")
        if imitation['reasons']:
            lines.extend(['', 'Gate findings:', ''])
            lines.extend(f'- {reason}' for reason in imitation['reasons'])
    lines.extend([
        '', '## Artifacts', '',
        f"- Full JSON evidence: `{data['evidence_json']}`",
        f"- Run directory: `{data['run_directory']}`",
        '- Timing measurements are Python simulation-host CPU measurements; they do not certify deployment-hardware real-time latency.',
        '- Energy is integrated resonator mechanical energy, not motor electrical consumption.',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _prepare_environment_config(config_checkpoint):
    _, cfg, checkpoint = load_policy(config_checkpoint)
    cfg = replace(cfg, domain_randomization=True,
                  initial_theta_std=math.radians(1.),
                  initial_theta_dot_std=math.radians(5.),
                  kick_probability=1., reference_mode='random_goal_hold',
                  goal_edge_margin_m=.015, goal_hold_min_seconds=2.5)
    return cfg, checkpoint


def _run_mpc_seeds(cfg, seeds, horizon, stride, tolerance_mm, base_params, label):
    mpc_config = goal_hold_mpc_config(cfg, horizon, stride)
    rows = []
    for seed in seeds:
        row = run_case('mpc', seed, cfg, horizon=horizon,
                       tolerance_mm=tolerance_mm, base_params=base_params,
                       mpc_config=mpc_config)
        rows.append(row)
        print('EVENT ' + json.dumps({'phase': label, 'seed': seed,
              'p95_ms': row['mpc_solve_p95_ms'],
              'p99_ms': row['mpc_solve_p99_ms'],
              'deadline_misses': row['mpc_deadline_miss_fraction'],
              'fallback': row['mpc_fallback_fraction'],
              'holds': row['completed_two_second_holds'],
              'goals': row['goal_count']}), flush=True)
    return rows, mpc_config


def _training_command(args, cfg, outdir, mpc_config, base_params):
    script = Path(__file__).resolve().with_name('train_residual_v4.py')
    config_path = _write_teacher_config(outdir, cfg, mpc_config, base_params)
    command = [sys.executable, str(script), 'teacher',
               '--resolved-teacher-config', str(config_path),
               '--samples', str(args.samples), '--episodes', str(args.episodes),
               '--epochs', str(args.epochs), '--seed', str(args.teacher_seed),
               '--outdir', str(outdir),
               '--skip-evaluation']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-model', type=Path, required=True,
                        help='Random-goal-hold checkpoint used to load the evaluation observation/config schema')
    parser.add_argument('--outdir', type=Path,
                        default=Path('runs/phase6/mpc_runtime_imitation_20261007'))
    parser.add_argument('--evidence-json', type=Path,
                        default=Path('docs/control/evidence/mpc_runtime_imitation_20261007.json'))
    parser.add_argument('--report', type=Path,
                        default=Path('docs/control/evidence/mpc_runtime_imitation_20261007.md'))
    parser.add_argument('--optimization-seeds', nargs='*', type=int,
                        default=[7001, 7002, 7003, 7004, 7005])
    parser.add_argument('--evaluation-seeds', nargs='*', type=int,
                        default=[8011, 8012, 8013, 8014, 8015, 8016, 8017, 8018, 8019, 8020])
    parser.add_argument('--confirmation-seeds', nargs='*', type=int,
                        default=[9001, 9002, 9003])
    parser.add_argument('--teacher-seed', type=int, default=4242)
    parser.add_argument('--samples', type=int, default=9600)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--tolerance-mm', type=float, default=5.)
    parser.add_argument('--plant-overrides', type=Path,
                        help='Optional JSON mapping to evaluate a specific geometry; default is nominal geometry')
    args = parser.parse_args()

    if args.samples < args.episodes or args.episodes < 4 or args.epochs < 1:
        raise ValueError('Use at least four teacher episodes, samples >= episodes, and positive epochs')
    teacher_episode_seeds = validate_seed_separation(
        args.optimization_seeds, args.evaluation_seeds, args.teacher_seed,
        args.episodes, args.confirmation_seeds)
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = args.outdir if args.outdir.is_absolute() else repo_root / args.outdir
    evidence_json = args.evidence_json if args.evidence_json.is_absolute() else repo_root / args.evidence_json
    report = args.report if args.report.is_absolute() else repo_root / args.report
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    cfg, config_checkpoint = _prepare_environment_config(args.config_model)
    plant_overrides = (json.loads(args.plant_overrides.read_text(encoding='utf-8'))
                       if args.plant_overrides else {})
    if not isinstance(plant_overrides, dict):
        raise ValueError('--plant-overrides must contain a JSON object')
    base_params = PlantParams(**plant_overrides) if plant_overrides else PlantParams()
    source = _source_fingerprint(repo_root)
    manifest = {
        'status': 'running', 'source': source,
        'config_checkpoint': str(args.config_model.resolve()),
        'environment_config': asdict(cfg), 'plant_parameters': asdict(base_params),
        'plant_overrides': plant_overrides,
        'optimization_seeds': args.optimization_seeds,
        'confirmation_seeds': args.confirmation_seeds,
        'evaluation_seeds': args.evaluation_seeds,
        'teacher_seed': args.teacher_seed,
        'teacher_episode_seeds': teacher_episode_seeds,
        'teacher_samples': args.samples, 'teacher_episodes': args.episodes,
        'teacher_epochs': args.epochs, 'horizon': args.horizon,
        'control_period_ms': cfg.control_dt * 1000.,
        'optimization_candidate': 'torque linearization stride 2; compare against stride 1',
    }
    _write_json(run_dir / 'run_manifest.json', manifest)

    print('EVENT ' + json.dumps({'phase': 'mpc_optimization', 'candidate_strides': [1, 2],
          'optimization_seeds': args.optimization_seeds}), flush=True)
    baseline_rows, baseline_config = _run_mpc_seeds(
        cfg, args.optimization_seeds, args.horizon, 1, args.tolerance_mm,
        base_params, 'mpc_stride_1')
    candidate_rows, candidate_config = _run_mpc_seeds(
        cfg, args.optimization_seeds, args.horizon, 2, args.tolerance_mm,
        base_params, 'mpc_stride_2')
    optimization = judge_mpc_optimization(baseline_rows, candidate_rows)
    initial_accepted = optimization['accepted']
    initial_selected_stride = optimization['selected_stride']
    print('EVENT ' + json.dumps({'phase': 'optimization_decision',
          'accepted': initial_accepted, 'selected_stride': initial_selected_stride,
          'p95_speedup_fraction': optimization['p95_speedup_fraction'],
          'reasons': optimization['reasons']}), flush=True)

    confirmation = {'seeds': args.confirmation_seeds}
    if initial_accepted:
        print('EVENT ' + json.dumps({'phase': 'mpc_confirmation',
              'seeds': args.confirmation_seeds, 'candidate_strides': [1, 2]}), flush=True)
        confirmation_baseline_rows, _ = _run_mpc_seeds(
            cfg, args.confirmation_seeds, args.horizon, 1, args.tolerance_mm,
            base_params, 'confirmation_stride_1')
        confirmation_candidate_rows, _ = _run_mpc_seeds(
            cfg, args.confirmation_seeds, args.horizon, 2, args.tolerance_mm,
            base_params, 'confirmation_stride_2')
        confirmation_decision = judge_mpc_optimization(
            confirmation_baseline_rows, confirmation_candidate_rows)
        selected_stride = select_confirmed_stride(optimization, confirmation_decision)
        confirmation.update({
            'baseline_rows': confirmation_baseline_rows,
            'candidate_rows': confirmation_candidate_rows,
            'decision': confirmation_decision,
        })
        if not confirmation_decision['accepted']:
            optimization['reasons'].extend(
                'confirmation: ' + reason for reason in confirmation_decision['reasons'])
    else:
        confirmation_selected_rows, _ = _run_mpc_seeds(
            cfg, args.confirmation_seeds, args.horizon, 1, args.tolerance_mm,
            base_params, 'confirmation_selected_stride_1')
        selected_stride = 1
        confirmation.update({'selected_rows': confirmation_selected_rows,
                             'decision': None})
    optimization['initial_accepted'] = initial_accepted
    optimization['initial_selected_stride'] = initial_selected_stride
    optimization['selected_stride'] = selected_stride
    optimization['accepted'] = bool(initial_accepted and selected_stride == 2)
    selected_mpc_config = candidate_config if selected_stride == 2 else baseline_config
    _write_json(run_dir / 'optimization.json', {
        'baseline_mpc_config': asdict(baseline_config),
        'candidate_mpc_config': asdict(candidate_config),
        'baseline_rows': baseline_rows, 'candidate_rows': candidate_rows,
        'decision': optimization, 'confirmation': confirmation,
    })

    teacher_dir = run_dir / 'teacher'
    command = _training_command(args, cfg, teacher_dir, selected_mpc_config, base_params)
    manifest['teacher_command'] = command
    manifest['selected_stride'] = selected_stride
    _write_json(run_dir / 'run_manifest.json', manifest)
    print('EVENT ' + json.dumps({'phase': 'teacher_training_start',
          'selected_stride': selected_stride, 'samples': args.samples,
          'episodes': args.episodes, 'epochs': args.epochs}), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)

    teacher_fit = json.loads((teacher_dir / 'teacher_fit.json').read_text(encoding='utf-8'))
    teacher_config = json.loads((teacher_dir / 'teacher_config.json').read_text(encoding='utf-8'))
    _assert_teacher_config_matches(teacher_config, cfg, selected_mpc_config, base_params)
    model_path = teacher_dir / 'policy_candidate.pt'
    model, eval_cfg, checkpoint = load_policy(model_path)
    if _json_ready(asdict(eval_cfg)) != _json_ready(asdict(cfg)):
        raise RuntimeError('Saved student environment config differs from the selected teacher task')
    eval_cfg = replace(eval_cfg, domain_randomization=True,
                       initial_theta_std=math.radians(1.),
                       initial_theta_dot_std=math.radians(5.),
                       kick_probability=1.)
    evaluation_mpc_config = MPCConfig(**teacher_config['mpc_config'])
    mpc_rows = []
    imitation_rows = []
    lqr_rows = []
    print('EVENT ' + json.dumps({'phase': 'heldout_evaluation_start',
          'evaluation_seeds': args.evaluation_seeds,
          'linearization_stride': evaluation_mpc_config.linearization_stride}), flush=True)
    for seed in args.evaluation_seeds:
        for controller in ('lqr', 'mpc', 'policy'):
            row = run_case(
                controller, seed, eval_cfg, model=model, horizon=args.horizon,
                tolerance_mm=args.tolerance_mm, base_params=base_params,
                mpc_config=evaluation_mpc_config if controller == 'mpc' else None)
            if controller == 'lqr':
                lqr_rows.append(row)
            elif controller == 'mpc':
                mpc_rows.append(row)
            else:
                imitation_rows.append(row)
            print('EVENT ' + json.dumps({'phase': 'heldout_evaluation',
                  'controller': 'imitation' if controller == 'policy' else controller,
                  'seed': seed,
                  'mpc_p95_ms': row['mpc_solve_p95_ms'],
                  'imitation_p95_ms': row['policy_action_p95_ms'],
                  'energy_mjs': row['integrated_resonator_energy_mJs'],
                  'position_rmse_mm': row['position_rmse_mm'],
                  'holds': row['completed_two_second_holds'],
                  'goals': row['goal_count']}), flush=True)

    mpc_imitation_decision = compare_imitation_to_mpc(mpc_rows, imitation_rows)
    _, standard_summary = summarize(lqr_rows + mpc_rows + imitation_rows)
    data = {
        'status': 'completed', 'source': source,
        'run_directory': str(run_dir), 'evidence_json': str(evidence_json),
        'config': asdict(eval_cfg), 'plant_parameters': asdict(base_params),
        'plant_overrides': plant_overrides,
        'training': {
            'teacher_config': teacher_config,
            'teacher_fit': teacher_fit,
            'teacher_checkpoint': str(model_path),
            'teacher_checkpoint_source_revision': checkpoint.get('extra', {}).get('source_revision'),
            'teacher_command': command,
            'teacher_fallback_labels_excluded': teacher_fit['excluded_fallback_labels'],
        },
        'optimization': optimization,
        'confirmation': confirmation,
        'optimization_baseline_rows': baseline_rows,
        'optimization_candidate_rows': candidate_rows,
        'selected_mpc_config': asdict(evaluation_mpc_config),
        'confirmation_seeds': args.confirmation_seeds,
        'evaluation_seeds': args.evaluation_seeds,
        'evaluation_rows': lqr_rows + mpc_rows + imitation_rows,
        'evaluation_summary': standard_summary,
        'imitation': mpc_imitation_decision,
        'interpretation': {
            'energy_metric': 'integrated resonator mechanical energy in mJ; not motor electrical consumption',
            'latency_scope': 'Python process controller-call timing on the simulation host; not deployment-hardware timing',
            'student_output': 'deterministic next residual action; no future state or trajectory prediction',
        },
    }
    _write_json(run_dir / 'summary.json', data)
    _write_json(evidence_json, data)
    _write_markdown(run_dir / 'report.md', data)
    _write_markdown(report, data)
    manifest.update({
        'status': 'completed',
        'selected_mpc_config': asdict(evaluation_mpc_config),
        'teacher_checkpoint': str(model_path),
        'evidence_json': str(evidence_json),
        'report': str(report),
        'imitation_useful': mpc_imitation_decision['useful'],
    })
    _write_json(run_dir / 'run_manifest.json', manifest)
    print('RESULT ' + json.dumps({
        'optimization_accepted': optimization['accepted'],
        'selected_stride': selected_stride,
        'mpc_p95_speedup': optimization['p95_speedup_fraction'],
        'imitation_useful': mpc_imitation_decision['useful'],
        'imitation_p95_speedup': mpc_imitation_decision['p95_latency_speedup'],
        'evidence_json': str(evidence_json), 'report': str(report),
        'run_directory': str(run_dir),
    }), flush=True)


if __name__ == '__main__':
    main()

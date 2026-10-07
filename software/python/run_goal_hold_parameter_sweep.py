"""Run a bounded overnight PPO/geometry sweep on randomized goal-and-hold tasks.

Each candidate starts from a fixed saved policy (or a declared fresh
initialization), is trained on one stable plant geometry, and is paired against
LQR, the repository MPC, and the incumbent PPO policy on common evaluation
seeds. A separate final seed set is used for the strongest candidates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time

from active_vibration_rig_2d import PlantParams


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'software' / 'python' / 'train_residual_v4.py'
EVALUATE = ROOT / 'software' / 'python' / 'evaluate_goal_hold_v4.py'
INCUMBENT = ROOT / 'runs' / 'phase5' / 'random_goal_hold_20261006' / 'attempt_07' / 'policy_candidate.pt'
PRIOR = ROOT / 'runs' / 'phase5' / 'random_goal_hold_20261006' / 'attempt_03' / 'policy_candidate.pt'

GEOMETRIES = [
    ('nominal', {}),
    ('short_lever', {'lever_com_distance': 0.090, 'lever_inertia_com': 1.62e-4}),
    ('long_lever', {'lever_com_distance': 0.100, 'lever_inertia_com': 1.994e-4}),
    ('short_light', {'lever_com_distance': 0.090, 'lever_inertia_com': 1.62e-4,
                     'resonator_mass': 0.108}),
    ('long_heavy', {'lever_com_distance': 0.100, 'lever_inertia_com': 1.994e-4,
                    'resonator_mass': 0.132}),
    ('softer_torsion', {'k_theta': 0.390}),
    ('stiffer_torsion', {'k_theta': 0.450}),
]

RECIPES = [
    {'name': 'incumbent_recipe', 'init': 'incumbent', 'learning_rate': 1e-4,
     'anchor_kl': .001, 'energy_gate_weight': 64, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .0001, 'symmetry_coef': .025, 'rollout': 256},
    {'name': 'larger_update', 'init': 'incumbent', 'learning_rate': 3e-4,
     'anchor_kl': .001, 'energy_gate_weight': 64, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .0001, 'symmetry_coef': .025, 'rollout': 256},
    {'name': 'low_anchor', 'init': 'incumbent', 'learning_rate': 3e-4,
     'anchor_kl': .0002, 'energy_gate_weight': 96, 'energy_gate_step_scale_mjs': .75,
     'entropy_coef': .001, 'symmetry_coef': .01, 'rollout': 256},
    {'name': 'energy_focused', 'init': 'incumbent', 'learning_rate': 2e-4,
     'anchor_kl': .0003, 'energy_gate_weight': 160, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .001, 'symmetry_coef': 0., 'rollout': 256},
    {'name': 'wide_energy_gate', 'init': 'incumbent', 'learning_rate': 1e-4,
     'anchor_kl': .00025, 'energy_gate_weight': 192, 'energy_gate_step_scale_mjs': 2.,
     'entropy_coef': .0005, 'symmetry_coef': .005, 'rollout': 256},
    {'name': 'high_exploration', 'init': 'incumbent', 'learning_rate': 5e-4,
     'anchor_kl': .0001, 'energy_gate_weight': 128, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .002, 'symmetry_coef': 0., 'rollout': 256},
    {'name': 'conservative_energy', 'init': 'incumbent', 'learning_rate': 5e-5,
     'anchor_kl': .002, 'energy_gate_weight': 96, 'energy_gate_step_scale_mjs': .5,
     'entropy_coef': .0001, 'symmetry_coef': .025, 'rollout': 256},
    {'name': 'prior_checkpoint', 'init': 'prior', 'learning_rate': 3e-4,
     'anchor_kl': .001, 'energy_gate_weight': 96, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .001, 'symmetry_coef': .01, 'rollout': 256},
    {'name': 'fresh_policy', 'init': 'scratch', 'learning_rate': 3e-4,
     'anchor_kl': .001, 'energy_gate_weight': 96, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .001, 'symmetry_coef': .025, 'rollout': 256},
    {'name': 'longer_rollout', 'init': 'incumbent', 'learning_rate': 2e-4,
     'anchor_kl': .0005, 'energy_gate_weight': 160, 'energy_gate_step_scale_mjs': 1.,
     'entropy_coef': .0005, 'symmetry_coef': .01, 'rollout': 512},
]

DEV_SEEDS = [68011, 68012, 68013, 68014, 68015]
FINAL_SEEDS = [69201, 69202, 69203, 69204, 69205, 69206, 69207, 69208]
TRAIN_TIMEOUT_SECONDS = 12 * 60
EVAL_TIMEOUT_SECONDS = 10 * 60
FINAL_RESERVE_SECONDS = 65 * 60


def emit(stream, event, **fields):
    payload = {'event': event, 'utc': datetime.now(timezone.utc).isoformat(), **fields}
    line = json.dumps(payload, sort_keys=True)
    stream.write(line + '\n')
    stream.flush()
    print(line, flush=True)


def run_logged(command, log_path, label, timeout_seconds, stream):
    """Run a child while streaming output into both a durable log and CPJ."""
    emit(stream, 'process_started', label=label, command=[str(x) for x in command],
         timeout_seconds=timeout_seconds)
    process = subprocess.Popen(
        [str(x) for x in command], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    lines = queue.Queue()

    def read_output():
        try:
            for child_line in process.stdout:
                lines.put(child_line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, name=f'{label}-output', daemon=True)
    reader.start()
    end = time.monotonic() + timeout_seconds
    timed_out = False
    with log_path.open('w', encoding='utf-8', newline='') as log:
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0 and process.poll() is None:
                timed_out = True
                process.kill()
                emit(stream, 'process_timeout', label=label, timeout_seconds=timeout_seconds)
            try:
                child_line = lines.get(timeout=max(.1, min(30., max(remaining, .1))))
            except queue.Empty:
                emit(stream, 'process_still_running', label=label)
                continue
            if child_line is None:
                break
            log.write(child_line)
            log.flush()
            print(child_line, end='', flush=True)
    return_code = process.wait()
    reader.join(timeout=2.)
    return return_code, timed_out


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_fingerprint():
    files = [TRAIN, ROOT / 'software' / 'python' / 'rig_rl_env_v4.py',
             ROOT / 'software' / 'python' / 'evaluate_v4.py', EVALUATE, Path(__file__)]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def geometry_params(overrides):
    params = PlantParams(**overrides)
    stiffness = params.effective_small_angle_stiffness
    if (not math.isfinite(stiffness) or stiffness < .12 or params.use_geometric_springs):
        raise ValueError(f'Unstable or unsupported geometry: {overrides}; k_eff={stiffness}')
    return params


def command_for_eval(model_path, json_path, seeds, geometry_path):
    command = [sys.executable, EVALUATE, '--model', model_path, '--json', json_path,
               '--seeds', *map(str, seeds)]
    if geometry_path:
        command.extend(['--plant-overrides', geometry_path])
    return command


def eval_summary(model_path, output_path, log_path, seeds, geometry_path, label, stream):
    command = command_for_eval(model_path, output_path, seeds, geometry_path)
    code, timed_out = run_logged(command, log_path, label, EVAL_TIMEOUT_SECONDS, stream)
    if timed_out or not output_path.exists():
        raise RuntimeError(f'{label} evaluation did not produce JSON (exit={code}, timeout={timed_out})')
    data = read_json(output_path)
    data['process_exit_code'] = code
    return data


def candidate_rank(item):
    decision = item.get('promotion', {})
    summary = item.get('summary', {})
    ratio = decision.get('policy_mpc_integrated_resonator_energy_ratio', float('inf'))
    return (not decision.get('accepted_vs_mpc', False), ratio,
            not decision.get('accepted_vs_lqr', False),
            summary.get('policy', {}).get('mean_position_rmse_mm', float('inf')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=8.,
                        help='Bounded sweep duration; must be between six and ten hours')
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if not 6. <= args.hours <= 10.:
        raise ValueError('--hours must be between 6 and 10')
    if args.steps < 2048:
        raise ValueError('--steps must be at least 2048')
    if not TRAIN.exists() or not EVALUATE.exists():
        raise FileNotFoundError('Training/evaluation entrypoints are missing')
    if not INCUMBENT.exists() or not PRIOR.exists():
        raise FileNotFoundError(f'Expected saved start checkpoints: {INCUMBENT} and {PRIOR}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_root = (args.run_dir if args.run_dir else
                ROOT / 'runs' / 'phase5' / 'random_goal_hold_sweep_20261006' / f'overnight_{stamp}')
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    geometry_root = out_root / 'geometries'
    geometry_root.mkdir()
    event_path = out_root / 'events.jsonl'
    run_started = time.monotonic()
    wall_seconds = args.hours * 3600.
    stop_new_trials_at = run_started + max(6 * 3600., wall_seconds - FINAL_RESERVE_SECONDS)
    seed_rng = random.Random(20261006)

    geometry_files = {}
    for geometry_name, overrides in GEOMETRIES:
        params = geometry_params(overrides)
        geometry_path = geometry_root / f'{geometry_name}.json'
        geometry_path.write_text(json.dumps(overrides, indent=2), encoding='utf-8')
        geometry_files[geometry_name] = str(geometry_path)

    git_head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        'benchmark': 'SEDP-V4-RANDOM-GOAL-HOLD',
        'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'planned_wall_hours': args.hours,
        'minimum_wall_hours': 6.,
        'training_steps_per_candidate': args.steps,
        'training_task': {'reference_mode': 'random_goal_hold', 'episode_seconds': 12.,
                          'goal_edge_margin_mm': 15., 'goal_move_seconds': 1.,
                          'goal_hold_min_seconds': 2.5, 'goal_hold_extra_seconds': .5},
        'incumbent_checkpoint': str(INCUMBENT),
        'prior_checkpoint': str(PRIOR),
        'development_seeds': DEV_SEEDS,
        'final_holdout_seeds': FINAL_SEEDS,
        'geometry_overrides': {name: overrides for name, overrides in GEOMETRIES},
        'parameter_recipes': RECIPES,
        'source_git_head': git_head,
        'source_fingerprints_sha256': source_fingerprint(),
        'output_directory': str(out_root),
    }
    manifest_path = out_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    results = []
    failures = []

    with event_path.open('a', encoding='utf-8', buffering=1) as events:
        emit(events, 'sweep_started', output_directory=str(out_root),
             planned_wall_hours=args.hours, geometry_count=len(GEOMETRIES),
             recipe_count=len(RECIPES), steps=args.steps)

        # Paired incumbent measurements are recorded for every geometry before
        # candidate screening, making “materially better” directly measurable.
        geometry_baselines = {}
        for geometry_name, overrides in GEOMETRIES:
            geometry_path = geometry_files[geometry_name] if overrides else None
            baseline_path = out_root / 'geometry_baselines' / geometry_name / 'incumbent_dev.json'
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = eval_summary(INCUMBENT, baseline_path,
                                    baseline_path.with_suffix('.log'), DEV_SEEDS,
                                    geometry_path, f'incumbent-{geometry_name}', events)
                geometry_baselines[geometry_name] = data
                emit(events, 'incumbent_geometry_measured', geometry=geometry_name,
                     policy_mpc_energy_ratio=data['promotion'].get(
                         'policy_mpc_integrated_resonator_energy_ratio'))
            except Exception as exc:
                failures.append({'phase': 'incumbent_baseline', 'geometry': geometry_name,
                                 'error': repr(exc)})
                emit(events, 'incumbent_geometry_failed', geometry=geometry_name, error=repr(exc))
        (out_root / 'geometry_baselines.json').write_text(
            json.dumps({k: v.get('summary', {}) for k, v in geometry_baselines.items()}, indent=2),
            encoding='utf-8')

        trial_index = 0
        while time.monotonic() < stop_new_trials_at:
            queue_items = [(geometry, recipe) for geometry, _ in GEOMETRIES
                           for recipe in RECIPES]
            seed_rng.shuffle(queue_items)
            pass_id = trial_index // len(queue_items) + 1
            for geometry_name, recipe in queue_items:
                if time.monotonic() >= stop_new_trials_at:
                    break
                trial_index += 1
                seed = 31000 + trial_index * 37 + seed_rng.randrange(1, 100000)
                trial_name = f'trial_{trial_index:04d}_pass{pass_id}_{geometry_name}_{recipe["name"]}'
                trial_dir = out_root / 'trials' / trial_name
                trial_dir.mkdir(parents=True, exist_ok=False)
                geometry_path = geometry_files[geometry_name]
                geometry_params_for_trial = dict(recipe)
                geometry_params_for_trial.update(geometry=geometry_name, seed=seed,
                                                  trial=trial_index, pass_id=pass_id)
                (trial_dir / 'trial.json').write_text(
                    json.dumps(geometry_params_for_trial, indent=2), encoding='utf-8')
                emit(events, 'candidate_started', trial=trial_index, pass_id=pass_id,
                     geometry=geometry_name, recipe=recipe['name'], seed=seed,
                     init=recipe['init'], steps=args.steps)

                train_command = [sys.executable, TRAIN, 'ppo', '--outdir', trial_dir,
                    '--steps', str(args.steps), '--envs', '8', '--rollout', str(recipe['rollout']),
                    '--seed', str(seed), '--learning-rate', str(recipe['learning_rate']),
                    '--anchor-kl', str(recipe['anchor_kl']), '--energy-gate-weight',
                    str(recipe['energy_gate_weight']), '--energy-gate-step-scale-mjs',
                    str(recipe['energy_gate_step_scale_mjs']), '--entropy-coef',
                    str(recipe['entropy_coef']), '--symmetry-coef', str(recipe['symmetry_coef']),
                    '--reference-mode', 'random_goal_hold', '--episode-seconds', '12',
                    '--goal-edge-margin-mm', '15', '--goal-move-seconds', '1',
                    '--goal-hold-min-seconds', '2.5', '--goal-hold-extra-seconds', '.5',
                    '--plant-overrides', geometry_path, '--skip-evaluation']
                init_kind = recipe['init']
                if init_kind == 'incumbent':
                    train_command.extend(['--init', INCUMBENT])
                elif init_kind == 'prior':
                    train_command.extend(['--init', PRIOR])
                elif init_kind != 'scratch':
                    raise ValueError(f'Unknown initialization recipe: {init_kind}')

                try:
                    train_code, train_timeout = run_logged(
                        train_command, trial_dir / 'training.log', f'train-{trial_index}',
                        TRAIN_TIMEOUT_SECONDS, events)
                    checkpoint = trial_dir / 'policy_candidate.pt'
                    if train_timeout or train_code != 0 or not checkpoint.exists():
                        raise RuntimeError(f'training failed: exit={train_code}, timeout={train_timeout}')
                    eval_path = trial_dir / 'evaluation_dev.json'
                    eval_data = eval_summary(checkpoint, eval_path,
                        trial_dir / 'evaluation_dev.log', DEV_SEEDS, geometry_path,
                        f'eval-{trial_index}', events)
                    summary = eval_data['summary']
                    incumbent_data = geometry_baselines.get(geometry_name)
                    incumbent_energy = (incumbent_data or {}).get('summary', {}).get(
                        'policy', {}).get('mean_integrated_resonator_energy_mJs')
                    candidate_energy = summary.get('policy', {}).get(
                        'mean_integrated_resonator_energy_mJs')
                    incumbent_ratio = (candidate_energy / max(incumbent_energy, 1e-9)
                                       if incumbent_energy is not None and candidate_energy is not None
                                       else None)
                    item = {
                        'trial': trial_index, 'pass_id': pass_id, 'geometry': geometry_name,
                        'recipe': recipe['name'], 'seed': seed, 'checkpoint': str(checkpoint),
                        'evaluation': str(eval_path), 'summary': summary,
                        'promotion': eval_data['promotion'],
                        'energy_ratio_vs_incumbent_policy': incumbent_ratio,
                    }
                    results.append(item)
                    (trial_dir / 'result.json').write_text(json.dumps(item, indent=2),
                                                          encoding='utf-8')
                    emit(events, 'candidate_finished', trial=trial_index,
                         geometry=geometry_name, recipe=recipe['name'],
                         policy_mpc_ratio=summary.get(
                             'policy_mpc_integrated_resonator_energy_ratio'),
                         policy_lqr_ratio=summary.get(
                             'policy_lqr_integrated_resonator_energy_ratio'),
                         energy_ratio_vs_incumbent=incumbent_ratio,
                         accepted_vs_mpc=eval_data['promotion'].get('accepted_vs_mpc'))
                except Exception as exc:
                    failure = {'trial': trial_index, 'geometry': geometry_name,
                               'recipe': recipe['name'], 'seed': seed, 'error': repr(exc)}
                    failures.append(failure)
                    (trial_dir / 'failure.json').write_text(json.dumps(failure, indent=2),
                                                           encoding='utf-8')
                    emit(events, 'candidate_failed', **failure)
                manifest['completed_candidates'] = len(results)
                manifest['failed_candidates'] = len(failures)
                manifest['elapsed_seconds'] = time.monotonic() - run_started
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                (out_root / 'screening_summary.json').write_text(json.dumps({
                    'completed_candidates': len(results), 'failed_candidates': len(failures),
                    'top_candidates': sorted(results, key=candidate_rank)[:20],
                    'failures': failures[-20:],
                }, indent=2), encoding='utf-8')

        # Confirm the two strongest distinct-geometry candidates on new seeds,
        # and pair each with the incumbent checkpoint on exactly the same set.
        finalists = []
        seen = set()
        for item in sorted(results, key=candidate_rank):
            if item['geometry'] in seen:
                continue
            seen.add(item['geometry'])
            finalists.append(item)
            if len(finalists) == 2:
                break
        final_results = []
        incumbent_final = {}
        for rank, item in enumerate(finalists, start=1):
            geometry_name = item['geometry']
            geometry_path = geometry_files[geometry_name]
            final_dir = out_root / 'finalists' / f'{rank:02d}_{item["trial"]}_{geometry_name}'
            final_dir.mkdir(parents=True, exist_ok=True)
            try:
                candidate_eval = eval_summary(Path(item['checkpoint']),
                    final_dir / 'candidate_holdout.json', final_dir / 'candidate_holdout.log',
                    FINAL_SEEDS, geometry_path, f'final-candidate-{rank}', events)
                if geometry_name not in incumbent_final:
                    incumbent_final[geometry_name] = eval_summary(INCUMBENT,
                        final_dir / 'incumbent_holdout.json', final_dir / 'incumbent_holdout.log',
                        FINAL_SEEDS, geometry_path, f'final-incumbent-{rank}', events)
            except Exception as exc:
                failure = {'trial': item['trial'], 'geometry': geometry_name,
                           'phase': 'final_confirmation', 'error': repr(exc)}
                failures.append(failure)
                emit(events, 'finalist_failed', **failure)
                continue
            candidate_energy = candidate_eval['summary']['policy']['mean_integrated_resonator_energy_mJs']
            incumbent_energy = incumbent_final[geometry_name]['summary']['policy'][
                'mean_integrated_resonator_energy_mJs']
            final_item = {
                'trial': item['trial'], 'geometry': geometry_name, 'recipe': item['recipe'],
                'checkpoint': item['checkpoint'], 'candidate_holdout': candidate_eval,
                'incumbent_holdout': str(final_dir / 'incumbent_holdout.json'),
                'energy_ratio_vs_incumbent_policy': candidate_energy / max(incumbent_energy, 1e-9),
                'accepted_vs_mpc': candidate_eval['promotion'].get('accepted_vs_mpc', False),
            }
            final_results.append(final_item)
            emit(events, 'finalist_confirmed', trial=item['trial'], geometry=geometry_name,
                 recipe=item['recipe'],
                 policy_mpc_ratio=candidate_eval['promotion'].get(
                     'policy_mpc_integrated_resonator_energy_ratio'),
                 energy_ratio_vs_incumbent=final_item['energy_ratio_vs_incumbent_policy'],
                 accepted_vs_mpc=final_item['accepted_vs_mpc'])

        manifest.update({
            'status': 'completed',
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - run_started,
            'completed_candidates': len(results),
            'failed_candidates': len(failures),
            'finalist_trials': [item['trial'] for item in finalists],
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        final_summary = {
            'status': 'completed', 'elapsed_seconds': time.monotonic() - run_started,
            'planned_wall_hours': args.hours, 'completed_candidates': len(results),
            'failed_candidates': len(failures),
            'screening_top_20': sorted(results, key=candidate_rank)[:20],
            'finalists': final_results, 'failures': failures,
            'output_directory': str(out_root),
        }
        (out_root / 'summary.json').write_text(json.dumps(final_summary, indent=2),
                                               encoding='utf-8')
        emit(events, 'sweep_completed', completed_candidates=len(results),
             failed_candidates=len(failures), finalist_trials=final_summary['finalists'],
             elapsed_seconds=final_summary['elapsed_seconds'],
             summary=str(out_root / 'summary.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

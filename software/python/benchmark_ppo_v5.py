"""Reproducible profiling and seeded-parity check for the v4 PPO rollout path."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import subprocess
import time
from pathlib import Path
from types import MethodType

import numpy as np

from rig_rl_env_v4 import RLEnvConfigV4, VectorRigEnvV4
from state_estimator import held_transition, numerical_jacobian


def _legacy_propagate(self, torque, dt):
    """Reference implementation used before batched EKF propagation."""
    old_mean, old_covariance = self.mean.copy(), self.covariance.copy()

    def transition(state):
        return np.r_[held_transition(self.plant, state[:7], torque,
                                    self.physics_dt, dt), state[7]]

    jacobian = numerical_jacobian(transition, old_mean)
    self.mean = transition(old_mean)
    process_noise = np.diag(np.square([2e-5, .06, 3e-5, .025, 2e-4, .025, .008, 1e-4]))
    self.covariance = jacobian @ old_covariance @ jacobian.T
    self.covariance += process_noise * (dt / self.control_dt)
    self.covariance = (self.covariance + self.covariance.T) * .5


def _env(seed, n_envs, scalar_reference=False):
    env = VectorRigEnvV4(n_envs, seed, RLEnvConfigV4(history_length=8))
    env.reset()
    if scalar_reference:
        for item in env.envs:
            estimator = item.loop.estimator
            estimator._propagate = MethodType(_legacy_propagate, estimator)
    return env


def _benchmark_once(mode, n_envs, steps, repeat):
    actions = np.zeros((n_envs, 1), dtype=np.float32)
    env = _env(9000 + repeat, n_envs, scalar_reference=(mode == 'scalar_reference'))
    for _ in range(10):
        env.step(actions)
    started = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - started
    return n_envs * steps / elapsed


def _seeded_parity(steps):
    scalar = _env(7229, 1, scalar_reference=True)
    vectorized = _env(7229, 1)
    old_obs = scalar.envs[0]._current_obs.copy()
    new_obs = vectorized.envs[0]._current_obs.copy()
    errors = {'plant_state': 0., 'estimator_mean': 0., 'covariance': 0.,
              'observation': 0., 'reward': 0., 'applied_torque': 0.}
    for index in range(steps):
        action = np.array([[.12 * np.sin(index * .19)]], dtype=np.float32)
        old_obs, old_reward, _, old_infos = scalar.step(action)
        new_obs, new_reward, _, new_infos = vectorized.step(action)
        old_env, new_env = scalar.envs[0], vectorized.envs[0]
        errors['plant_state'] = max(errors['plant_state'],
                                    float(np.max(np.abs(old_env.y - new_env.y))))
        errors['estimator_mean'] = max(
            errors['estimator_mean'],
            float(np.max(np.abs(old_env.loop.estimator.mean - new_env.loop.estimator.mean))),
        )
        errors['covariance'] = max(
            errors['covariance'],
            float(np.max(np.abs(old_env.loop.estimator.covariance
                                - new_env.loop.estimator.covariance))),
        )
        errors['observation'] = max(errors['observation'],
                                    float(np.max(np.abs(old_obs - new_obs))))
        errors['reward'] = max(errors['reward'],
                               float(np.max(np.abs(old_reward - new_reward))))
        errors['applied_torque'] = max(
            errors['applied_torque'],
            abs(float(old_infos[0]['tau_cmd']) - float(new_infos[0]['tau_cmd'])),
        )
    return {'steps': steps, 'max_absolute_errors': errors}


def _profile_scalar(steps):
    env = _env(8127, 4, scalar_reference=True)
    actions = np.zeros((4, 1), dtype=np.float32)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(steps):
        env.step(actions)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(16)
    return stream.getvalue()


def _source_revision():
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100,
                        help='Vector steps per repeat, per environment count')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--parity-steps', type=int, default=100)
    parser.add_argument('--profile', action='store_true',
                        help='Include a cProfile sample of the scalar reference path')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if min(args.n_envs, args.steps, args.repeats, args.parity_steps) < 1:
        parser.error('counts must be positive')

    started = time.perf_counter()
    rates = {'scalar_reference': [], 'vectorized': []}
    paired_speedups = []
    pair_orders = []
    for repeat in range(args.repeats):
        modes = (('scalar_reference', 'vectorized') if repeat % 2 == 0
                 else ('vectorized', 'scalar_reference'))
        pair = {}
        for mode in modes:
            pair[mode] = _benchmark_once(mode, args.n_envs, args.steps, repeat)
            rates[mode].append(pair[mode])
        pair_orders.append(list(modes))
        paired_speedups.append(pair['vectorized'] / pair['scalar_reference'])
    result = {
        'benchmark': 'sedp-v4-ppo-rollout-throughput',
        'source_revision': _source_revision(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'config': {'n_envs': args.n_envs, 'vector_steps_per_repeat': args.steps,
                   'repeats': args.repeats, 'history_length': 8,
                   'sensor_noise': True, 'actuator_mode': 'torque'},
        'pair_order': pair_orders,
        'scalar_reference': {'transitions_per_second': rates['scalar_reference'],
                             'median_transitions_per_second': float(np.median(rates['scalar_reference']))},
        'vectorized': {'transitions_per_second': rates['vectorized'],
                       'median_transitions_per_second': float(np.median(rates['vectorized']))},
        'paired_speedups': paired_speedups,
        'median_paired_speedup': float(np.median(paired_speedups)),
    }
    result['seeded_parity'] = _seeded_parity(args.parity_steps)
    if args.profile:
        result['scalar_profile'] = _profile_scalar(60)
    result['elapsed_seconds'] = time.perf_counter() - started
    encoded = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()

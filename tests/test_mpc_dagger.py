import sys
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class ConstantPolicy:
    def eval(self):
        return self

    def deterministic(self, observations):
        return torch.full((len(observations), 1), .4, dtype=torch.float32)


class MpcDaggerTests(unittest.TestCase):
    def test_collection_labels_student_visited_states_but_applies_student_actions(self):
        from constrained_mpc import ConstrainedMPC
        from evaluate_goal_hold_v4 import goal_hold_mpc_config
        from rig_rl_env_v4 import RLEnvConfigV4, RigRLEnvV4
        from train_residual_v4 import collect_dagger

        cfg = RLEnvConfigV4(reference_mode='random_goal_hold', episode_seconds=2.,
                            goal_move_seconds=.01, goal_hold_min_seconds=2.,
                            goal_hold_extra_seconds=0.)
        original_step = RigRLEnvV4.step
        applied = []

        def record_student_action(env, action):
            applied.append(float(np.asarray(action).reshape(-1)[0]))
            return original_step(env, action)

        def teacher_label(_teacher, *_args, **_kwargs):
            _teacher.diagnostics = {'success': True, 'solve_seconds': .001}
            return -.2

        with patch.object(RigRLEnvV4, 'step', record_student_action), \
                patch.object(ConstrainedMPC, 'action', teacher_label):
            observations, labels, groups, valid, student_actions, diagnostics = collect_dagger(
                cfg, ConstantPolicy(), samples=2, episodes=1, seed=314,
                horizon=2, mpc_config=goal_hold_mpc_config(cfg, 2))

        self.assertEqual(observations.shape[0], 2)
        np.testing.assert_allclose(labels[:, 0], [-.2, -.2])
        np.testing.assert_allclose(student_actions[:, 0], [.4, .4])
        np.testing.assert_allclose(applied, [.4, .4])
        np.testing.assert_array_equal(groups, [0, 0])
        np.testing.assert_array_equal(valid, [True, True])
        self.assertEqual(len(diagnostics), 2)

    def test_distillation_split_keeps_whole_episodes_and_warm_starts(self):
        from ppo_agent_v2 import ActorCriticV4
        from train_residual_v4 import _initialize_distillation_model, _split_episode_groups

        groups = np.repeat(np.arange(8), 4)
        valid = np.ones(len(groups), dtype=bool)
        train, validation, heldout = _split_episode_groups(groups, valid)
        self.assertEqual(heldout.tolist(), [6, 7])
        self.assertFalse(set(groups[train]) & set(groups[validation]))
        self.assertEqual(set(groups[validation]), {6, 7})

        source = ActorCriticV4(26 * 8, 1, 128)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(.125)
        initialized = _initialize_distillation_model(26 * 8, source)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(initialized.state_dict()[key], value)

    def test_dagger_and_fresh_evaluation_seed_sets_are_disjoint(self):
        from benchmark_mpc_dagger import validate_dagger_seed_separation

        prior = {
            'optimization': [7001, 7002, 7003, 7004, 7005],
            'confirmation': [9001, 9002, 9003],
            'teacher': [4242 + 1009 * i for i in range(16)],
            'evaluation': list(range(8011, 8021)),
        }
        dagger, evaluation = validate_dagger_seed_separation(
            prior, dagger_seed=42420, dagger_episodes=16,
            evaluation_seeds=list(range(12011, 12021)))
        self.assertEqual(dagger, [42420 + 1009 * i for i in range(16)])
        self.assertEqual(evaluation, list(range(12011, 12021)))
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_dagger_seed_separation(
                prior, dagger_seed=42420, dagger_episodes=16,
                evaluation_seeds=[42420])

    def test_within_two_percent_energy_count_includes_both_directions(self):
        from benchmark_mpc_dagger import compare_student_versions

        ratios = [1.0, .985, .97, 1.015, 1.03]
        original = []
        dagger = []
        for seed, ratio in enumerate(ratios):
            base = {
                'seed': seed, 'integrated_resonator_energy_mJs': 1.,
                'position_rmse_mm': .2, 'angle_rms_deg': .1,
                'peak_angle_deg': .3, 'peak_rail_fraction': .7,
                'policy_action_p95_ms': .2, 'completed_two_second_holds': 3,
            }
            original.append(dict(base))
            dagger.append({**base, 'integrated_resonator_energy_mJs': ratio})
        result = compare_student_versions(original, dagger)
        self.assertEqual(result['seeds_lower_energy'], 2)
        self.assertEqual(result['seeds_energy_within_2_percent'], 3)


if __name__ == '__main__':
    unittest.main()

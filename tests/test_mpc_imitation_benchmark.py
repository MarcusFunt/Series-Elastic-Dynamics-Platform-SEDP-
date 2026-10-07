import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class MpcImitationBenchmarkTests(unittest.TestCase):
    def test_seed_protocol_disjoins_training_from_optimization_and_evaluation(self):
        from benchmark_mpc_imitation import validate_seed_separation

        teacher_seeds = validate_seed_separation([7001, 7002], [8011], 4242, 3,
                                                 [9001, 9002])
        self.assertEqual(teacher_seeds, [4242, 5251, 6260])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_seed_separation([7001, 7002], [8011], 7002, 3)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            validate_seed_separation([], [8011], 4242, 3)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_seed_separation([7001], [8011], 4242, 3, [7001])

    def test_stride_two_requires_both_initial_and_fresh_confirmation_gates(self):
        from benchmark_mpc_imitation import select_confirmed_stride

        self.assertEqual(select_confirmed_stride({'accepted': True}, {'accepted': True}), 2)
        self.assertEqual(select_confirmed_stride({'accepted': True}, {'accepted': False}), 1)
        self.assertEqual(select_confirmed_stride({'accepted': False}, {'accepted': True}), 1)

    def test_resolved_teacher_config_preserves_task_mpc_and_plant(self):
        from active_vibration_rig_2d import PlantParams
        from benchmark_mpc_imitation import _write_teacher_config
        from evaluate_goal_hold_v4 import goal_hold_mpc_config
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg = RLEnvConfigV4(reference_mode='random_goal_hold', episode_seconds=13.,
                            goal_move_seconds=.8, goal_hold_min_seconds=3.,
                            goal_hold_extra_seconds=.7, goal_edge_margin_m=.017)
        plant = PlantParams()
        mpc = goal_hold_mpc_config(cfg, horizon=10, linearization_stride=2)
        with TemporaryDirectory() as tmp:
            path = _write_teacher_config(Path(tmp), cfg, mpc, plant)
            resolved = __import__('json').loads(path.read_text(encoding='utf-8'))
        self.assertEqual(resolved['env_config']['episode_seconds'], 13.)
        self.assertEqual(resolved['env_config']['goal_edge_margin_m'], .017)
        self.assertEqual(resolved['env_config']['goal_hold_min_seconds'], 3.)
        self.assertEqual(resolved['mpc_config']['horizon'], 10)
        self.assertEqual(resolved['mpc_config']['linearization_stride'], 2)
        self.assertEqual(resolved['plant_parameters'], __import__('dataclasses').asdict(plant))
        from benchmark_mpc_imitation import _assert_teacher_config_matches
        _assert_teacher_config_matches(resolved, cfg, mpc, plant)
        resolved['env_config']['episode_seconds'] = 12.
        with self.assertRaisesRegex(RuntimeError, 'env_config'):
            _assert_teacher_config_matches(resolved, cfg, mpc, plant)

    def test_saved_teacher_stride_cannot_be_changed_when_refitting_labels(self):
        from constrained_mpc import MPCConfig
        from evaluate_goal_hold_v4 import goal_hold_mpc_config
        from rig_rl_env_v4 import RLEnvConfigV4
        from train_residual_v4 import _dataset_mpc_config

        cfg = RLEnvConfigV4(reference_mode='random_goal_hold')
        saved = {'mpc_config': __import__('dataclasses').asdict(
            goal_hold_mpc_config(cfg, horizon=8, linearization_stride=1))}
        fitted = _dataset_mpc_config(saved, cfg, requested_stride=1)
        self.assertIsInstance(fitted, MPCConfig)
        with self.assertRaisesRegex(ValueError, 'recollect teacher data'):
            _dataset_mpc_config(saved, cfg, requested_stride=2)

    def test_saved_teacher_horizon_is_the_default_for_standalone_evaluation(self):
        from dataclasses import asdict
        from evaluate_goal_hold_v4 import goal_hold_mpc_config, resolve_mpc_config
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg = RLEnvConfigV4(reference_mode='random_goal_hold')
        saved = asdict(goal_hold_mpc_config(cfg, horizon=4, linearization_stride=2))
        checkpoint = {'extra': {'mpc_config': saved}}
        defaulted = resolve_mpc_config(cfg, checkpoint)
        self.assertEqual(defaulted.horizon, 4)
        self.assertEqual(defaulted.linearization_stride, 2)
        explicit = resolve_mpc_config(cfg, checkpoint, horizon_override=6)
        self.assertEqual(explicit.horizon, 6)
        self.assertEqual(explicit.linearization_stride, 2)

    def test_stride_candidate_requires_behavior_gate_and_material_speedup(self):
        from benchmark_mpc_imitation import judge_mpc_optimization

        def row(seed, **changes):
            values = {
                'seed': seed, 'terminated': False, 'exclusion_zone_violations': 0,
                'goal_targets_respect_margin': True, 'goal_count': 3,
                'completed_two_second_holds': 3, 'position_rmse_mm': .2,
                'angle_rms_deg': .1, 'peak_angle_deg': .3,
                'integrated_resonator_energy_mJs': .01,
                'saturation_fraction': 0., 'mpc_fallback_fraction': 0.,
                'peak_rail_fraction': .70,
                'mpc_solve_p95_ms': 8., 'mpc_deadline_miss_fraction': .01,
            }
            values.update(changes)
            return values

        baseline = [row(7001), row(7002)]
        optimized = [row(7001, mpc_solve_p95_ms=6.8),
                     row(7002, mpc_solve_p95_ms=6.9)]
        decision = judge_mpc_optimization(baseline, optimized)
        self.assertTrue(decision['accepted'])
        self.assertEqual(decision['selected_stride'], 2)
        self.assertGreaterEqual(decision['p95_speedup_fraction'], .10)

        regressed = [row(7001, mpc_solve_p95_ms=4., position_rmse_mm=.5),
                     row(7002, mpc_solve_p95_ms=4.)]
        rejected = judge_mpc_optimization(baseline, regressed)
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['selected_stride'], 1)
        self.assertTrue(any('position_rmse_mm' in reason for reason in rejected['reasons']))

        rail_regression = [row(7001, mpc_solve_p95_ms=6., peak_rail_fraction=.72),
                           row(7002, mpc_solve_p95_ms=6.)]
        rail_rejected = judge_mpc_optimization(baseline, rail_regression)
        self.assertFalse(rail_rejected['accepted'])
        self.assertTrue(any('peak rail' in reason for reason in rail_rejected['reasons']))

    def test_imitation_gate_separates_control_quality_from_speed(self):
        from benchmark_mpc_imitation import compare_imitation_to_mpc

        mpc = [{
            'seed': 8011, 'position_rmse_mm': .2, 'angle_rms_deg': .1,
            'peak_angle_deg': .3, 'integrated_resonator_energy_mJs': .01,
            'terminated': False, 'exclusion_zone_violations': 0,
            'goal_targets_respect_margin': True, 'goal_count': 3,
            'completed_two_second_holds': 3, 'saturation_fraction': 0.,
            'peak_rail_fraction': .70,
            'mpc_solve_p95_ms': 8., 'mpc_solve_p99_ms': 9.,
        }]
        imitation = [{
            **mpc[0], 'policy_action_p95_ms': .1, 'policy_action_p99_ms': .2,
            'position_rmse_mm': .21,
            'policy_action_p50_ms': .05, 'policy_deadline_miss_fraction': 0.,
        }]
        mpc[0].update(mpc_solve_p50_ms=4., mpc_deadline_miss_fraction=0.)
        decision = compare_imitation_to_mpc(mpc, imitation)
        self.assertTrue(decision['performance_preserved'])
        self.assertTrue(decision['useful'])
        self.assertAlmostEqual(decision['p95_latency_speedup'], 80.)

        imitation[0]['policy_action_p95_ms'] = 5.
        imitation[0]['policy_action_p99_ms'] = 5.1
        modest_speedup = compare_imitation_to_mpc(mpc, imitation)
        self.assertTrue(modest_speedup['performance_preserved'])
        self.assertFalse(modest_speedup['useful'])
        self.assertTrue(modest_speedup['speed_reasons'])

        imitation[0]['position_rmse_mm'] = 1.
        rejected = compare_imitation_to_mpc(mpc, imitation)
        self.assertFalse(rejected['performance_preserved'])
        self.assertFalse(rejected['useful'])

        imitation[0].update(position_rmse_mm=.21, peak_rail_fraction=.72)
        rail_rejected = compare_imitation_to_mpc(mpc, imitation)
        self.assertFalse(rail_rejected['performance_preserved'])
        self.assertTrue(any('peak rail' in reason
                            for reason in rail_rejected['performance_reasons']))

    def test_report_reads_teacher_fit_from_training_artifact(self):
        from benchmark_mpc_imitation import _write_markdown

        data = {
            'source': {'git_revision': 'abc123', 'source_sha256': 'deadbeef'},
            'config': {'control_dt': .01, 'actuator_mode': 'torque'},
            'optimization': {
                'accepted': True, 'baseline_mean_seed_p95_ms': 8.,
                'candidate_mean_seed_p95_ms': 6., 'p95_speedup_fraction': .25,
                'pairs': [], 'reasons': [],
            },
            'optimization_baseline_rows': [], 'optimization_candidate_rows': [],
            'training': {'teacher_fit': {
                'best_heldout_action_mae': .01, 'train_samples': 10,
                'validation_samples': 3, 'excluded_fallback_labels': 1,
            }},
            'imitation': {
                'performance_preserved': True, 'useful': True,
                'mean_mpc_solve_p95_ms': 8., 'mean_imitation_action_p95_ms': .1,
                'mean_mpc_solve_p50_ms': 4., 'mean_mpc_solve_p99_ms': 9.,
                'mean_imitation_action_p50_ms': .05, 'mean_imitation_action_p99_ms': .2,
                'p99_latency_speedup': 45.,
                'mean_mpc_deadline_miss_fraction': 0.,
                'mean_imitation_deadline_miss_fraction': 0.,
                'p95_latency_speedup': 80., 'mean_energy_ratio': 1.,
                'pairs': [], 'reasons': [],
            },
            'evidence_json': 'evidence.json', 'run_directory': 'runs/test',
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.md'
            _write_markdown(path, data)
            self.assertIn('Best held-out action MAE: 0.010000',
                          path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()

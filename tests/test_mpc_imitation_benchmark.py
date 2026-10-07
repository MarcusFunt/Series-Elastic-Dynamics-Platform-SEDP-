import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class MpcImitationBenchmarkTests(unittest.TestCase):
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

    def test_imitation_gate_separates_control_quality_from_speed(self):
        from benchmark_mpc_imitation import compare_imitation_to_mpc

        mpc = [{
            'seed': 8011, 'position_rmse_mm': .2, 'angle_rms_deg': .1,
            'peak_angle_deg': .3, 'integrated_resonator_energy_mJs': .01,
            'terminated': False, 'exclusion_zone_violations': 0,
            'goal_targets_respect_margin': True, 'goal_count': 3,
            'completed_two_second_holds': 3, 'saturation_fraction': 0.,
            'mpc_solve_p95_ms': 8., 'mpc_solve_p99_ms': 9.,
        }]
        imitation = [{
            **mpc[0], 'policy_action_p95_ms': .1, 'policy_action_p99_ms': .2,
            'position_rmse_mm': .21,
        }]
        decision = compare_imitation_to_mpc(mpc, imitation)
        self.assertTrue(decision['performance_preserved'])
        self.assertTrue(decision['useful'])
        self.assertAlmostEqual(decision['p95_latency_speedup'], 80.)

        imitation[0]['policy_action_p95_ms'] = 5.
        modest_speedup = compare_imitation_to_mpc(mpc, imitation)
        self.assertTrue(modest_speedup['performance_preserved'])
        self.assertFalse(modest_speedup['useful'])
        self.assertTrue(modest_speedup['speed_reasons'])

        imitation[0]['position_rmse_mm'] = 1.
        rejected = compare_imitation_to_mpc(mpc, imitation)
        self.assertFalse(rejected['performance_preserved'])
        self.assertFalse(rejected['useful'])

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

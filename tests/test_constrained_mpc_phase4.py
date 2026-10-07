import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class ConstrainedMPCPhase4Tests(unittest.TestCase):
    def setUp(self):
        from active_vibration_rig_2d import ControllerParams, PlantParams
        from constrained_mpc import ConstrainedMPC, MPCConfig
        self.params = PlantParams()
        self.cp = ControllerParams()
        self.mpc = ConstrainedMPC(self.params, self.cp, MPCConfig(horizon=4))
        self.state = np.zeros(7)
        self.references = [(0., 0., 0.)] * 5

    def test_solver_statuses_have_machine_readable_categories(self):
        cases = [
            (9, 'iteration_limit'),
            (4, 'infeasible_constraints'),
            (6, 'numerical_solver_error'),
            (8, 'solver_failure'),
        ]
        for status, expected in cases:
            with self.subTest(status=status):
                result = SimpleNamespace(success=False, status=status,
                                         message='controlled fake result', x=np.zeros(4), nit=3)
                self.assertEqual(self.mpc._solver_outcome(result), expected)

    def test_solver_exception_is_reported_as_numerical_error_and_falls_back_bounded(self):
        with patch('constrained_mpc.minimize', side_effect=FloatingPointError('bad Hessian')):
            action = self.mpc.action(self.state, self.references)
        self.assertEqual(action, 0.)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'numerical_solver_error')
        self.assertEqual(self.mpc.diagnostics['reason'], 'bad Hessian')
        self.assertEqual(self.mpc.diagnostics['fallback_action'], 0.)
        self.assertLessEqual(abs(self.mpc.diagnostics['fallback_action']), 1.)

    def test_model_linearization_exception_is_also_categorized(self):
        with patch.object(self.mpc, '_transition_augmented',
                          side_effect=FloatingPointError('bad transition linearization')):
            action = self.mpc.action(self.state, self.references)
        self.assertEqual(action, 0.)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'numerical_solver_error')
        self.assertIn('bad transition linearization', self.mpc.diagnostics['reason'])

    def test_iteration_limit_and_infeasible_initial_bounds_are_distinct(self):
        failed = SimpleNamespace(success=False, status=9, message='Iteration limit reached',
                                 x=np.zeros(4), nit=60)
        with patch('constrained_mpc.minimize', return_value=failed):
            self.assertEqual(self.mpc.action(self.state, self.references), 0.)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'iteration_limit')

        state = self.state.copy()
        state[2] = .09
        self.assertEqual(self.mpc.action(state, self.references), 0.)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'infeasible_initial_bounds')

    def test_deadline_expires_as_a_time_limit_outcome(self):
        from constrained_mpc import MPCConfig, ConstrainedMPC
        timed = ConstrainedMPC(self.params, self.cp, MPCConfig(horizon=4, time_limit_seconds=0.0))
        action = timed.action(self.state, self.references)
        self.assertEqual(action, 0.)
        self.assertEqual(timed.diagnostics['outcome'], 'time_limit')

    def test_step_dir_rollout_matches_environment_interval_and_does_not_mutate_actuator(self):
        from constrained_mpc import ConstrainedMPC, MPCConfig
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        cfg = RLEnvConfigV4(actuator_mode='step_dir', domain_randomization=False,
                            kick_probability=0., initial_theta_std=0.,
                            initial_theta_dot_std=0., sensor_noise=False, oracle_state=True)
        env = RigRLEnvV4(cfg=cfg, seed=12)
        env.reset()
        env.y[3] = .04
        env.step_dir_actuator.command(3.0)
        actuator = copy.deepcopy(env.step_dir_actuator)
        initial_actuator = copy.deepcopy(actuator)
        teacher = ConstrainedMPC(env.base_params, env.cp, MPCConfig(
            horizon=4, physics_dt=cfg.physics_dt, control_dt=cfg.control_dt,
            residual_accel_limit=cfg.residual_accel_limit, actuator_mode='step_dir',
            step_dir=cfg.step_dir))

        predicted, predicted_actuator = teacher._transition_active(
            env.y.copy(), 1.0, actuator)
        env._step_dir_interval(env.controller.project_accel(env.y, 1.0))
        np.testing.assert_allclose(predicted, env.y, rtol=0., atol=1e-12)
        self.assertAlmostEqual(predicted_actuator.velocity_rad_s,
                               env.step_dir_actuator.velocity_rad_s, places=12)
        self.assertAlmostEqual(predicted_actuator.position_rad,
                               env.step_dir_actuator.position_rad, places=12)
        self.assertAlmostEqual(actuator.velocity_rad_s, initial_actuator.velocity_rad_s)
        self.assertAlmostEqual(actuator.position_rad, initial_actuator.position_rad)

        augmented = teacher._augmented_state(env.y, env.step_dir_actuator)
        template = copy.deepcopy(env.step_dir_actuator)
        for requested in (-.4, .25):
            augmented = teacher._transition_augmented(augmented, requested, template)
            env._step_dir_interval(env.controller.project_accel(env.y, requested))
            np.testing.assert_allclose(augmented[:7], env.y, rtol=0., atol=1e-12)
            self.assertAlmostEqual(augmented[7], env.step_dir_actuator.velocity_rad_s, places=12)
            self.assertAlmostEqual(augmented[8], env.step_dir_actuator.position_rad, places=12)

    def test_torque_rollout_matches_environment_transition(self):
        from constrained_mpc import ConstrainedMPC, MPCConfig
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        cfg = RLEnvConfigV4(domain_randomization=False, kick_probability=0.,
                            initial_theta_std=0., initial_theta_dot_std=0.,
                            sensor_noise=False, oracle_state=True)
        env = RigRLEnvV4(cfg=cfg, seed=5)
        env.reset()
        env.y[4] = .025
        env.loop.state = env.y.copy()
        teacher = ConstrainedMPC(env.base_params, env.cp, MPCConfig(
            horizon=4, physics_dt=cfg.physics_dt, control_dt=cfg.control_dt,
            residual_accel_limit=cfg.residual_accel_limit))
        total_accel = .7
        predicted, _ = teacher._transition_active(env.y, total_accel)
        base = env.loop.base_accel(env.reference.sample(env.t))
        action = (total_accel-base)/cfg.residual_accel_limit
        env.step([action])
        np.testing.assert_allclose(predicted, env.y, rtol=0., atol=1e-12)

    def test_prediction_uses_queued_commands_without_advancing_the_live_queue(self):
        from timing_models import CommandDelayQueue
        queue = CommandDelayQueue(delay=.015, jitter=.003, seed=23)
        queue.issue(.5, 0.)
        before_pending = copy.deepcopy(queue.pending)
        before_active = (queue.active_value, queue.active_issue_time,
                         queue.active_arrival_time, queue.active_apply_time,
                         queue._sequence)
        before_rng = copy.deepcopy(queue.rng.bit_generator.state)
        sources, fixed = self.mpc._command_source_plan(queue, 0.)
        self.assertEqual(len(sources), 4)
        self.assertEqual(queue.pending, before_pending)
        self.assertEqual((queue.active_value, queue.active_issue_time,
                          queue.active_arrival_time, queue.active_apply_time,
                          queue._sequence), before_active)
        self.assertEqual(queue.rng.bit_generator.state, before_rng)
        self.assertTrue(any(source is not None for source in sources))
        self.assertEqual(len(fixed), 4)

    def test_step_dir_evaluation_runs_and_records_outcome_rates(self):
        from evaluate_v4 import run_case
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg = RLEnvConfigV4(
            episode_seconds=.02, actuator_mode='step_dir', domain_randomization=False,
            kick_probability=0., initial_theta_std=0., initial_theta_dot_std=0.,
            sensor_noise=False)
        row = run_case('mpc', seed=7, cfg=cfg, horizon=4)
        self.assertEqual(row['actuator_mode'], 'step_dir')
        self.assertEqual(sum(row['mpc_outcome_counts'].values()), row['samples'])
        self.assertAlmostEqual(sum(row['mpc_outcome_rates'].values()), 1.)

    def test_mpc_uses_live_delayed_queue_state(self):
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4
        from evaluate_v4 import reference_preview

        cfg = RLEnvConfigV4(
            episode_seconds=.02, command_delay=.005, command_jitter=.001,
            domain_randomization=False, kick_probability=0., initial_theta_std=0.,
            initial_theta_dot_std=0., sensor_noise=False)
        env = RigRLEnvV4(cfg=cfg, seed=11)
        env.reset()
        action = self.mpc.action(
            env.loop.state, reference_preview(env, self.mpc.cfg.horizon),
            command_queue=env.command_queue, step_dir_actuator=env.step_dir_actuator,
            time_s=env.t)
        self.assertLessEqual(abs(action), 1.)
        self.assertIn(self.mpc.diagnostics['outcome'],
                      {'success', 'infeasible_constraints', 'nonlinear_actuator_rejection',
                       'nonlinear_state_rejection', 'iteration_limit', 'solver_failure',
                       'numerical_solver_error'})
        env.step([action])

    def test_success_and_nonlinear_rejection_are_distinct(self):
        success = SimpleNamespace(success=True, status=0, message='Optimization terminated successfully',
                                  x=np.zeros(4), nit=2)
        with patch('constrained_mpc.minimize', return_value=success):
            self.mpc.action(self.state, self.references)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'success')
        self.assertTrue(self.mpc.diagnostics['success'])

        nonlinear = {'nonlinear_peak_rail_fraction': .9,
                     'nonlinear_max_constraint_violation': .01}
        with patch('constrained_mpc.minimize', return_value=success), patch.object(
                self.mpc, '_nonlinear_accept',
                return_value=(False, 'nonlinear_state_rejection', 'nonlinear rail gate', nonlinear)):
            self.mpc.action(self.state, self.references)
        self.assertEqual(self.mpc.diagnostics['outcome'], 'nonlinear_state_rejection')

    def test_paired_objective_report_fails_per_case_regressions(self):
        from evaluate_v4 import mpc_comparison
        shared = {'scenario':'step','seed':0,'randomized':False,'actuator_mode':'torque',
                  'command_delay_seconds':0.,'command_jitter_seconds':0.,
                  'angle_rms_deg':2.,'peak_angle_deg':4.,'position_rmse_mm':5.,
                  'peak_rail_fraction':.5,'saturation_fraction':0.,
                  'stop_contact_fraction':0.,'max_soft_rail_violation_fraction':0.,
                  'max_carriage_speed_violation_m_s':0.,
                  'max_motor_speed_violation_rad_s':0.,'terminated':False,
                  'energy_integral_mJs':1.,'mpc_fallback_fraction':.02,
                  'mpc_outcome_counts':{'success':98,'nonlinear_state_rejection':2}}
        baseline = dict(shared, controller='mpc_baseline')
        candidate = dict(shared, controller='mpc', angle_rms_deg=1.8,
                         energy_integral_mJs=.9, position_rmse_mm=6.)
        result = mpc_comparison([baseline,candidate])
        self.assertFalse(result['accepted'])
        self.assertTrue(result['per_case_regressions_checked'])
        self.assertTrue(any('position_rmse_mm' in reason for reason in result['reasons']))

    def test_goal_hold_config_is_shared_and_evaluation_records_latency_and_stride(self):
        from evaluate_goal_hold_v4 import goal_hold_mpc_config, run_case
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg = RLEnvConfigV4(
            episode_seconds=.02, domain_randomization=False, kick_probability=0.,
            initial_theta_std=0., initial_theta_dot_std=0., sensor_noise=False,
            reference_mode='random_goal_hold')
        mpc_cfg = goal_hold_mpc_config(cfg, horizon=4, linearization_stride=2)
        self.assertEqual(mpc_cfg.linearization_stride, 2)
        self.assertEqual(mpc_cfg.angle_weight, 3.2)
        self.assertEqual(mpc_cfg.angular_rate_weight, .8)
        self.assertEqual(mpc_cfg.action_weight, .04)

        row = run_case('mpc', seed=7, cfg=cfg, horizon=4, mpc_config=mpc_cfg)
        self.assertEqual(row['mpc_linearization_stride'], 2)
        self.assertIn('mpc_solve_p50_ms', row)
        self.assertIn('mpc_solve_p99_ms', row)
        self.assertIn('mpc_deadline_miss_fraction', row)


if __name__ == '__main__':
    unittest.main()

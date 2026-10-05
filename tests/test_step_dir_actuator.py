import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class StepDirActuatorTests(unittest.TestCase):
    def test_speed_and_acceleration_limits_and_integer_step_pulses(self):
        from active_vibration_rig_2d import StepDirActuator, StepDirParams

        actuator = StepDirActuator(StepDirParams(
            step_angle_rad=0.1, max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0, tracking_time_constant_s=0.01,
        ))
        actuator.command(5.0, acceleration_limit_rad_s2=4.0)
        diagnostics = []
        for _ in range(30):
            diagnostics.append(actuator.advance(0.01, 0.0, 0.0, torque_limit_nm=0.48))

        self.assertTrue(all(isinstance(d['step_pulses'], int) for d in diagnostics))
        self.assertLessEqual(max(abs(d['step_velocity_rad_s']) for d in diagnostics), 1.0)
        self.assertLessEqual(max(abs(d['step_acceleration_rad_s2']) for d in diagnostics), 2.0 + 1e-12)
        self.assertTrue(diagnostics[-1]['velocity_saturated'])
        self.assertTrue(diagnostics[0]['acceleration_limit_saturated'])
        self.assertGreater(sum(d['step_pulses'] for d in diagnostics), 0)
        self.assertEqual(diagnostics[-1]['step_count'], actuator.step_count)

    def test_lag_and_motor_tracking_error_are_reported(self):
        from active_vibration_rig_2d import StepDirActuator, StepDirParams

        actuator = StepDirActuator(StepDirParams(
            step_angle_rad=0.01, max_velocity_rad_s=10.0,
            max_acceleration_rad_s2=1000.0, tracking_time_constant_s=0.2,
        ))
        actuator.command(4.0)
        result = actuator.advance(0.01, actual_phi_rad=0.0,
                                  actual_omega_rad_s=0.0, torque_limit_nm=0.48)
        self.assertGreater(result['requested_velocity_rad_s'], result['step_velocity_rad_s'])
        self.assertGreater(result['velocity_tracking_error_rad_s'], 0.0)
        self.assertGreaterEqual(result['motor_tracking_error_rad'], 0.0)
        self.assertTrue(np.isfinite(result['torque_command_nm']))

    def test_reversal_slews_through_zero_and_emits_opposite_direction_pulses(self):
        from active_vibration_rig_2d import StepDirActuator, StepDirParams

        actuator = StepDirActuator(StepDirParams(
            step_angle_rad=0.05, max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0, tracking_time_constant_s=0.01,
        ))
        actuator.command(1.0)
        positive = [actuator.advance(0.01, 0.0, 0.0, 0.48) for _ in range(100)]
        self.assertTrue(any(d['direction'] == 1 for d in positive))
        speed_before_reverse = actuator.velocity_rad_s

        actuator.command(-1.0)
        first_reverse = actuator.advance(0.01, 0.0, 0.0, 0.48)
        self.assertGreaterEqual(first_reverse['step_velocity_rad_s'], 0.0)
        self.assertLess(first_reverse['step_velocity_rad_s'], speed_before_reverse)
        reverse = [actuator.advance(0.01, 0.0, 0.0, 0.48) for _ in range(250)]
        self.assertTrue(any(d['direction'] == -1 for d in reverse))
        self.assertLess(actuator.step_count, positive[-1]['step_count'])

    def test_replay_is_deterministic(self):
        from active_vibration_rig_2d import StepDirActuator, StepDirParams

        params = StepDirParams()
        a = StepDirActuator(params)
        b = StepDirActuator(params)
        outputs_a = []
        outputs_b = []
        commands = [(3.0, 80.0)] * 20 + [(-2.0, 120.0)] * 35
        for velocity, accel in commands:
            a.command(velocity, accel)
            b.command(velocity, accel)
            outputs_a.append(a.advance(0.001, 0.01, 0.2, 0.4))
            outputs_b.append(b.advance(0.001, 0.01, 0.2, 0.4))
        self.assertEqual(outputs_a, outputs_b)
        self.assertEqual(a.step_count, b.step_count)

    def test_torque_environment_remains_default_and_uses_existing_transition(self):
        from rig_rl_env_v3 import RigRLEnvV3, RLEnvConfigV3

        cfg = RLEnvConfigV3(domain_randomization=False, kick_probability=0.0,
                            observation_noise_std=0.0)
        env = RigRLEnvV3(cfg=cfg, seed=13)
        env.reset(seed=13)
        self.assertEqual(cfg.actuator_mode, 'torque')
        action = 0.2
        reference = env.reference.sample(env.t)
        base = env._base_accel(reference)
        residual = action * cfg.residual_accel_limit
        total = env.controller.project_accel(env.y, base + residual)
        expected_torque = env.controller.torque_from_accel(env.y, total)
        expected_y = env.y.copy()
        for _ in range(cfg.substeps):
            expected_y = env.plant.rk4(expected_y, expected_torque, cfg.physics_dt)

        _, _, _, _, info = env.step(action)
        np.testing.assert_allclose(env.y, expected_y, rtol=0.0, atol=1e-13)
        self.assertAlmostEqual(info['tau_cmd'], expected_torque, places=13)

    def test_step_dir_environment_reports_pulse_and_tracking_diagnostics(self):
        from active_vibration_rig_2d import StepDirParams
        from rig_rl_env_v3 import RigRLEnvV3, RLEnvConfigV3

        cfg = RLEnvConfigV3(
            domain_randomization=False, kick_probability=0.0,
            observation_noise_std=0.0, actuator_mode='step_dir',
            step_dir=StepDirParams(max_velocity_rad_s=20.0,
                                   max_acceleration_rad_s2=600.0),
        )
        env = RigRLEnvV3(cfg=cfg, seed=7)
        env.reset(seed=7)
        infos = [env.step(1.0)[-1] for _ in range(8)]
        self.assertTrue(all(info['actuator_mode'] == 'step_dir' for info in infos))
        self.assertTrue(all(isinstance(info['step_count'], int) for info in infos))
        self.assertTrue(any(info['step_pulses_interval'] > 0 for info in infos))
        self.assertTrue(all(np.isfinite(info['motor_tracking_error_rad']) for info in infos))
        self.assertTrue(all(abs(info['step_velocity_rad_s']) <= cfg.step_dir.max_velocity_rad_s
                            for info in infos))

    def test_v4_can_use_step_dir_mode(self):
        from active_vibration_rig_2d import StepDirParams
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        cfg = RLEnvConfigV4(domain_randomization=False, kick_probability=0.0,
                            sensor_noise=False, oracle_state=True,
                            actuator_mode='step_dir',
                            step_dir=StepDirParams(max_velocity_rad_s=20.0))
        env = RigRLEnvV4(cfg=cfg, seed=21)
        env.reset(seed=21)
        _, _, _, _, info = env.step(0.5)
        self.assertEqual(info['actuator_mode'], 'step_dir')
        self.assertIn('step_count', info)
        self.assertIn('torque_saturated', info)

    def test_mpc_evaluation_uses_aligned_step_dir_prediction(self):
        from evaluate_v4 import run_case
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg = RLEnvConfigV4(actuator_mode='step_dir', episode_seconds=.02,
                            domain_randomization=False, kick_probability=0.,
                            initial_theta_std=0., initial_theta_dot_std=0.,
                            sensor_noise=False)
        row = run_case('mpc', cfg=cfg, horizon=2)
        self.assertEqual(row['actuator_mode'], 'step_dir')
        self.assertEqual(sum(row['mpc_outcome_counts'].values()), row['samples'])

    def test_invalid_actuator_mode_is_rejected(self):
        from rig_rl_env_v3 import RLEnvConfigV3

        with self.assertRaises(ValueError):
            RLEnvConfigV3(actuator_mode='stepping')

    def test_step_dir_configuration_survives_checkpoint_style_serialization(self):
        from dataclasses import asdict
        from active_vibration_rig_2d import StepDirParams
        from rig_rl_env_v3 import RLEnvConfigV3

        config = RLEnvConfigV3(actuator_mode='step_dir',
                               step_dir=StepDirParams(max_velocity_rad_s=37.0))
        restored = RLEnvConfigV3(**asdict(config))
        self.assertEqual(restored.actuator_mode, 'step_dir')
        self.assertEqual(restored.step_dir.max_velocity_rad_s, 37.0)


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class SensorTimingTests(unittest.TestCase):
    def test_sensor_and_command_timing_configuration_round_trips(self):
        from dataclasses import asdict
        from rig_rl_env_v4 import RLEnvConfigV4

        config = RLEnvConfigV4(
            command_delay=.012, command_jitter=.002,
            sensor_timing={'gyro': {'sample_period': .02, 'delay': .01,
                                    'jitter': .003, 'dropout_probability': .1}})
        restored = RLEnvConfigV4(**asdict(config))
        self.assertEqual(restored.command_delay, config.command_delay)
        self.assertEqual(restored.command_jitter, config.command_jitter)
        self.assertEqual(restored.sensor_timing, config.sensor_timing)

    def test_sensor_channels_have_independent_rates_and_dropout_streams(self):
        from active_vibration_rig_2d import PlantParams
        from timing_models import AsynchronousSensorSuite, normalize_sensor_timing

        timing = normalize_sensor_timing({
            'motor_angle': {'sample_period': .005},
            'lever_angle': {'sample_period': .01, 'dropout_probability': 1.0},
            'gyro': {'sample_period': .02, 'dropout_probability': 1.0},
        }, .01, physics_dt=.001)
        sensors = AsynchronousSensorSuite(PlantParams(), timing, noise=False, seed=12)
        state = np.zeros(7)
        received = []
        for tick in range(31):
            received.extend(sensors.tick(state, tick * .001))

        self.assertEqual(sensors.sample_count['motor_angle'], 7)
        self.assertEqual(sensors.sample_count['gyro'], 2)
        self.assertEqual(sensors.dropout_count['gyro'], 2)
        self.assertEqual(sensors.dropout_count['lever_angle'], 4)
        self.assertEqual([packet['channel'] for packet in received],
                         ['motor_angle'] * 7)
        self.assertEqual(received[1]['acquisition_time'], .005)

    def test_sensor_jitter_is_seeded_and_never_arrives_before_acquisition(self):
        from active_vibration_rig_2d import PlantParams
        from timing_models import AsynchronousSensorSuite, normalize_sensor_timing

        timing = normalize_sensor_timing({
            'motor_angle': {'sample_period': .01, 'delay': .02, 'jitter': .005},
        }, .01)
        state = np.zeros(7)
        suites = [AsynchronousSensorSuite(PlantParams(), timing, noise=False, seed=44)
                  for _ in range(2)]
        outputs = [[], []]
        for tick in range(61):
            time = tick * .001
            for i, suite in enumerate(suites):
                outputs[i].extend(packet for packet in suite.tick(state, time)
                                  if packet['channel'] == 'motor_angle')

        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(all(packet['arrival_time'] >= packet['acquisition_time']
                            for packet in outputs[0]))
        self.assertGreater(len({packet['arrival_time'] for packet in outputs[0]}), 1)

    def test_delayed_measurement_updates_acquisition_state_then_replays(self):
        from active_vibration_rig_2d import PlantParams
        from state_estimator import StateEstimator

        delayed = StateEstimator(PlantParams(), .001, .01, history_horizon=.05)
        current = StateEstimator(PlantParams(), .001, .01, history_horizon=.05)
        initial = {'motor_angle': 0., 'lever_angle': 0., 'gyro': 0., 'time': 0.}
        delayed.update(initial)
        current.update(initial)
        delayed.predict(0.); delayed.predict(0.)
        current.predict(0.); current.predict(0.)

        delayed.update({'lever_angle': .2, 'time': .005, 'arrival_time': .02})
        current.update({'lever_angle': .2, 'time': .02, 'arrival_time': .02})

        self.assertEqual(delayed.time, .02)
        self.assertTrue(delayed.last_update_was_delayed)
        self.assertGreater(abs(delayed.state[5] - current.state[5]), 1e-4)

    def test_explicit_zero_impairment_preserves_synchronous_environment(self):
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        common = dict(domain_randomization=False, kick_probability=0,
                      initial_theta_std=0, initial_theta_dot_std=0,
                      sensor_noise=False)
        baseline = RigRLEnvV4(cfg=RLEnvConfigV4(**common), seed=27)
        configured = RigRLEnvV4(cfg=RLEnvConfigV4(
            **common, sensor_timing={
                'motor_angle': {'sample_period': .01},
                'lever_angle': {'sample_period': .01},
                'gyro': {'sample_period': .01},
            }), seed=27)
        np.testing.assert_array_equal(baseline.reset()[0], configured.reset()[0])
        for action in (.2, -.1, .0):
            first = baseline.step([action])
            second = configured.step([action])
            np.testing.assert_array_equal(baseline.y, configured.y)
            np.testing.assert_array_equal(first[0], second[0])


class CommandTimingTests(unittest.TestCase):
    def test_pending_commands_apply_in_issue_order(self):
        from timing_models import CommandDelayQueue

        queue = CommandDelayQueue(delay=.015, seed=3)
        first = queue.issue(1., 0.)
        second = queue.issue(2., .01)
        third = queue.issue(3., .02)
        fourth = queue.issue(4., .03)

        self.assertEqual(first['command_applied'], 0.)
        self.assertEqual(second['command_applied'], 0.)
        self.assertEqual(third['command_applied'], 1.)
        self.assertEqual(fourth['command_applied'], 2.)
        self.assertEqual([first['command_arrival_time'], second['command_arrival_time'],
                          third['command_arrival_time'], fourth['command_arrival_time']],
                         [.015, .025, .035, .045])

    def test_zero_delay_environment_reports_immediate_command_application(self):
        from rig_rl_env_v3 import RigRLEnvV3, RLEnvConfigV3

        env = RigRLEnvV3(cfg=RLEnvConfigV3(domain_randomization=False,
                                           kick_probability=0), seed=4)
        env.reset()
        _, _, _, _, info = env.step([.25])
        self.assertEqual(info['command_issue_time'], 0.)
        self.assertEqual(info['command_arrival_time'], 0.)
        self.assertEqual(info['command_applied_time'], 0.)
        self.assertAlmostEqual(info['command_requested'], info['command_applied'])
        self.assertFalse(info['command_pending'])

    def test_environment_holds_safe_command_until_delayed_command_arrives(self):
        from rig_rl_env_v3 import RigRLEnvV3, RLEnvConfigV3

        env = RigRLEnvV3(cfg=RLEnvConfigV3(domain_randomization=False,
                                           kick_probability=0,
                                           command_delay=.015), seed=8)
        env.reset()
        _, _, _, _, first = env.step([.5])
        _, _, _, _, second = env.step([-.5])
        _, _, _, _, third = env.step([.25])

        self.assertEqual(first['command_applied'], 0.)
        self.assertTrue(first['command_pending'])
        self.assertEqual(second['command_applied'], 0.)
        self.assertEqual(third['command_active_issue_time'], 0.)
        self.assertAlmostEqual(third['command_applied_time'], .02)

    def test_v4_reports_sensor_acquisition_and_arrival_times(self):
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        cfg = RLEnvConfigV4(
            domain_randomization=False, kick_probability=0, sensor_noise=False,
            sensor_timing={'gyro': {'sample_period': .02, 'delay': .015},
                           'motor_angle': {'sample_period': .01},
                           'lever_angle': {'sample_period': .01}})
        env = RigRLEnvV4(cfg=cfg, seed=9)
        env.reset()
        env.step([0.])
        _, _, _, _, info = env.step([0.])
        delayed_gyro = [packet for packet in info['sensor_packets']
                        if packet['channel'] == 'gyro'
                        and packet['acquisition_time'] == 0.0]
        self.assertEqual(len(delayed_gyro), 1)
        self.assertEqual(delayed_gyro[0]['arrival_time'], .015)
        self.assertGreaterEqual(info['sensor_messages_processed'], 1)
        self.assertGreaterEqual(info['sensor_delayed_measurements_processed'], 1)


if __name__ == '__main__':
    unittest.main()

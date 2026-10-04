import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))

class FormulationTests(unittest.TestCase):
    def test_reflection_preserves_margin(self):
        from ppo_agent_v2 import reflect_observation
        import torch
        x = torch.arange(12, dtype=torch.float32)[None]
        reflected = reflect_observation(x)
        self.assertEqual(float(reflected[0, 11]), 11.)
        self.assertEqual(float(reflected[0, 4]), -4.)
        torch.testing.assert_close(reflect_observation(reflected), x)

    def test_timeout_bootstraps_final_state_without_cross_reset_trace(self):
        from ppo_agent_v2 import RolloutBufferV2
        import torch
        b = RolloutBufferV2(2, 1, 12, 1, 'cpu')
        self.assertTrue(hasattr(b, 'truncated'))
        b.rewards[:] = 1.
        b.dones[0] = 1.
        b.truncated[0] = 1.
        b.final_values[0] = 10.
        b.compute_gae(torch.tensor([2.]), .9, 1.)
        self.assertAlmostEqual(float(b.returns[0]), 10., places=5)
        self.assertAlmostEqual(float(b.returns[1]), 2.8, places=5)

    def test_simulator_holds_command_for_control_interval(self):
        from active_vibration_rig_2d import Simulator, PlantParams, ControllerParams, MotionParams
        sim = Simulator(PlantParams(), ControllerParams(), MotionParams(), dt=.001, control_dt=.01)
        sim.controller.mode = 'lqr'
        sim.step(10)
        commands = [r['tau_cmd'] for r in sim.history]
        self.assertEqual(len(set(commands)), 1)

    def test_nonintegral_timing_rejected(self):
        from rig_rl_env_v3 import RLEnvConfigV3
        with self.assertRaises(ValueError):
            RLEnvConfigV3(physics_dt=.003, control_dt=.01)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_v2_eleven_signed_features_still_reflect(self):
        import torch
        from ppo_agent_v2 import reflect_observation
        x=torch.ones(2,11)
        torch.testing.assert_close(reflect_observation(x),-x)

if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'software/python'))

class ActorTests(unittest.TestCase):
    def test_value_loss_does_not_change_actor_gradients(self):
        import torch
        from ppo_agent_v2 import ActorCriticV4
        m=ActorCriticV4(208,1,128)
        m.value(torch.zeros(3,208)).square().add(m.value(torch.ones(3,208))).mean().backward()
        self.assertTrue(all(p.grad is None for p in m.trunk.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.critic_trunk.parameters()))

    def test_initial_exploration_is_bounded_and_small(self):
        import torch
        from ppo_agent_v2 import ActorCriticV4
        m=ActorCriticV4(208,1,128)
        d=m._dist(torch.zeros(1,208))
        self.assertLess(float(2*d.stddev[0,0].detach()),.2)


if __name__ == '__main__':
    unittest.main()

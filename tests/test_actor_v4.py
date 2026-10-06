import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'software/python'))

class ActorTests(unittest.TestCase):
    def test_ppo_diagnostics_measure_final_policy_against_rollout(self):
        import torch
        from ppo_agent_v2 import ActorCriticV4, PPOConfigV2, RolloutBufferV2, ppo_update_v2

        torch.manual_seed(41)
        model=ActorCriticV4(4,1,16)
        buffer=RolloutBufferV2(8,1,4,1,'cpu')
        buffer.obs[:,0]=torch.tensor([
            [1.,0.,0.,0.],[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,-1.,0.,0.],
            [0.,0.,1.,0.],[0.,0.,-1.,0.],[0.,0.,0.,1.],[0.,0.,0.,-1.],
        ])
        buffer.actions[:,0,0]=torch.tensor([-.7,-.4,-.2,0.,.2,.4,.6,.8])
        with torch.no_grad():
            old_lp,_,_=model.evaluate_actions(buffer.obs[:,0],buffer.actions[:,0])
        buffer.logp[:,0]=old_lp
        buffer.advantages[:,0]=torch.tensor([1.,-.7,.6,-.5,.4,-.3,.2,-.1])
        buffer.returns[:,0]=0.
        cfg=PPOConfigV2(epochs=2,minibatch_size=4,clip_ratio=.1,
                        value_coef=0.,entropy_coef=0.,anchor_kl_coef=0.,
                        symmetry_coef=0.,target_kl=1e6)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)

        logged=ppo_update_v2(model,optimizer,buffer,cfg,reflection_signs=[1.,1.,1.,1.])

        with torch.no_grad():
            final_lp,_,_=model.evaluate_actions(buffer.obs[:,0],buffer.actions[:,0])
            logratio=final_lp-old_lp
            expected_kl=float((logratio.exp()-1-logratio).mean())
            expected_clipfrac=float(((logratio.exp()-1).abs()>.1).float().mean())
        self.assertGreater(expected_kl,1e-5)
        self.assertAlmostEqual(logged['kl'],expected_kl,places=6)
        self.assertAlmostEqual(logged['clipfrac'],expected_clipfrac,places=6)

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

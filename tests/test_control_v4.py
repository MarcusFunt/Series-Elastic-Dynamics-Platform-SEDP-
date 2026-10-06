import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))

class ControlV4Tests(unittest.TestCase):
    def test_gate_energy_reward_matches_evaluation_increment(self):
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4

        common=dict(domain_randomization=False,kick_probability=0,
                    initial_theta_std=0,initial_theta_dot_std=0,sensor_noise=False)
        baseline=RigRLEnvV4(cfg=RLEnvConfigV4(**common),seed=7)
        aligned=RigRLEnvV4(cfg=RLEnvConfigV4(**common,
            energy_gate_weight=.7,energy_gate_step_scale_mJs=.1),seed=7)
        baseline.reset();aligned.reset()
        for action in (.1,-.2,0.):
            _,old_reward,_,_,old_info=baseline.step([action])
            _,new_reward,_,_,info=aligned.step([action])
            self.assertAlmostEqual(info['energy'],old_info['energy'])
            increment=info['energy']*aligned.cfg.control_dt*1000
            expected_cost=.7*increment/.1
            self.assertAlmostEqual(info['cost_gate_energy'],expected_cost)
            self.assertAlmostEqual(old_reward-new_reward,
                                   expected_cost*aligned.cfg.reward_scale)

    def test_rollout_diagnostics_reconcile_reward_and_executed_residual(self):
        from train_residual_v4 import rollout_diagnostics
        from rig_rl_env_v4 import RLEnvConfigV4

        cfg=RLEnvConfigV4()
        infos=[{'requested_residual_accel':.1,'effective_residual_accel':.08,
                'energy_progress_reward':.3,'cost_pos':.4,'cost_gate_energy':.2,
                'unscaled_reward':cfg.alive_bonus-.4-.2+.3,'terminated':False},
               {'requested_residual_accel':-.1,'effective_residual_accel':-.05,
                'energy_progress_reward':-.1,'cost_pos':.2,'cost_gate_energy':.1,
                'unscaled_reward':cfg.alive_bonus-.2-.1-.1-50.,'terminated':True}]
        stats=rollout_diagnostics(np.array([[.2],[-.4]]),
                                  np.array([[.1],[-.3]]),infos,cfg)
        self.assertAlmostEqual(stats['action_sample_mean'],-.1)
        self.assertAlmostEqual(stats['action_mean_abs'],.2)
        self.assertAlmostEqual(stats['executed_residual_accel_mean_abs'],.065)
        self.assertAlmostEqual(stats['reward_components']['termination_penalty'],-25.)
        self.assertAlmostEqual(stats['reward_reconstruction_error'],0.)

    def test_measurements_do_not_expose_true_torque_or_carriage(self):
        from state_estimator import MeasurementModel
        from active_vibration_rig_2d import PlantParams
        m=MeasurementModel(PlantParams(), noise=False)
        y=np.array([7., 1., .03, .2, -.02, .3, .4])
        z=m.sample(y, 0.)
        self.assertEqual(set(z), {'motor_angle', 'lever_angle', 'gyro', 'time'})

    def test_estimator_tracks_encoder_wrap(self):
        from state_estimator import MeasurementModel, StateEstimator
        from active_vibration_rig_2d import PlantParams
        p=PlantParams(); m=MeasurementModel(p, noise=False); e=StateEstimator(p, .001, .01)
        y=np.array([0.,0.,0.,0.,-.02,.1,0.])
        for i in range(10):
            e.update(m.sample(y, i*.01))
        self.assertLess(abs(e.state[4]+.02), .003)
        self.assertLess(abs(e.state[5]-.1), .02)
        self.assertTrue(np.all(np.linalg.eigvalsh(e.covariance)>0))

    def test_context_preview_does_not_advance_random_reference(self):
        from rig_rl_env_v3 import SmoothRandomReference, RLEnvConfigV3
        r=SmoothRandomReference(np.random.default_rng(2), RLEnvConfigV3(), .065)
        before=(r.t0,r.x1,r.change_count,str(r.rng.bit_generator.state))
        r.preview(5.)
        self.assertEqual(before,(r.t0,r.x1,r.change_count,str(r.rng.bit_generator.state)))

    def test_history_reflection_and_reset(self):
        from residual_control_v4 import ObservationHistory, FRAME_SIGNS
        h=ObservationHistory(3)
        a=np.arange(len(FRAME_SIGNS),dtype=np.float32)
        h.reset(a); obs=h.append(a+1)
        self.assertEqual(obs.shape,(3*len(FRAME_SIGNS),))
        reflected=obs*np.tile(FRAME_SIGNS,3)
        self.assertEqual(reflected[11],obs[11])
        self.assertEqual(reflected[12],-obs[12])
        h.reset(np.zeros_like(a)); self.assertEqual(float(h.value.sum()),0.)

    def test_mpc_returns_bounded_teacher_action_and_handles_failure(self):
        from constrained_mpc import ConstrainedMPC, MPCConfig
        from active_vibration_rig_2d import PlantParams, ControllerParams
        p=PlantParams(); m=ConstrainedMPC(p, ControllerParams(), MPCConfig(horizon=8))
        y=np.zeros(7);y[4]=.03
        action=m.action(y,[(0.,0.,0.)]*9)
        self.assertLessEqual(abs(action),1.)
        self.assertTrue(np.isfinite(action))
        self.assertIn('success',m.diagnostics)
        with self.assertRaises(ValueError): m.action(y,[])

    def test_v4_training_evaluation_use_same_transition(self):
        from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4
        c=RLEnvConfigV4(domain_randomization=False,kick_probability=0,initial_theta_std=0,initial_theta_dot_std=0,sensor_noise=False)
        a=RigRLEnvV4(cfg=c,seed=1);b=RigRLEnvV4(cfg=c,seed=1)
        oa,_=a.reset();ob,_=b.reset()
        np.testing.assert_array_equal(oa,ob)
        for action in [.1,-.2,0.]:
            oa,_,_,_,_=a.step([action]);ob,_,_,_,_=b.step([action])
            np.testing.assert_array_equal(a.y,b.y)
            np.testing.assert_array_equal(oa,ob)
        self.assertEqual(oa.shape,(a.observation_dim,))


class RuntimeParityTests(unittest.TestCase):
    def test_runtime_adapter_matches_environment_commands_and_states(self):
        import torch
        from ppo_agent_v2 import ActorCriticV2
        from rig_rl_policy_v4 import EstimatedResidualPolicyV4
        from rig_rl_env_v4 import RigRLEnvV4,RLEnvConfigV4
        from active_vibration_rig_2d import Simulator,Trajectory,MotionParams
        cfg=RLEnvConfigV4(domain_randomization=False,kick_probability=0,initial_theta_std=0,initial_theta_dot_std=0,sensor_noise=False)
        env=RigRLEnvV4(cfg=cfg,seed=0);env.reset()
        reference=Trajectory('aggressive',MotionParams())
        obs=env.set_reference(reference)
        model=ActorCriticV2(env.observation_dim,1,128).eval()
        sim=Simulator(env.base_params,env.cp,MotionParams(),dt=cfg.physics_dt,control_dt=cfg.control_dt)
        sim.trajectory=reference
        adapter=EstimatedResidualPolicyV4(model,cfg,env.base_params,env.cp,reference)
        sim.controller.set_rl_policy(adapter);sim.controller.mode='ppo'
        for _ in range(5):
            with torch.no_grad():action=model.deterministic(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            obs,*_=env.step(action);sim.step(cfg.substeps)
            np.testing.assert_allclose(sim.y,env.y,rtol=0,atol=1e-12)
            self.assertAlmostEqual(sim._held_command,env.last_u,places=12)

class SafetyAndSchemaTests(unittest.TestCase):
    def test_mpc_infeasible_state_falls_back(self):
        from constrained_mpc import ConstrainedMPC,MPCConfig
        from active_vibration_rig_2d import PlantParams,ControllerParams
        m=ConstrainedMPC(PlantParams(),ControllerParams(),MPCConfig(horizon=8))
        y=np.zeros(7);y[2]=.08
        self.assertEqual(m.action(y,[(0.,0.,0.)]*9),0.)
        self.assertFalse(m.diagnostics['success'])

    def test_checkpoint_round_trip_preserves_v4_architecture(self):
        import tempfile
        import torch
        from dataclasses import asdict
        from ppo_agent_v2 import ActorCriticV4,PPOConfigV2,save_checkpoint_v2
        from evaluate_v4 import load_policy
        from rig_rl_env_v4 import RLEnvConfigV4
        cfg=RLEnvConfigV4(history_length=2);model=ActorCriticV4(52,1,128)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt'
            save_checkpoint_v2(path,model,PPOConfigV2(),asdict(cfg),{'observation_schema':'sedp-v4-estimated-context-history'})
            restored,loaded,_=load_policy(path)
            self.assertIsInstance(restored,ActorCriticV4)
            self.assertEqual(loaded.history_length,2)
            torch.testing.assert_close(model.deterministic(torch.zeros(1,52)),restored.deterministic(torch.zeros(1,52)))

    def test_nominal_parameters_are_not_a_live_ground_truth_reference(self):
        from residual_control_v4 import ResidualControlLoop
        from rig_rl_env_v4 import RLEnvConfigV4
        from active_vibration_rig_2d import PlantParams,ControllerParams
        p=PlantParams();loop=ResidualControlLoop(p,ControllerParams(),RLEnvConfigV4())
        original=loop.estimator.params.k_theta
        p.k_theta*=2
        self.assertEqual(loop.estimator.params.k_theta,original)

class PromotionTests(unittest.TestCase):
    def row(self,controller):
        return {'controller':controller,'scenario':'step','seed':0,'randomized':False,
                'peak_angle_deg':1.,'position_rmse_mm':6.,'peak_rail_fraction':.6,
                'saturation_fraction':0.,'stop_contact_fraction':0.,'energy_integral_mJs':1.,'terminated':False}

    def test_nonfinite_metrics_cannot_promote(self):
        from evaluate_v4 import promotion
        base=self.row('lqr');candidate=self.row('policy');candidate['energy_integral_mJs']=float('nan')
        self.assertFalse(promotion([base,candidate])['accepted'])

    def test_lower_energy_does_not_excuse_peak_regression(self):
        from evaluate_v4 import promotion
        base=self.row('lqr');candidate=self.row('policy')
        candidate['energy_integral_mJs']=.8;candidate['peak_angle_deg']=1.2
        self.assertFalse(promotion([base,candidate])['accepted'])

    def test_valid_paired_improvement_promotes(self):
        from evaluate_v4 import promotion
        base=self.row('lqr');candidate=self.row('policy');candidate['energy_integral_mJs']=.8
        self.assertTrue(promotion([base,candidate])['accepted'])

if __name__ == '__main__':
    unittest.main()

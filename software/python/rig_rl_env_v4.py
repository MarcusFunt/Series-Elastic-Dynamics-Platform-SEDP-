"""Estimated-state, context-conditioned residual environment.

Evaluation and training both call this transition. Reward/metrics alone use
true state. Parameters available to controller/observer remain nominal.
"""
from dataclasses import dataclass
import math
import numpy as np
from rig_rl_env_v3 import RLEnvConfigV3, RigRLEnvV3
from residual_control_v4 import ResidualControlLoop, FRAME_SIGNS


@dataclass
class RLEnvConfigV4(RLEnvConfigV3):
    history_length: int = 8
    sensor_noise: bool = True
    linear_encoder: bool = False
    oracle_state: bool = False
    observation_noise_std: float = 0.
    reward_scale: float = .02

    def __post_init__(self):
        super().__post_init__()
        if self.history_length < 1: raise ValueError('history_length must be positive')
        if self.reward_scale<=0: raise ValueError('reward_scale must be positive')
        if self.residual_accel_limit <= 0: raise ValueError('residual_accel_limit must be positive')


class RigRLEnvV4(RigRLEnvV3):
    def __init__(self, base_params=None, controller_params=None, cfg=None, seed=0):
        super().__init__(base_params,controller_params,cfg or RLEnvConfigV4(),seed)
        self.observation_dim=len(FRAME_SIGNS)*self.cfg.history_length
        self.reflection_signs=np.tile(FRAME_SIGNS,self.cfg.history_length)
        self.loop=None

    def _base_accel(self, ref): return self.loop.base_accel(ref)

    def _observation(self, ref):
        # Called once at reset by the inherited reset routine.
        self.loop=ResidualControlLoop(self.base_params,self.cp,self.cfg,
                                     self.seed_value+7919,measurement_params=self.p)
        self._current_obs=self.loop.observe(self.y,self.t,self.reference,initial=True)
        return self._current_obs

    def set_reference(self, reference):
        """Change announced motion without assimilating a duplicate measurement."""
        self.reference=reference
        self._current_obs=self.loop.history.reset(self.loop.frame(reference,self.t))
        return self._current_obs

    def step(self, action):
        action=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1))
        if not np.isfinite(action): raise ValueError('Action must be finite')
        ref=self.reference.sample(self.t)
        torque=self.loop.command(action,ref)
        command=self.loop.last_command
        self.last_base_accel=command['base_accel'];self.last_residual=command['requested_residual_accel']
        self.last_effective_residual=command['effective_residual_accel'];self.last_u=torque
        interval_peak_rail=0.;interval_saturation=False
        for _ in range(self.cfg.substeps):
            ext=self.kick_torque if self.kick_at>=0 and self.kick_at<=self.t<self.kick_at+self.kick_duration else 0.
            self.y=self.plant.rk4(self.y,torque,self.cfg.physics_dt,external_torque=ext)
            self.t+=self.cfg.physics_dt
            interval_peak_rail=max(interval_peak_rail,abs(self.y[2])/self.p.rail_half_travel)
            interval_saturation=interval_saturation or abs(torque)>=.995*self.plant.motor_torque_limit(self.y[1])
        self.loop.predict(torque)
        self.steps+=1
        ref=self.reference.sample(self.t)
        effective=self.loop.previous_effective_action
        reward,costs,pred,progress=self._reward(effective,ref,command['base_accel'],command['effective_residual_accel'])
        self.prev_energy=self._energy_norm();self.prev_action=effective
        term=bool(interval_peak_rail>self.cfg.terminate_rail_fraction or
                  abs(self.y[4])>self.cfg.terminate_theta or not np.all(np.isfinite(self.y)))
        trunc=self.steps>=self.cfg.max_steps
        if term:reward-=50.
        self._current_obs=self.loop.observe(self.y,self.t,self.reference)
        info=self._info(ref,reward,costs)
        info.update(command)
        info['interval_peak_rail_fraction']=interval_peak_rail
        info['interval_saturation']=interval_saturation
        info.update({'requested_action':action,'energy_progress_reward':progress,
                     'estimator_position_error':float(self.loop.estimator.state[2]-self.y[2]),
                     'estimator_angle_error':float(self.loop.estimator.state[4]-self.y[4]),
                     'estimator_innovation':self.loop.estimator.innovation_norm})
        info['unscaled_reward']=reward
        return self._current_obs,reward*self.cfg.reward_scale,term,trunc,info


class VectorRigEnvV4:
    def __init__(self,n_envs=8,seed=0,cfg=None):
        self.envs=[RigRLEnvV4(cfg=cfg,seed=seed+1009*i) for i in range(n_envs)]
        self.n_envs=n_envs; self.obs_dim=self.envs[0].observation_dim;self.action_dim=1
        self.reflection_signs=self.envs[0].reflection_signs

    def reset(self): return np.stack([e.reset()[0] for e in self.envs])

    def step(self,actions):
        obs=[];rewards=[];dones=[];infos=[]
        for env,action in zip(self.envs,actions):
            o,r,te,tr,info=env.step(action);done=te or tr
            info.update(terminated=te,truncated=tr and not te)
            if done:
                info['final_observation']=o.copy();o,_=env.reset()
            obs.append(o);rewards.append(r);dones.append(done);infos.append(info)
        return np.stack(obs),np.asarray(rewards,np.float32),np.asarray(dones,np.float32),infos

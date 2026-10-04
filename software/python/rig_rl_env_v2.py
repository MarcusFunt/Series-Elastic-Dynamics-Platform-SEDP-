#!/usr/bin/env python3
"""Improved residual-RL environment for the active vibration rig.

Main changes versus v1:
- potential-based resonator-energy shaping (reward actual energy removal)
- event/settling rewards after target transitions
- stopping-distance rail barrier + state-dependent residual safety scaling
- curriculum-friendly reference parameters
- moderate domain randomization and disturbance injection
- same 11-D observation and 1-D bounded residual action, so old warm-start
  checkpoints remain compatible.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict, replace
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from active_vibration_rig_2d import PlantParams, ControllerParams, RigPlant, clamp

@dataclass
class RLEnvConfigV2:
    physics_dt: float = 0.0005
    control_dt: float = 0.0100
    episode_seconds: float = 6.0
    action_mode: str = "residual"
    residual_torque_limit: float = 0.30
    direct_torque_limit: float = 0.46
    min_target_hold: float = 0.55
    max_target_hold: float = 1.00
    max_target_fraction: float = 0.65
    reversal_probability: float = 0.70
    max_reference_speed: float = 0.75
    max_reference_accel: float = 12.0
    initial_theta_std: float = math.radians(1.5)
    initial_theta_dot_std: float = math.radians(7.0)
    kick_probability: float = 0.35
    kick_torque_min: float = 0.008
    kick_torque_max: float = 0.035
    kick_duration_min: float = 0.020
    kick_duration_max: float = 0.075
    domain_randomization: bool = True
    mass_factor: Tuple[float,float] = (0.85, 1.20)
    stiffness_factor: Tuple[float,float] = (0.82, 1.20)
    damping_factor: Tuple[float,float] = (0.65, 1.45)
    belt_stiffness_factor: Tuple[float,float] = (0.80, 1.22)
    rail_damping_factor: Tuple[float,float] = (0.82, 1.22)
    motor_torque_factor: Tuple[float,float] = (0.92, 1.08)
    position_scale: float = 0.018
    velocity_scale: float = 0.55
    theta_scale: float = math.radians(8.0)
    theta_dot_scale: float = math.radians(150.0)
    w_position: float = 1.10
    w_velocity: float = 0.05
    w_theta: float = 2.10
    w_theta_dot: float = 0.55
    w_energy_abs: float = 0.75
    w_energy_progress: float = 3.00
    w_action: float = 0.018
    w_action_delta: float = 0.045
    w_rail: float = 7.0
    alive_bonus: float = 0.18
    settle_delay: float = 0.16
    settle_pos_tol: float = 0.0040
    settle_theta_tol: float = math.radians(1.2)
    settle_theta_dot_tol: float = math.radians(12.0)
    settled_bonus: float = 0.45
    transition_bonus: float = 2.5
    rail_soft_fraction: float = 0.72
    rail_hard_fraction: float = 0.96
    assumed_braking_accel: float = 7.5
    rail_velocity_buffer: float = 0.004
    rail_terminate_fraction: float = 1.02
    theta_terminate: float = math.radians(70.0)
    observation_noise_std: float = 0.0015
    reward_gamma: float = 0.995

    @property
    def substeps(self) -> int:
        return max(1, int(round(self.control_dt/self.physics_dt)))
    @property
    def max_steps(self) -> int:
        return max(1, int(round(self.episode_seconds/self.control_dt)))

class RandomReferenceV2:
    def __init__(self, rng: np.random.Generator, cfg: RLEnvConfigV2, half_travel: float):
        self.rng=rng; self.cfg=cfg; self.half_travel=half_travel
        self.target=0.0; self.prev_target=0.0
        self.next_change=0.0; self.last_change=0.0; self.change_count=0
        self._schedule_next(0.0, first=True)
    def _new_target(self) -> float:
        A=self.cfg.max_target_fraction*self.half_travel
        if self.rng.random() < self.cfg.reversal_probability:
            if abs(self.target) < 0.20*A:
                return A if self.rng.random()<0.5 else -A
            return -math.copysign(A,self.target)
        mag=float(self.rng.uniform(0.35*A,A))
        return mag if self.rng.random()<0.5 else -mag
    def _schedule_next(self,t:float,first:bool=False):
        self.prev_target=self.target
        self.target=0.0 if first else self._new_target()
        self.last_change=t
        if not first: self.change_count += 1
        self.next_change=t+float(self.rng.uniform(self.cfg.min_target_hold,self.cfg.max_target_hold))
    def sample(self,t:float):
        changed=False
        if t>=self.next_change:
            self._schedule_next(t); changed=True
        return (self.target,0.0,0.0), changed

class RigRLEnvV2:
    observation_dim=11; action_dim=1
    def __init__(self, base_params:Optional[PlantParams]=None, controller_params:Optional[ControllerParams]=None,
                 cfg:Optional[RLEnvConfigV2]=None, seed:int=0):
        self.base_params=base_params or PlantParams(); self.cp=controller_params or ControllerParams(); self.cfg=cfg or RLEnvConfigV2()
        self.rng=np.random.default_rng(seed); self.seed_value=seed
        self.p=self.base_params; self.plant=RigPlant(self.p); self.y=np.zeros(7,dtype=float)
        self.t=0.0; self.steps=0; self.prev_action=0.0; self.last_u=0.0
        self.reference=RandomReferenceV2(self.rng,self.cfg,self.p.rail_half_travel)
        self.kick_at=-1.0; self.kick_torque=0.0; self.kick_duration=0.0
        self.prev_energy_norm=0.0; self.last_info={}

    def _f(self,b): return float(self.rng.uniform(*b))
    def _randomized_params(self):
        p=replace(self.base_params)
        if not self.cfg.domain_randomization: return p
        p.resonator_mass*=self._f(self.cfg.mass_factor); p.k_theta*=self._f(self.cfg.stiffness_factor); p.c_theta*=self._f(self.cfg.damping_factor)
        p.belt_stiffness*=self._f(self.cfg.belt_stiffness_factor); p.b_x*=self._f(self.cfg.rail_damping_factor); p.motor_hold_torque*=self._f(self.cfg.motor_torque_factor)
        return p

    def _energy_norm(self):
        p,c=self.p,self.cfg; J=max(p.resonator_inertia_pivot,1e-9); k=max(abs(p.effective_small_angle_stiffness),0.02)
        E=0.5*J*self.y[5]**2+0.5*k*self.y[4]**2
        Eref=0.5*J*c.theta_dot_scale**2+0.5*k*c.theta_scale**2
        return float(E/max(Eref,1e-9)), float(E)

    def reset(self,seed:Optional[int]=None):
        if seed is not None: self.rng=np.random.default_rng(seed); self.seed_value=seed
        self.p=self._randomized_params(); self.plant=RigPlant(self.p); self.y=np.zeros(7,dtype=float)
        self.y[4]=float(self.rng.normal(0,self.cfg.initial_theta_std)); self.y[5]=float(self.rng.normal(0,self.cfg.initial_theta_dot_std))
        self.t=0.0; self.steps=0; self.prev_action=0.0; self.last_u=0.0
        self.reference=RandomReferenceV2(self.rng,self.cfg,self.p.rail_half_travel)
        if self.rng.random()<self.cfg.kick_probability:
            self.kick_at=float(self.rng.uniform(0.15,1.25)); self.kick_duration=float(self.rng.uniform(self.cfg.kick_duration_min,self.cfg.kick_duration_max))
            self.kick_torque=float(self.rng.uniform(self.cfg.kick_torque_min,self.cfg.kick_torque_max))*(1 if self.rng.random()<0.5 else -1)
        else: self.kick_at=-1.; self.kick_duration=0.; self.kick_torque=0.
        ref,_=self.reference.sample(self.t); self.prev_energy_norm,_=self._energy_norm()
        obs=self._observation(ref); return obs,self._info(ref,0.0,{})

    def _servo_torque(self,ref):
        xr,vr,ar=ref; p,cp=self.p,self.cp
        F=cp.x_kp*(xr-self.y[2])+cp.x_kd*(vr-self.y[3])+(p.carriage_mass+p.resonator_mass)*ar
        return p.pulley_radius*F

    def _safety_scale(self,residual:float)->float:
        p,c=self.p,self.cfg; L=max(p.rail_half_travel,1e-9); x=float(self.y[2]); v=float(self.y[3])
        frac=abs(x)/L
        if frac<=c.rail_soft_fraction: return 1.0
        span=max(c.rail_hard_fraction-c.rail_soft_fraction,1e-6)
        scale=float(np.clip((c.rail_hard_fraction-frac)/span,0.08,1.0))
        outward=(x*residual>0) or (x*v>0 and v*residual>0)
        return scale if outward else min(1.0,0.35+0.65*scale)

    def _apply_action(self,action,ref):
        a=clamp(float(action),-1.,1.)
        if self.cfg.action_mode=='residual':
            r=a*self.cfg.residual_torque_limit
            r*=self._safety_scale(r)
            return self._servo_torque(ref)+r
        return a*self.cfg.direct_torque_limit

    def _observation(self,ref):
        p,c=self.p,self.cfg; xr,vr,ar=ref; rp=max(p.pulley_radius,1e-6); L=max(p.rail_half_travel,1e-6)
        vals=np.array([rp*self.y[0]/L,rp*self.y[1]/c.max_reference_speed,self.y[2]/L,self.y[3]/c.max_reference_speed,
            self.y[4]/c.theta_scale,self.y[5]/c.theta_dot_scale,self.y[6]/max(p.motor_hold_torque,1e-6),xr/L,
            vr/c.max_reference_speed,ar/c.max_reference_accel,self.prev_action],dtype=np.float32)
        vals=np.clip(vals,-8,8)
        if c.observation_noise_std>0: vals += self.rng.normal(0,c.observation_noise_std,size=vals.shape).astype(np.float32)
        return vals

    def _rail_barrier(self):
        p,c=self.p,self.cfg; L=max(p.rail_half_travel,1e-9); x=float(self.y[2]); v=float(self.y[3])
        outward_v=max(0.0, math.copysign(1.0,x if abs(x)>1e-9 else v)*v)
        stop_d=outward_v*outward_v/(2*max(c.assumed_braking_accel,1e-3))
        predicted=(abs(x)+stop_d+c.rail_velocity_buffer)/L
        z=max(0.0,(predicted-c.rail_soft_fraction)/max(c.rail_hard_fraction-c.rail_soft_fraction,1e-6))
        return min(z*z*(1+z*z), 25.0), predicted

    def _base_costs(self,a,ref):
        p,c=self.p,self.cfg; xr,vr,_=ref
        ex=(self.y[2]-xr)/c.position_scale; ev=(self.y[3]-vr)/c.velocity_scale; eth=self.y[4]/c.theta_scale; ew=self.y[5]/c.theta_dot_scale
        En,_=self._energy_norm(); rail, pred=self._rail_barrier(); da=a-self.prev_action
        return {'position':c.w_position*ex*ex,'velocity':c.w_velocity*ev*ev,'theta':c.w_theta*eth*eth,'theta_dot':c.w_theta_dot*ew*ew,
                'energy_abs':c.w_energy_abs*En,'action':c.w_action*a*a,'action_delta':c.w_action_delta*da*da,'rail':c.w_rail*rail}, pred

    def _settling_bonus(self,ref):
        c=self.cfg
        if self.t-self.reference.last_change<c.settle_delay: return 0.0
        ok=(abs(self.y[2]-ref[0])<c.settle_pos_tol and abs(self.y[4])<c.settle_theta_tol and abs(self.y[5])<c.settle_theta_dot_tol)
        return c.settled_bonus if ok else 0.0

    def _transition_score(self,old_target):
        c=self.cfg
        ex=(self.y[2]-old_target)/max(c.settle_pos_tol,1e-6); eth=self.y[4]/max(c.settle_theta_tol,1e-6); ew=self.y[5]/max(c.settle_theta_dot_tol,1e-6)
        return c.transition_bonus*math.exp(-0.55*(ex*ex+eth*eth+0.45*ew*ew))

    def _info(self,ref,reward,costs):
        En,E=self._energy_norm(); rail,pred=self._rail_barrier()
        return {'t':self.t,'reward':reward,'x':float(self.y[2]),'x_ref':float(ref[0]),'theta':float(self.y[4]),'theta_dot':float(self.y[5]),
                'energy':E,'energy_norm':En,'tau_cmd':float(self.last_u),'rail_fraction':float(abs(self.y[2])/max(self.p.rail_half_travel,1e-9)),
                'predicted_rail_fraction':float(pred),'target_changes':float(self.reference.change_count),**{f'cost_{k}':float(v) for k,v in costs.items()}}

    def step(self,action:Sequence[float]|np.ndarray|float):
        a=clamp(float(np.asarray(action).reshape(-1)[0]),-1.,1.)
        ref,_=self.reference.sample(self.t); u=self._apply_action(a,ref); self.last_u=u
        for _ in range(self.cfg.substeps):
            ext= self.kick_torque if self.kick_at>=0 and self.kick_at<=self.t<self.kick_at+self.kick_duration else 0.0
            self.y=self.plant.rk4(self.y,u,self.cfg.physics_dt,external_torque=ext); self.t+=self.cfg.physics_dt
        self.steps+=1
        old_target=self.reference.target
        ref,changed=self.reference.sample(self.t)
        costs,pred=self._base_costs(a,ref); cost=sum(costs.values())
        En,_=self._energy_norm()
        energy_progress=self.cfg.w_energy_progress*(self.prev_energy_norm-self.cfg.reward_gamma*En)
        reward=float(self.cfg.alive_bonus-cost+energy_progress+self._settling_bonus(ref))
        if changed: reward += self._transition_score(self.reference.prev_target)
        self.prev_energy_norm=En
        rail_frac=abs(self.y[2])/max(self.p.rail_half_travel,1e-9)
        terminated=bool(rail_frac>self.cfg.rail_terminate_fraction or abs(self.y[4])>self.cfg.theta_terminate or not np.all(np.isfinite(self.y)))
        truncated=bool(self.steps>=self.cfg.max_steps)
        if terminated: reward-=25.0
        self.prev_action=a; obs=self._observation(ref); info=self._info(ref,reward,costs); info['energy_progress_reward']=energy_progress; info['transition']=float(changed)
        self.last_info=info; return obs,reward,terminated,truncated,info

    def config_dict(self): return asdict(self.cfg)

class VectorRigEnvV2:
    def __init__(self,n_envs:int,seed:int=0,cfg:Optional[RLEnvConfigV2]=None):
        self.envs=[RigRLEnvV2(cfg=cfg,seed=seed+1009*i) for i in range(n_envs)]; self.n_envs=n_envs; self.obs_dim=11; self.action_dim=1
    def reset(self): return np.stack([e.reset()[0] for e in self.envs])
    def step(self,actions):
        obs=[]; rew=[]; done=[]; infos=[]
        for i,e in enumerate(self.envs):
            o,r,term,trunc,info=e.step(actions[i]); d=term or trunc
            if d:
                info=dict(info); info['episode_done']=1.0; o,_=e.reset()
            obs.append(o); rew.append(r); done.append(d); infos.append(info)
        return np.stack(obs),np.asarray(rew,np.float32),np.asarray(done,np.float32),infos

if __name__=='__main__':
    e=RigRLEnvV2(seed=1); o,i=e.reset(); R=0
    for _ in range(600):
        o,r,t,tr,i=e.step([0]); R+=r
        if t or tr: break
    print('smoke',R,math.degrees(i['theta']),i['predicted_rail_fraction'])

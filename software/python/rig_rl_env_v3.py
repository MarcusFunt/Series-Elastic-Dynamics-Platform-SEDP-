#!/usr/bin/env python3
"""SEDP residual-RL v3 environment.

v3 fixes the main formulation error in v2: the learned action is no longer raw
motor torque. PPO outputs a bounded residual carriage acceleration on top of
the already constrained reduced-LQR controller. The deterministic controller
retains trajectory and rail-safety responsibilities; RL only makes small
corrections inside the same control-barrier projection.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np

from active_vibration_rig_2d import (
    Controller, ControllerParams, PlantParams, RigPlant, StepDirActuator,
    StepDirParams, clamp,
)
from timing_models import CommandDelayQueue


@dataclass
class RLEnvConfigV3:
    physics_dt: float = 0.001
    control_dt: float = 0.010
    episode_seconds: float = 6.0
    residual_accel_limit: float = 2.5
    actuator_mode: str = 'torque'
    step_dir: StepDirParams = field(default_factory=StepDirParams)
    command_delay: float = 0.0
    command_jitter: float = 0.0

    segment_time_min: float = 0.55
    segment_time_max: float = 0.90
    move_fraction_min: float = 0.62
    move_fraction_max: float = 0.82
    target_fraction_min: float = 0.42
    target_fraction_max: float = 0.64
    instant_step_probability: float = 0.10

    domain_randomization: bool = True
    mass_factor: Tuple[float,float] = (0.86, 1.18)
    stiffness_factor: Tuple[float,float] = (0.82, 1.22)
    damping_factor: Tuple[float,float] = (0.65, 1.45)
    belt_stiffness_factor: Tuple[float,float] = (0.82, 1.20)
    rail_damping_factor: Tuple[float,float] = (0.82, 1.20)
    motor_torque_factor: Tuple[float,float] = (0.92, 1.08)
    initial_theta_std: float = math.radians(1.0)
    initial_theta_dot_std: float = math.radians(5.0)
    kick_probability: float = 0.30
    kick_torque_min: float = 0.006
    kick_torque_max: float = 0.025
    kick_duration_min: float = 0.02
    kick_duration_max: float = 0.06

    pos_scale: float = 0.006
    vel_scale: float = 0.35
    theta_scale: float = math.radians(5.0)
    theta_dot_scale: float = math.radians(100.0)
    accel_scale: float = 11.0
    motor_speed_scale: float = 60.0

    w_pos: float = 1.8
    w_vel: float = 0.12
    w_theta: float = 2.4
    w_theta_dot: float = 0.55
    w_energy: float = 0.65
    w_energy_progress: float = 2.5
    w_action: float = 0.030
    w_action_delta: float = 0.060
    w_adverse_tracking: float = 0.0
    w_rail: float = 12.0
    alive_bonus: float = 0.25
    reward_gamma: float = 0.995

    rail_soft_fraction: float = 0.72
    rail_hard_fraction: float = 0.80
    assumed_braking_accel: float = 8.0
    rail_velocity_buffer: float = 0.003
    terminate_rail_fraction: float = 0.82
    terminate_theta: float = math.radians(55.0)
    observation_noise_std: float = 0.0010
    effective_action_observation: bool = True

    def __post_init__(self):
        if self.actuator_mode not in ('torque', 'step_dir'):
            raise ValueError("actuator_mode must be 'torque' or 'step_dir'")
        if isinstance(self.step_dir, dict):
            self.step_dir = StepDirParams(**self.step_dir)
        elif not isinstance(self.step_dir, StepDirParams):
            raise ValueError('step_dir must be a StepDirParams instance or serialized mapping')
        if self.physics_dt <= 0 or self.control_dt <= 0 or self.episode_seconds <= 0:
            raise ValueError("Timing values must be positive")
        if (not math.isfinite(self.command_delay) or not math.isfinite(self.command_jitter)
                or self.command_delay < 0 or self.command_jitter < 0):
            raise ValueError('command_delay and command_jitter must be finite and nonnegative')
        if not math.isclose(self.control_dt/self.physics_dt, round(self.control_dt/self.physics_dt), abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if not math.isclose(self.episode_seconds/self.control_dt, round(self.episode_seconds/self.control_dt), abs_tol=1e-9):
            raise ValueError("episode_seconds must be an integer multiple of control_dt")

    @property
    def substeps(self) -> int:
        return max(1, int(round(self.control_dt/self.physics_dt)))

    @property
    def max_steps(self) -> int:
        return max(1, int(round(self.episode_seconds/self.control_dt)))


class SmoothRandomReference:
    """Random minimum-jerk point-to-point reference with occasional hard steps."""

    def __init__(self, rng: np.random.Generator, cfg: RLEnvConfigV3, half_travel: float):
        self.rng=rng; self.cfg=cfg; self.L=half_travel
        self.x0=0.0; self.x1=0.0; self.t0=0.0; self.T=0.7; self.Tmove=0.5
        self.instant=False; self.change_count=0
        self._new_segment(0.0, first=True)

    def _new_target(self) -> float:
        A=self.L*float(self.rng.uniform(self.cfg.target_fraction_min,self.cfg.target_fraction_max))
        if abs(self.x1) < 0.1*A:
            return A if self.rng.random()<0.5 else -A
        if self.rng.random()<0.82:
            return -math.copysign(A,self.x1)
        return math.copysign(A,self.x1)

    def _new_segment(self,t:float,first=False):
        old=self.x1 if not first else 0.0
        self.x0=float(old); self.x1=0.0 if first else self._new_target(); self.t0=t
        self.T=float(self.rng.uniform(self.cfg.segment_time_min,self.cfg.segment_time_max))
        mf=float(self.rng.uniform(self.cfg.move_fraction_min,self.cfg.move_fraction_max))
        self.Tmove=max(1e-3,self.T*mf)
        self.instant=(not first and self.rng.random()<self.cfg.instant_step_probability)
        if not first:self.change_count+=1

    def sample(self,t:float):
        if t-self.t0 >= self.T:
            self._new_segment(t)
        return self.preview(t)

    def preview(self,t:float):
        """Preview the announced segment; hold its target beyond the boundary.

        Future random targets are not announced yet and must not be revealed.
        """
        q=t-self.t0
        if self.instant or q>=self.Tmove:
            return self.x1,0.0,0.0
        z=clamp(q/self.Tmove,0.0,1.0)
        s=10*z**3-15*z**4+6*z**5
        sd=(30*z**2-60*z**3+30*z**4)/self.Tmove
        sdd=(60*z-180*z**2+120*z**3)/(self.Tmove*self.Tmove)
        dx=self.x1-self.x0
        return self.x0+dx*s, dx*sd, dx*sdd


class RigRLEnvV3:
    observation_dim=12
    action_dim=1

    def __init__(self, base_params:Optional[PlantParams]=None, controller_params:Optional[ControllerParams]=None,
                 cfg:Optional[RLEnvConfigV3]=None, seed:int=0):
        self.base_params=base_params or PlantParams()
        self.cp=controller_params or ControllerParams()
        self.cfg=cfg or RLEnvConfigV3()
        self.rng=np.random.default_rng(seed); self.seed_value=seed
        self.command_queue=CommandDelayQueue(self.cfg.command_delay,self.cfg.command_jitter,seed+15485863)
        self.p=self.base_params
        self.plant=RigPlant(self.p)
        self.step_dir_actuator=StepDirActuator(self.cfg.step_dir)
        self.last_actuator_diagnostics={}
        self.controller=Controller(self.plant,self.cp,self.cfg.physics_dt)
        self.y=np.zeros(7,dtype=float)
        self.t=0.0; self.steps=0; self.prev_action=0.0; self.previous_requested_action=0.0; self.prev_energy=0.0
        self.reference=SmoothRandomReference(self.rng,self.cfg,self.p.rail_half_travel)
        self.kick_at=-1.;self.kick_duration=0.;self.kick_torque=0.
        self.last_u=0.;self.last_base_accel=0.;self.last_residual=0.

    def _f(self,b):return float(self.rng.uniform(*b))

    def _randomized_params(self):
        p=replace(self.base_params)
        if not self.cfg.domain_randomization:return p
        p.resonator_mass*=self._f(self.cfg.mass_factor)
        p.k_theta*=self._f(self.cfg.stiffness_factor)
        p.c_theta*=self._f(self.cfg.damping_factor)
        p.belt_stiffness*=self._f(self.cfg.belt_stiffness_factor)
        p.b_x*=self._f(self.cfg.rail_damping_factor)
        p.motor_hold_torque*=self._f(self.cfg.motor_torque_factor)
        return p

    def _energy(self):
        k=max(abs(self.p.effective_small_angle_stiffness),0.02)
        J=max(self.p.resonator_inertia_pivot,1e-9)
        return float(.5*J*self.y[5]**2+.5*k*self.y[4]**2)

    def _energy_norm(self):
        c=self.cfg;k=max(abs(self.p.effective_small_angle_stiffness),0.02)
        J=max(self.p.resonator_inertia_pivot,1e-9)
        Eref=.5*J*c.theta_dot_scale**2+.5*k*c.theta_scale**2
        return self._energy()/max(Eref,1e-9)

    def _rail_barrier(self):
        p,c=self.p,self.cfg;L=max(p.rail_half_travel,1e-9)
        x=float(self.y[2]);v=float(self.y[3])
        sign=1.0 if x>=0 else -1.0
        out=max(0.0,sign*v)
        stop=out*out/(2*max(c.assumed_braking_accel,1e-3))
        pred=(abs(x)+stop+c.rail_velocity_buffer)/L
        z=max(0.,(pred-c.rail_soft_fraction)/max(c.rail_hard_fraction-c.rail_soft_fraction,1e-6))
        return min(z*z*(1+z*z),50.0),pred

    def reset(self,seed:Optional[int]=None):
        if seed is not None:self.rng=np.random.default_rng(seed);self.seed_value=seed
        self.p=self._randomized_params()
        self.plant=RigPlant(self.p)
        self.step_dir_actuator=StepDirActuator(self.cfg.step_dir)
        self.last_actuator_diagnostics={}
        self.controller=Controller(self.plant,self.cp,self.cfg.physics_dt)
        self.y=np.zeros(7,dtype=float)
        self.y[4]=float(self.rng.normal(0,self.cfg.initial_theta_std))
        self.y[5]=float(self.rng.normal(0,self.cfg.initial_theta_dot_std))
        self.t=0.;self.steps=0;self.prev_action=0.;self.previous_requested_action=0.
        self.last_u=0.;self.last_base_accel=0.;self.last_residual=0.;self.last_effective_residual=0.
        self.reference=SmoothRandomReference(self.rng,self.cfg,self.p.rail_half_travel)
        if self.rng.random()<self.cfg.kick_probability:
            self.kick_at=float(self.rng.uniform(.2,1.2))
            self.kick_duration=float(self.rng.uniform(self.cfg.kick_duration_min,self.cfg.kick_duration_max))
            self.kick_torque=self._f((self.cfg.kick_torque_min,self.cfg.kick_torque_max))*(1 if self.rng.random()<.5 else -1)
        else:
            self.kick_at=-1.;self.kick_duration=0.;self.kick_torque=0.
        self.command_queue=CommandDelayQueue(
            self.cfg.command_delay,self.cfg.command_jitter,self.seed_value+15485863)
        self.prev_energy=self._energy_norm()
        ref=self.reference.sample(0.)
        return self._observation(ref),self._info(ref,0.,{})

    def _base_accel(self,ref):
        a=self.controller.tracking_accel(self.y,ref)+self.controller.lqr_residual_accel(self.y,ref)
        return self.controller.project_accel(self.y,a)

    def _step_dir_interval(self, total_accel: float):
        """Track one control-period acceleration request through STEP/DIR drive limits."""
        max_carriage_speed = self.cfg.step_dir.max_velocity_rad_s * self.p.pulley_radius
        target_carriage_velocity = clamp(
            float(self.y[3]) + float(total_accel) * self.cfg.control_dt,
            -max_carriage_speed,
            max_carriage_speed,
        )
        requested_motor_velocity = target_carriage_velocity / max(self.p.pulley_radius, 1e-12)
        self.step_dir_actuator.command(requested_motor_velocity)
        torque_samples=[]
        pulse_count=0
        torque_saturated=False
        peak_rail_fraction=0.0
        last={}
        for _ in range(self.cfg.substeps):
            ext=self.kick_torque if self.kick_at>=0 and self.kick_at<=self.t<self.kick_at+self.kick_duration else 0.
            last=self.step_dir_actuator.advance(
                self.cfg.physics_dt, float(self.y[0]), float(self.y[1]),
                self.plant.motor_torque_limit(float(self.y[1])),
            )
            torque=last['torque_command_nm']
            torque_samples.append(float(torque))
            pulse_count+=int(last['step_pulses'])
            torque_saturated=torque_saturated or bool(last['torque_saturated'])
            self.y=self.plant.rk4(self.y,torque,self.cfg.physics_dt,external_torque=ext)
            self.t+=self.cfg.physics_dt
            self._after_physics_step()
            peak_rail_fraction=max(peak_rail_fraction,
                                   abs(float(self.y[2]))/max(self.p.rail_half_travel,1e-12))
        last.update({
            'actuator_mode':'step_dir',
            'commanded_carriage_velocity_m_s':target_carriage_velocity,
            'requested_motor_velocity_rad_s':requested_motor_velocity,
            'step_pulses_interval':pulse_count,
            'torque_saturated':torque_saturated,
            'interval_peak_rail_fraction':peak_rail_fraction,
        })
        self.last_actuator_diagnostics=last
        return float(np.mean(torque_samples)), last

    def _after_physics_step(self):
        """Hook for subclasses that sample sensors at physics ticks."""

    def _observation(self,ref):
        c,p=self.cfg,self.p;xr,vr,ar=ref
        L=max(p.rail_half_travel,1e-9);base=self._base_accel(ref)
        vals=np.array([
            (self.y[2]-xr)/c.pos_scale,(self.y[3]-vr)/c.vel_scale,
            self.y[4]/c.theta_scale,self.y[5]/c.theta_dot_scale,
            self.y[2]/L,self.y[3]/c.vel_scale,base/c.accel_scale,
            self.y[1]/c.motor_speed_scale,self.y[6]/max(p.motor_hold_torque,1e-6),
            ar/c.accel_scale,(self.prev_action if c.effective_action_observation else self.previous_requested_action),1.0-abs(self.y[2])/L
        ],dtype=np.float32)
        vals=np.clip(vals,-8,8)
        if c.observation_noise_std>0:
            vals+=self.rng.normal(0,c.observation_noise_std,size=vals.shape).astype(np.float32)
        return vals

    def _reward(self,a,ref,base,residual):
        c=self.cfg;xr,vr,_=ref
        ex=(self.y[2]-xr)/c.pos_scale;ev=(self.y[3]-vr)/c.vel_scale
        eth=self.y[4]/c.theta_scale;ew=self.y[5]/c.theta_dot_scale
        En=self._energy_norm()
        progress=c.w_energy_progress*(self.prev_energy-c.reward_gamma*En)
        rail,pred=self._rail_barrier();da=a-self.prev_action
        e=float(xr-self.y[2])
        adverse=max(0.0,-residual*e)/(max(c.residual_accel_limit*c.pos_scale,1e-9))
        costs={'pos':c.w_pos*ex*ex,'vel':c.w_vel*ev*ev,'theta':c.w_theta*eth*eth,
               'theta_dot':c.w_theta_dot*ew*ew,'energy':c.w_energy*En,
               'action':c.w_action*a*a,'action_delta':c.w_action_delta*da*da,
               'adverse':c.w_adverse_tracking*adverse,'rail':c.w_rail*rail}
        return float(c.alive_bonus-sum(costs.values())+progress),costs,pred,progress

    def _info(self,ref,reward,costs):
        rail,pred=self._rail_barrier()
        return {'t':self.t,'reward':reward,'x':float(self.y[2]),'x_ref':float(ref[0]),
            'theta':float(self.y[4]),'theta_dot':float(self.y[5]),
            'rail_fraction':float(abs(self.y[2])/max(self.p.rail_half_travel,1e-9)),
            'predicted_rail_fraction':float(pred),'energy':self._energy(),
            'base_accel':self.last_base_accel,'residual_accel':self.last_residual,
            'tau_cmd':self.last_u,**{f'cost_{k}':float(v) for k,v in costs.items()}}

    def step(self,action:Sequence[float]|np.ndarray|float):
        a=clamp(float(np.asarray(action).reshape(-1)[0]),-1.,1.)
        ref=self.reference.sample(self.t)
        base=self._base_accel(ref)
        residual=a*self.cfg.residual_accel_limit
        total=self.controller.project_accel(self.y,base+residual)
        timing=self.command_queue.issue(total,self.t)
        applied_accel=self.controller.project_accel(self.y,timing['command_applied'])
        command_saturated=not math.isclose(applied_accel,timing['command_applied'],
                                           rel_tol=0.0,abs_tol=1e-12)
        self.last_base_accel=base;self.last_residual=residual
        if self.cfg.actuator_mode == 'torque':
            u=self.controller.torque_from_accel(self.y,applied_accel)
            for _ in range(self.cfg.substeps):
                ext=self.kick_torque if self.kick_at>=0 and self.kick_at<=self.t<self.kick_at+self.kick_duration else 0.
                self.y=self.plant.rk4(self.y,u,self.cfg.physics_dt,external_torque=ext)
                self.t+=self.cfg.physics_dt
                self._after_physics_step()
            self.last_actuator_diagnostics={'actuator_mode':'torque'}
        else:
            u,_=self._step_dir_interval(applied_accel)
        self.last_effective_residual=applied_accel-base
        self.last_effective_action=self.last_effective_residual/max(self.cfg.residual_accel_limit,1e-9)
        self.last_u=u
        self.steps+=1
        ref=self.reference.sample(self.t)
        effective_action=self.last_effective_action
        reward,costs,pred,progress=self._reward(effective_action,ref,base,self.last_effective_residual)
        self.prev_energy=self._energy_norm();self.prev_action=effective_action;self.previous_requested_action=a
        rail=abs(self.y[2])/max(self.p.rail_half_travel,1e-9)
        term=bool(rail>self.cfg.terminate_rail_fraction or abs(self.y[4])>self.cfg.terminate_theta or not np.all(np.isfinite(self.y)))
        trunc=self.steps>=self.cfg.max_steps
        if term:reward-=50.
        obs=self._observation(ref)
        info=self._info(ref,reward,costs)
        info['energy_progress_reward']=progress
        info['requested_action']=a
        info['effective_residual_accel']=self.last_effective_residual
        info.update(timing)
        info['command_applied']=applied_accel
        info['command_saturated']=command_saturated
        info.update(self.last_actuator_diagnostics)
        return obs,reward,term,trunc,info

    def config_dict(self):return asdict(self.cfg)


class VectorRigEnvV3:
    def __init__(self,n_envs:int=8,seed:int=0,cfg:Optional[RLEnvConfigV3]=None):
        self.envs=[RigRLEnvV3(cfg=cfg,seed=seed+1009*i) for i in range(n_envs)]
        self.n_envs=n_envs;self.obs_dim=12;self.action_dim=1

    def reset(self):return np.stack([e.reset()[0] for e in self.envs])

    def step(self,actions):
        O=[];R=[];D=[];I=[]
        for i,e in enumerate(self.envs):
            o,r,te,tr,info=e.step(actions[i]);d=te or tr
            info=dict(info); info['terminated']=te; info['truncated']=tr and not te
            if d:
                info['final_observation']=o.copy();info['episode_done']=1.;o,_=e.reset()
            O.append(o);R.append(r);D.append(d);I.append(info)
        return np.stack(O),np.asarray(R,np.float32),np.asarray(D,np.float32),I


if __name__=='__main__':
    e=RigRLEnvV3(seed=1);o,i=e.reset();R=0
    for _ in range(600):
        o,r,te,tr,i=e.step([0.]);R+=r
    print('smoke',R,math.degrees(i['theta']),i['rail_fraction'])

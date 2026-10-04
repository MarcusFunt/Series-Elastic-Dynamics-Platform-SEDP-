#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from active_vibration_rig_2d import PlantParams,ControllerParams,clamp
from ppo_agent_v2 import load_checkpoint_v2
from rig_rl_env_v3 import RLEnvConfigV3

class PPOPolicyAdapterV3:
    """Runtime adapter: policy output is a residual carriage acceleration."""
    def __init__(self,checkpoint:Path|str,plant_params:PlantParams,controller_params:ControllerParams,physics_dt:float,device='cpu',command_dt=None):
        self.model,self.ckpt=load_checkpoint_v2(Path(checkpoint),device);self.device=torch.device(device);self.p=plant_params;self.cp=controller_params
        cfgd=dict(self.ckpt.get('env_config',{}));cfgd.setdefault('effective_action_observation',False);allowed=set(RLEnvConfigV3.__dataclass_fields__);cfgd={k:v for k,v in cfgd.items() if k in allowed};self.cfg=RLEnvConfigV3(**cfgd)
        if self.model.obs_dim != 12 or self.model.action_dim != 1:
            raise ValueError('Expected a v3 checkpoint with 12 observations and one action')
        self.physics_dt=physics_dt;self.control_steps=max(1,int(round(self.cfg.control_dt/(command_dt or physics_dt))));self.counter=0;self.last_action=0.;self.previous_effective_action=0.
    def reset(self):self.counter=0;self.last_action=0.;self.previous_effective_action=0.
    def _obs(self,y,ref,base):
        c,p=self.cfg,self.p;xr,vr,ar=ref;L=max(p.rail_half_travel,1e-9)
        return np.clip(np.asarray([(y[2]-xr)/c.pos_scale,(y[3]-vr)/c.vel_scale,y[4]/c.theta_scale,y[5]/c.theta_dot_scale,y[2]/L,y[3]/c.vel_scale,base/c.accel_scale,y[1]/c.motor_speed_scale,y[6]/max(p.motor_hold_torque,1e-6),ar/c.accel_scale,(self.previous_effective_action if c.effective_action_observation else self.last_action),1.-abs(y[2])/L],np.float32),-8,8)
    def record_effective_action(self, residual):
        self.previous_effective_action=float(residual/self.cfg.residual_accel_limit)

    @torch.no_grad()
    def residual_accel(self,y,ref:Tuple[float,float,float],base_accel:float)->float:
        if self.counter%self.control_steps==0:
            o=torch.tensor(self._obs(y,ref,base_accel),dtype=torch.float32,device=self.device).unsqueeze(0);self.last_action=clamp(float(self.model.deterministic(o)[0,0].cpu()),-1,1)
        self.counter+=1;return self.last_action*self.cfg.residual_accel_limit

#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
from typing import Tuple
import numpy as np, torch
from active_vibration_rig_2d import PlantParams,ControllerParams,clamp
from rig_rl_env_v2 import RLEnvConfigV2
from ppo_agent_v2 import load_checkpoint_v2

class PPOPolicyAdapterV2:
 def __init__(self,checkpoint,plant_params:PlantParams,controller_params:ControllerParams,physics_dt:float,device='cpu'):
  self.model,self.ckpt=load_checkpoint_v2(Path(checkpoint),device); d=dict(self.ckpt.get('env_config',{})); allowed={f.name for f in fields(RLEnvConfigV2)};d={k:v for k,v in d.items() if k in allowed};self.cfg=RLEnvConfigV2(**d);self.p=plant_params;self.cp=controller_params;self.physics_dt=physics_dt;self.control_steps=max(1,int(round(self.cfg.control_dt/physics_dt)));self.counter=0;self.prev_action=0.;self.last_torque=0.;self.device=torch.device(device)
 def reset(self):self.counter=0;self.prev_action=0.;self.last_torque=0.
 def _servo(self,y,ref):
  xr,vr,ar=ref;p,cp=self.p,self.cp;F=cp.x_kp*(xr-y[2])+cp.x_kd*(vr-y[3])+(p.carriage_mass+p.resonator_mass)*ar;return p.pulley_radius*F
 def _obs(self,y,ref):
  p,c=self.p,self.cfg;xr,vr,ar=ref;rp=max(p.pulley_radius,1e-6);L=max(p.rail_half_travel,1e-6);return np.clip(np.asarray([rp*y[0]/L,rp*y[1]/c.max_reference_speed,y[2]/L,y[3]/c.max_reference_speed,y[4]/c.theta_scale,y[5]/c.theta_dot_scale,y[6]/max(p.motor_hold_torque,1e-6),xr/L,vr/c.max_reference_speed,ar/c.max_reference_accel,self.prev_action],np.float32),-8,8)
 def _safety_scale(self,y,residual):
  p,c=self.p,self.cfg;L=max(p.rail_half_travel,1e-9);x=float(y[2]);v=float(y[3]);frac=abs(x)/L
  if frac<=c.rail_soft_fraction:return 1.
  span=max(c.rail_hard_fraction-c.rail_soft_fraction,1e-6);scale=float(np.clip((c.rail_hard_fraction-frac)/span,.08,1.));out=(x*residual>0) or (x*v>0 and v*residual>0);return scale if out else min(1.,.35+.65*scale)
 @torch.no_grad()
 def command(self,y,ref:Tuple[float,float,float]):
  if self.counter%self.control_steps==0:
   o=torch.tensor(self._obs(y,ref),dtype=torch.float32,device=self.device).unsqueeze(0);a=float(self.model.deterministic(o)[0,0].cpu());a=clamp(a,-1,1);self.prev_action=a
   if self.cfg.action_mode=='residual':
    r=a*self.cfg.residual_torque_limit;r*=self._safety_scale(y,r);self.last_torque=self._servo(y,ref)+r
   else:self.last_torque=a*self.cfg.direct_torque_limit
  self.counter+=1;return float(self.last_torque)

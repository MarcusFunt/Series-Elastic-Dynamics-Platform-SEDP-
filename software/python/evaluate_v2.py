#!/usr/bin/env python3
import math,csv
from pathlib import Path
import numpy as np
from active_vibration_rig_2d import PlantParams,ControllerParams,MotionParams,Simulator
from rig_rl_policy_v2 import PPOPolicyAdapterV2

def run(mode,traj,dur,model=None):
 s=Simulator(PlantParams(),ControllerParams(),MotionParams(),dt=.0005);s.trajectory.mode=traj
 if mode.startswith('ppo'):
  s.controller.set_rl_policy(PPOPolicyAdapterV2(model,s.p,s.cp,s.dt));s.controller.mode='ppo'
 else:s.controller.mode=mode
 s.step(int(dur/s.dt));return s

def met(s,traj):
 h=s.history;t=np.array([r['t'] for r in h]);th=np.array([r['theta'] for r in h]);w=np.array([r['theta_dot'] for r in h]);x=np.array([r['x'] for r in h]);tau=np.array([r['tau_cmd'] for r in h]);xr=np.array([s.trajectory.sample(float(tt))[0] for tt in t]);p=s.p;k=max(abs(p.effective_small_angle_stiffness),.02);E=.5*p.resonator_inertia_pivot*w*w+.5*k*th*th
 mask=t>=(s.mp.step_time+.3) if traj=='step' else np.ones_like(t,dtype=bool)
 return {'peak_deg':float(np.max(np.abs(np.degrees(th)))),'rms_deg':float(np.sqrt(np.mean(np.degrees(th)**2))),'resid_deg':float(np.sqrt(np.mean(np.degrees(th[mask])**2))),'pos_rmse_mm':float(1000*np.sqrt(np.mean((x-xr)**2))),'energy_mJs':float(1000*np.trapezoid(E,t)),'tau_rms':float(np.sqrt(np.mean(tau*tau))),'rail':float(np.max(abs(x))/p.rail_half_travel)}

models={'anchor':Path('models/ppo_v2_anchor.pt'),'settle':Path('runs/ppo_v2_fast/policy_settle.pt'),'reversal':Path('models/ppo_v2_reversal.pt'),'final':Path('models/ppo_v2_final.pt')}
rows=[]
for tr,dur in [('step',5.),('aggressive',6.)]:
 for name in ['servo','lqr']:
  m=met(run(name,tr,dur),tr);rows.append({'traj':tr,'controller':name,**m});print(tr,name,m)
 for name,p in models.items():
  if not p.exists(): continue
  m=met(run('ppo',tr,dur,p),tr);rows.append({'traj':tr,'controller':'ppo_'+name,**m});print(tr,'ppo_'+name,m)
if rows:
 Path('runs').mkdir(exist_ok=True)
 with open('runs/ppo_v2_eval.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

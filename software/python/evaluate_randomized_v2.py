#!/usr/bin/env python3
from pathlib import Path
import numpy as np, torch, math
from rig_rl_env_v2 import RLEnvConfigV2,RigRLEnvV2
from ppo_agent_v2 import load_checkpoint_v2

def eval_model(path,seeds):
 model,_=load_checkpoint_v2(Path(path)); cfg=RLEnvConfigV2(physics_dt=.001,residual_torque_limit=.45,domain_randomization=True,max_target_fraction=.64,min_target_hold=.55,max_target_hold=.95,reversal_probability=.82,kick_probability=.30,observation_noise_std=.0,w_energy_progress=3.2,w_rail=8.5)
 rows=[]
 for sd in seeds:
  env=RigRLEnvV2(cfg=cfg,seed=sd); obs,_=env.reset(seed=sd); R=0;ths=[];rail=[];pos=[]
  for _ in range(cfg.max_steps):
   with torch.no_grad(): a=float(model.deterministic(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))[0,0])
   obs,r,te,tr,info=env.step([a]);R+=r;ths.append(abs(math.degrees(info['theta'])));rail.append(info['rail_fraction']);pos.append(abs(info['x']-info['x_ref'])*1000)
   if te or tr: break
  rows.append((R,float(np.mean(ths)),float(np.max(ths)),float(np.max(rail)),float(np.mean(pos)),float(te)))
 return np.array(rows)

if __name__=='__main__':
 seeds=list(range(100,125))
 for name,path in [('base','models/ppo_v2_base.pt'),('trained','models/ppo_v2_reversal2.pt')]:
  if not Path(path).exists(): continue
  a=eval_model(path,seeds);print(name,'return',a[:,0].mean(),'mean_abs_theta',a[:,1].mean(),'peak_theta',a[:,2].mean(),'railmax',a[:,3].mean(),'poserr',a[:,4].mean(),'terminated',a[:,5].mean())

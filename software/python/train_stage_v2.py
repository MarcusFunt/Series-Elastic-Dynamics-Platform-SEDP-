#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from rig_rl_env_v2 import RLEnvConfigV2,VectorRigEnvV2
from ppo_agent_v2 import PPOConfigV2,RolloutBufferV2,load_checkpoint_v2,ppo_update_v2,save_checkpoint_v2

def cfg_for(name):
 if name=='reversal': return RLEnvConfigV2(physics_dt=.001,residual_torque_limit=.45,domain_randomization=True,max_target_fraction=.64,min_target_hold=.55,max_target_hold=.95,reversal_probability=.82,kick_probability=.30,observation_noise_std=.0012,w_energy_progress=3.2,w_rail=8.5)
 if name=='robust': return RLEnvConfigV2(physics_dt=.001,residual_torque_limit=.45,domain_randomization=True,max_target_fraction=.70,min_target_hold=.38,max_target_hold=.78,reversal_probability=.86,kick_probability=.50,observation_noise_std=.002,
  mass_factor=(.78,1.28),stiffness_factor=(.72,1.32),damping_factor=(.50,1.65),belt_stiffness_factor=(.68,1.38),rail_damping_factor=(.72,1.35),motor_torque_factor=(.86,1.12),w_energy_progress=3.0,w_rail=9.5)
 raise ValueError(name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--init',type=Path,required=True);ap.add_argument('--anchor',type=Path,default=Path('models/ppo_v2_anchor.pt'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['reversal','robust'],required=True);ap.add_argument('--steps',type=int,default=40000);ap.add_argument('--seed',type=int,default=47);args=ap.parse_args()
 dev=torch.device('cpu');model,_=load_checkpoint_v2(args.init,dev);anchor,_=load_checkpoint_v2(args.anchor,dev);anchor.eval();[p.requires_grad_(False) for p in anchor.parameters()]
 cfg=cfg_for(args.stage); pc=PPOConfigV2(n_envs=8,n_steps=256,learning_rate=1.2e-5 if args.stage=='robust' else 1.5e-5,epochs=3,minibatch_size=512,clip_ratio=.10,anchor_kl_coef=.020 if args.stage=='robust' else .028,symmetry_coef=.05,target_kl=.015,entropy_coef=.001)
 opt=torch.optim.Adam(model.parameters(),lr=pc.learning_rate,eps=1e-5); env=VectorRigEnvV2(8,args.seed,cfg);obs=torch.tensor(env.reset(),dtype=torch.float32);roll=pc.n_envs*pc.n_steps;nup=math.ceil(args.steps/roll);gs=0;t0=time.time()
 for u in range(nup):
  b=RolloutBufferV2(pc.n_steps,pc.n_envs,11,1,dev);ra=th=rp=0
  for t in range(pc.n_steps):
   with torch.no_grad():a,lp,_,v=model.sample(obs)
   no,r,d,infos=env.step(a.numpy());b.obs[t]=obs;b.actions[t]=a;b.logp[t]=lp;b.values[t]=v;b.rewards[t]=torch.tensor(r);b.dones[t]=torch.tensor(d);obs=torch.tensor(no,dtype=torch.float32);gs+=8;ra+=float(r.mean());th+=sum(abs(np.degrees(i['theta'])) for i in infos);rp+=sum(i['predicted_rail_fraction'] for i in infos)
  with torch.no_grad():lv=model.value(obs)
  b.compute_gae(lv,pc.gamma,pc.gae_lambda);ls=ppo_update_v2(model,opt,b,pc,anchor)
  if u%4==0 or u==nup-1:print(args.stage,gs,'r',ra/pc.n_steps,'th',th/(pc.n_steps*8),'rail',rp/(pc.n_steps*8),'anchor',ls['anchor_kl'])
 save_checkpoint_v2(args.out,model,pc,asdict(cfg),{'stage':args.stage,'steps':gs,'wall_s':time.time()-t0});print('saved',args.out)
if __name__=='__main__':main()

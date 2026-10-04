#!/usr/bin/env python3
"""Warm-start PPO v3 from a physically motivated bounded damping residual.

Teacher:
    a_res ~= 0.5*theta_dot + 1.0*theta
clipped to the v3 residual-acceleration budget.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from ppo_agent_v2 import ActorCriticV2,PPOConfigV2,save_checkpoint_v2
from rig_rl_env_v3 import RLEnvConfigV3,RigRLEnvV3

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=24000);ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--out',type=Path,default=Path('../../models/ppo_v3/heuristic_warmstart.pt'));ap.add_argument('--seed',type=int,default=71);args=ap.parse_args()
 np.random.seed(args.seed);torch.manual_seed(args.seed);cfg=RLEnvConfigV3(observation_noise_std=0.0);env=RigRLEnvV3(cfg=cfg,seed=args.seed);X=[];Y=[]
 while len(X)<args.samples:
  obs,_=env.reset()
  for _ in range(cfg.max_steps):
   target=np.clip((0.50*float(env.y[5])+1.00*float(env.y[4]))/cfg.residual_accel_limit,-1.,1.)
   X.append(obs.copy());Y.append([target]);obs,_,te,tr,_=env.step([target])
   if te or tr or len(X)>=args.samples:break
 X=torch.tensor(np.asarray(X,np.float32));Y=torch.tensor(np.asarray(Y,np.float32));m=ActorCriticV2(12,1,128);opt=torch.optim.Adam(m.parameters(),lr=5e-4)
 n=len(X)
 for ep in range(args.epochs):
  perm=torch.randperm(n);losses=[]
  for st in range(0,n,512):
   ix=perm[st:st+512];pred=m.deterministic(X[ix]);predm=m.deterministic(-X[ix]);loss=((pred-Y[ix])**2).mean()+0.01*((pred+predm)**2).mean();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
  with torch.no_grad():mae=float((m.deterministic(X)-Y).abs().mean())
  print(ep+1,float(np.mean(losses)),mae)
 pc=PPOConfigV2(hidden_size=128);save_checkpoint_v2(args.out,m,pc,asdict(cfg),{'v3':True,'pretrain':'physical damping residual','samples':len(X)});print('saved',args.out)
if __name__=='__main__':main()

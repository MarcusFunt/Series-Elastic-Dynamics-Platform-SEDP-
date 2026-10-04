#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from active_vibration_rig_2d import Controller,ControllerParams
from rig_rl_env_v2 import RLEnvConfigV2,RigRLEnvV2
from ppo_agent_v2 import PPOConfigV2,ActorCriticV2,load_v1_into_v2,save_checkpoint_v2

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--init',type=Path,default=Path('models/ppo_lqr_warmstart.pt')); ap.add_argument('--out',type=Path,default=Path('models/ppo_v2_anchor.pt')); ap.add_argument('--samples',type=int,default=30000); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--seed',type=int,default=23); args=ap.parse_args()
 np.random.seed(args.seed); torch.manual_seed(args.seed)
 cfg=RLEnvConfigV2(domain_randomization=True,observation_noise_std=0.0,kick_probability=0.25,min_target_hold=.65,max_target_hold=1.15,max_target_fraction=.60)
 env=RigRLEnvV2(cfg=cfg,seed=args.seed); X=[];Y=[]
 while len(X)<args.samples:
  obs,_=env.reset(); expert=Controller(env.plant,ControllerParams(),cfg.physics_dt); expert.mode='lqr'
  if expert.K is None: continue
  for _ in range(cfg.max_steps):
   ref,_=env.reference.sample(env.t); obs=env._observation(ref); us=env._servo_torque(ref); ul=expert.command(env.y,ref); lim=env.plant.motor_torque_limit(float(env.y[1])); ul=float(np.clip(ul,-lim,lim)); a=float(np.clip((ul-us)/cfg.residual_torque_limit,-1,1)); X.append(obs.copy());Y.append([a]); _,_,te,tr,_=env.step([a])
   if te or tr or len(X)>=args.samples: break
 print('dataset',len(X))
 if args.init.exists(): model,_=load_v1_into_v2(args.init); print('init',args.init)
 else: model=ActorCriticV2(11,1,128)
 model.train(); Xt=torch.tensor(np.asarray(X,np.float32)); Yt=torch.tensor(np.asarray(Y,np.float32)); opt=torch.optim.Adam(model.parameters(),lr=5e-4); n=len(Xt)
 for ep in range(args.epochs):
  perm=torch.randperm(n); ls=[]
  for s in range(0,n,512):
   ix=perm[s:s+512]; pred=model.deterministic(Xt[ix]); predm=model.deterministic(-Xt[ix]); loss=((pred-Yt[ix])**2).mean()+0.05*((pred+predm)**2).mean(); opt.zero_grad();loss.backward();opt.step();ls.append(float(loss.detach()))
  with torch.no_grad(): mae=float((model.deterministic(Xt)-Yt).abs().mean())
  print(f'ep {ep+1}/{args.epochs} mse={np.mean(ls):.5f} mae={mae:.4f}')
 pc=PPOConfigV2(); save_checkpoint_v2(args.out,model,pc,asdict(cfg),{'pretraining':'randomized LQR imitation','samples':len(X),'epochs':args.epochs}); print('saved',args.out)
if __name__=='__main__': main()

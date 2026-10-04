#!/usr/bin/env python3
"""Train v3 residual PPO and only preserve checkpoints that pass SEDP-B1 gates."""
from __future__ import annotations
import argparse,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch
from ppo_agent_v2 import ActorCriticV2,PPOConfigV2,RolloutBufferV2,ppo_update_v2,save_checkpoint_v2,load_checkpoint_v2
from rig_rl_env_v3 import RLEnvConfigV3,VectorRigEnvV3
from benchmark_suite import benchmark,summarize

class InMemoryPolicy:
    def __init__(self,model,cfg,device='cpu'):
        self.model=model;self.cfg=cfg;self.device=torch.device(device);self.counter=0;self.last_action=0.;self.control_steps=max(1,int(round(cfg.control_dt/.0005)))
    def reset(self):self.counter=0;self.last_action=0.
    def _obs(self,y,ref,base,p):
        c=self.cfg;xr,vr,ar=ref;L=max(p.rail_half_travel,1e-9)
        return np.clip(np.asarray([(y[2]-xr)/c.pos_scale,(y[3]-vr)/c.vel_scale,y[4]/c.theta_scale,y[5]/c.theta_dot_scale,y[2]/L,y[3]/c.vel_scale,base/c.accel_scale,y[1]/c.motor_speed_scale,y[6]/max(p.motor_hold_torque,1e-6),ar/c.accel_scale,self.last_action,1.-abs(y[2])/L],np.float32),-8,8)
    @torch.no_grad()
    def residual_accel(self,y,ref,base):
        if self.counter%self.control_steps==0:
            o=torch.tensor(self._obs(y,ref,base,self.p),dtype=torch.float32,device=self.device).unsqueeze(0);self.last_action=float(self.model.deterministic(o)[0,0])
        self.counter+=1;return float(np.clip(self.last_action,-1,1)*self.cfg.residual_accel_limit)

class BoundPolicy(InMemoryPolicy):
    def bind(self,p):self.p=p;return self

def score_rows(rows):
    rr=[r for r in rows if r.controller=='ppo']
    if not rr or not all(r.accepted for r in rr):return float('inf')
    return float(sum(r.angle_rms_deg + .06*r.position_rmse_mm + 2*max(0,r.peak_rail_fraction-.70) for r in rr))

def evaluate_model(model,cfg):
    from active_vibration_rig_2d import PlantParams
    pol=BoundPolicy(model,cfg);pol.p=PlantParams()
    rows=benchmark(['ppo'],ppo_policy=pol)
    return rows,score_rows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=120000);ap.add_argument('--outdir',type=Path,default=Path('../../models/ppo_v3'));ap.add_argument('--seed',type=int,default=73);ap.add_argument('--eval-every',type=int,default=16384);ap.add_argument('--init',type=Path);args=ap.parse_args()
    np.random.seed(args.seed);torch.manual_seed(args.seed);device=torch.device('cpu');cfg=RLEnvConfigV3();pc=PPOConfigV2(hidden_size=128,learning_rate=1.2e-5,gamma=.995,gae_lambda=.95,clip_ratio=.10,value_coef=.55,entropy_coef=.0005,max_grad_norm=.5,epochs=3,minibatch_size=512,n_steps=256,n_envs=8,anchor_kl_coef=.020,symmetry_coef=.025,target_kl=.015,seed=args.seed)
    if args.init:
        model,_=load_checkpoint_v2(args.init,device)
        if model.obs_dim != 12: raise SystemExit(f'v3 requires obs_dim=12, got {model.obs_dim}')
        model.train()
    else:
        model=ActorCriticV2(12,1,pc.hidden_size).to(device)
    anchor=ActorCriticV2(12,1,pc.hidden_size).to(device);anchor.load_state_dict(model.state_dict());anchor.eval();[p.requires_grad_(False) for p in anchor.parameters()]
    opt=torch.optim.Adam(model.parameters(),lr=pc.learning_rate,eps=1e-5);env=VectorRigEnvV3(pc.n_envs,args.seed,cfg);obs=torch.tensor(env.reset(),dtype=torch.float32,device=device);args.outdir.mkdir(parents=True,exist_ok=True)
    rows,best_score=evaluate_model(model,cfg);best_step=0
    save_checkpoint_v2(args.outdir/'policy_best.pt',model,pc,asdict(cfg),{'v3':True,'steps':0,'benchmark_score':best_score});print('initial score',best_score,'accepted',summarize(rows))
    roll=pc.n_envs*pc.n_steps;updates=math.ceil(args.steps/roll);steps=0;next_eval=args.eval_every;t0=time.time();sc=best_score
    for u in range(updates):
        b=RolloutBufferV2(pc.n_steps,pc.n_envs,12,1,device);ra=th=rail=0.
        for t in range(pc.n_steps):
            with torch.no_grad():a,lp,_,v=model.sample(obs)
            no,r,d,infos=env.step(a.cpu().numpy());b.obs[t]=obs;b.actions[t]=a;b.logp[t]=lp;b.values[t]=v;b.rewards[t]=torch.tensor(r,device=device);b.dones[t]=torch.tensor(d,device=device);obs=torch.tensor(no,dtype=torch.float32,device=device);steps+=pc.n_envs;ra+=float(np.mean(r));th+=sum(abs(np.degrees(i['theta'])) for i in infos);rail+=sum(i['rail_fraction'] for i in infos)
        with torch.no_grad():lv=model.value(obs)
        b.compute_gae(lv,pc.gamma,pc.gae_lambda);loss=ppo_update_v2(model,opt,b,pc,anchor)
        if u%4==0:print('step',steps,'reward',ra/pc.n_steps,'theta',th/(pc.n_steps*pc.n_envs),'rail',rail/(pc.n_steps*pc.n_envs),'kl',loss['anchor_kl'])
        if steps>=next_eval or u==updates-1:
            rows,sc=evaluate_model(model,cfg);accepted=summarize(rows).get('ppo',False);print('BENCH',steps,'score',sc,'accepted',accepted)
            save_checkpoint_v2(args.outdir/f'policy_{steps:08d}.pt',model,pc,asdict(cfg),{'v3':True,'steps':steps,'benchmark_score':sc,'accepted':accepted})
            if accepted and sc<best_score:
                best_score=sc;best_step=steps;save_checkpoint_v2(args.outdir/'policy_best.pt',model,pc,asdict(cfg),{'v3':True,'steps':steps,'benchmark_score':sc,'accepted':True})
            next_eval+=args.eval_every
    save_checkpoint_v2(args.outdir/'policy_final.pt',model,pc,asdict(cfg),{'v3':True,'steps':steps,'benchmark_score':sc,'wall_s':time.time()-t0});print('best',best_step,best_score,'final',steps,sc)
if __name__=='__main__':main()

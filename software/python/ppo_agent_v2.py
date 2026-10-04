#!/usr/bin/env python3
"""PPO v2: bounded Beta actor, anchor-KL regularization and symmetry loss."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Dict,Tuple,Optional
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

@dataclass
class PPOConfigV2:
    hidden_size:int=128; learning_rate:float=3e-5; gamma:float=0.995; gae_lambda:float=0.95
    clip_ratio:float=0.10; value_coef:float=0.55; entropy_coef:float=0.001
    max_grad_norm:float=0.5; epochs:int=3; minibatch_size:int=512; n_steps:int=256; n_envs:int=8; seed:int=7
    anchor_kl_coef:float=0.025; symmetry_coef:float=0.040; target_kl:float=0.020

class ActorCriticV2(nn.Module):
    def __init__(self,obs_dim:int,action_dim:int,hidden:int=128):
        super().__init__(); self.obs_dim=obs_dim; self.action_dim=action_dim
        self.trunk=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actor=nn.Linear(hidden,2*action_dim); self.critic=nn.Linear(hidden,1); self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,gain=np.sqrt(2.)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,gain=.01); nn.init.orthogonal_(self.critic.weight,gain=1.)
    def _dist(self,obs):
        z=self.trunk(obs); a,b=torch.chunk(self.actor(z),2,-1); return Beta(F.softplus(a)+1.05,F.softplus(b)+1.05)
    def value(self,obs): return self.critic(self.trunk(obs)).squeeze(-1)
    def sample(self,obs):
        d=self._dist(obs); z=d.rsample(); a=2*z-1; lp=d.log_prob(z).sum(-1)-self.action_dim*np.log(2.); ent=d.entropy().sum(-1)+self.action_dim*np.log(2.); return a,lp,ent,self.value(obs)
    def evaluate_actions(self,obs,action):
        d=self._dist(obs); z=torch.clamp((action+1)*.5,1e-6,1-1e-6); lp=d.log_prob(z).sum(-1)-self.action_dim*np.log(2.); ent=d.entropy().sum(-1)+self.action_dim*np.log(2.); return lp,ent,self.value(obs)
    def deterministic(self,obs):
        d=self._dist(obs); return 2*d.mean-1

class ActorCriticV4(ActorCriticV2):
    """Independent actor/critic features and controlled initial exploration."""
    def __init__(self,obs_dim,action_dim,hidden=128):
        super().__init__(obs_dim,action_dim,hidden)
        self.critic_trunk=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        for layer in self.critic_trunk.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,gain=np.sqrt(2.));nn.init.zeros_(layer.bias)
        with torch.no_grad():self.actor.bias.fill_(np.log(np.expm1(20.-1.05)))

    def value(self,obs):return self.critic(self.critic_trunk(obs)).squeeze(-1)


def reflect_observation(obs, signs=None):
    """Reflect signed coordinates, preserving margins and confidence features."""
    if signs is None:
        if obs.shape[-1] not in (11, 12):
            raise ValueError("Provide a reflection map for this observation schema")
        signs = [-1.] * obs.shape[-1]
        if obs.shape[-1] == 12:
            signs[11] = 1.
    return obs * torch.as_tensor(signs, dtype=obs.dtype, device=obs.device)


class RolloutBufferV2:
    def __init__(self,n_steps,n_envs,obs_dim,action_dim,device):
        self.n_steps=n_steps; self.n_envs=n_envs; self.device=device
        self.obs=torch.zeros((n_steps,n_envs,obs_dim),device=device); self.actions=torch.zeros((n_steps,n_envs,action_dim),device=device)
        self.logp=torch.zeros((n_steps,n_envs),device=device); self.rewards=torch.zeros((n_steps,n_envs),device=device); self.dones=torch.zeros((n_steps,n_envs),device=device); self.values=torch.zeros((n_steps,n_envs),device=device)
        self.truncated=torch.zeros_like(self.rewards); self.final_values=torch.zeros_like(self.rewards)
        self.advantages=torch.zeros_like(self.rewards); self.returns=torch.zeros_like(self.rewards)
    def compute_gae(self,last_value,gamma,lam):
        gae=torch.zeros(self.n_envs,device=self.device)
        for t in reversed(range(self.n_steps)):
            nv=last_value if t==self.n_steps-1 else self.values[t+1]; nt=1-self.dones[t]
            bootstrap=nv*nt+self.final_values[t]*self.truncated[t]
            delta=self.rewards[t]+gamma*bootstrap-self.values[t]; gae=delta+gamma*lam*nt*gae; self.advantages[t]=gae
        self.returns=self.advantages+self.values
    def flattened(self):
        b=self.n_steps*self.n_envs; return self.obs.reshape(b,-1),self.actions.reshape(b,-1),self.logp.reshape(b),self.advantages.reshape(b),self.returns.reshape(b)

def ppo_update_v2(model,optimizer,buffer,cfg: PPOConfigV2,anchor:Optional[ActorCriticV2]=None, reflection_signs=None):
    obs,actions,oldlp,adv,ret=buffer.flattened(); adv=(adv-adv.mean())/(adv.std()+1e-8); n=len(obs)
    L={k:0. for k in ['policy','value','entropy','kl','clipfrac','anchor_kl','symmetry']}; count=0
    stop=False
    for _ in range(cfg.epochs):
        idx=torch.randperm(n,device=obs.device)
        for st in range(0,n,cfg.minibatch_size):
            mb=idx[st:st+cfg.minibatch_size]; lp,ent,val=model.evaluate_actions(obs[mb],actions[mb]); ratio=torch.exp(lp-oldlp[mb])
            pl=-torch.min(ratio*adv[mb],torch.clamp(ratio,1-cfg.clip_ratio,1+cfg.clip_ratio)*adv[mb]).mean(); vl=.5*(ret[mb]-val).pow(2).mean(); em=ent.mean()
            anchor_kl=torch.tensor(0.,device=obs.device)
            if anchor is not None and cfg.anchor_kl_coef>0:
                with torch.no_grad(): ad=anchor._dist(obs[mb])
                cd=model._dist(obs[mb]); anchor_kl=kl_divergence(ad,cd).sum(-1).mean()
            mu=model.deterministic(obs[mb]); mu_m=model.deterministic(reflect_observation(obs[mb], reflection_signs)); sym=(mu+mu_m).pow(2).mean()
            loss=pl+cfg.value_coef*vl-cfg.entropy_coef*em+cfg.anchor_kl_coef*anchor_kl+cfg.symmetry_coef*sym
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if isinstance(model,ActorCriticV4):
                nn.utils.clip_grad_norm_(list(model.trunk.parameters())+list(model.actor.parameters()),cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(list(model.critic_trunk.parameters())+list(model.critic.parameters()),cfg.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad(): logratio=lp-oldlp[mb]; ak=(torch.exp(logratio)-1-logratio).mean(); cf=((ratio-1).abs()>cfg.clip_ratio).float().mean()
            vals=[pl,vl,em,ak,cf,anchor_kl,sym]
            for k,v in zip(L,vals): L[k]+=float(v.detach())
            count+=1
            if float(ak)>1.5*cfg.target_kl: stop=True; break
        if stop: break
    for k in L: L[k]/=max(1,count)
    return L

def load_v1_into_v2(path:Path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False); cls=ActorCriticV4 if 'critic_trunk.0.weight' in ck['model_state'] else ActorCriticV2; m=cls(int(ck['obs_dim']),int(ck['action_dim']),int(ck['hidden_size'])); m.load_state_dict(ck['model_state']); m.to(device); return m,ck

def save_checkpoint_v2(path:Path,model,cfg,env_cfg,extra=None):
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({'format':'active_vibration_rig_ppo_v2','obs_dim':model.obs_dim,'action_dim':model.action_dim,'hidden_size':cfg.hidden_size,'model_state':model.state_dict(),'ppo_config':asdict(cfg),'env_config':env_cfg,'extra':extra or {}},path)

def load_checkpoint_v2(path:Path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False); cls=ActorCriticV4 if 'critic_trunk.0.weight' in ck['model_state'] else ActorCriticV2; m=cls(int(ck['obs_dim']),int(ck['action_dim']),int(ck['hidden_size'])); m.load_state_dict(ck['model_state']); m.to(device).eval(); return m,ck

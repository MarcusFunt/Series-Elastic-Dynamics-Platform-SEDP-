"""Collect MPC demonstrations, distill a residual actor, and fine-tune with PPO.

Teacher fallback actions are excluded from fitting. Validation is split by
whole episode. Policies are never promoted by training reward alone.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ppo_agent_v2 import (ActorCriticV2,ActorCriticV4,PPOConfigV2,RolloutBufferV2,ppo_update_v2,
                          save_checkpoint_v2,reflect_observation)
from active_vibration_rig_2d import StepDirParams
from rig_rl_env_v4 import RigRLEnvV4,RLEnvConfigV4,VectorRigEnvV4
from constrained_mpc import ConstrainedMPC,MPCConfig
from residual_control_v4 import FRAME_SIGNS
from evaluate_v4 import reference_preview,evaluate,promotion,load_policy

SCHEMA='sedp-v4-estimated-context-history'


def rollout_diagnostics(sampled_actions,mean_actions,infos,cfg):
    """Summarize actions and reconstruct each unscaled transition reward."""
    sampled=np.asarray(sampled_actions,dtype=float).reshape(-1)
    means=np.asarray(mean_actions,dtype=float).reshape(-1)
    executed=np.asarray([i['effective_residual_accel'] for i in infos],dtype=float)
    costs={k:float(np.mean([i[k] for i in infos]))
           for k in infos[0] if k.startswith('cost_')}
    components={'alive_bonus':float(cfg.alive_bonus),
                'energy_progress':float(np.mean([i['energy_progress_reward'] for i in infos])),
                'termination_penalty':-50.*float(np.mean([i['terminated'] for i in infos])),
                **{k:-v for k,v in costs.items()}}
    reconstructed=np.asarray([cfg.alive_bonus+i['energy_progress_reward']
                  -sum(v for k,v in i.items() if k.startswith('cost_'))
                  -50.*i['terminated'] for i in infos])
    actual=np.asarray([i['unscaled_reward'] for i in infos])
    return {'action_sample_mean':float(sampled.mean()),
            'action_sample_std':float(sampled.std()),
            'action_mean':float(means.mean()),'action_mean_abs':float(np.abs(means).mean()),
            'action_spread':float(means.std()),
            'executed_residual_accel_mean':float(executed.mean()),
            'executed_residual_accel_mean_abs':float(np.abs(executed).mean()),
            'executed_residual_accel_std':float(executed.std()),
            'reward_costs':costs,'reward_components':components,
            'reward_reconstruction_error':float(np.max(np.abs(reconstructed-actual)))}


def metadata(steps,mode,cfg=None,**kwargs):
    actuator_mode=cfg.actuator_mode if cfg is not None else 'torque'
    controller_timing=('held torque at control_dt' if actuator_mode=='torque' else
                       'acceleration-derived velocity target at control_dt; STEP pulses integrated at physics_dt')
    return {'observation_schema':SCHEMA,'steps':steps,'mode':mode,
            'controller_timing':controller_timing,'actuator_mode':actuator_mode,
            'nominal_controller_parameters':True,**kwargs}


def source_revision():
    """Return the repository revision used to produce a training artifact."""
    root=Path(__file__).resolve().parents[2]
    try:
        result=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],
                              capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_teacher(cfg,samples,episodes,seed,horizon):
    if samples<episodes or episodes<2:raise ValueError('Need at least two episodes and samples >= episodes')
    X=[];Y=[];groups=[];valid=[];diagnostics=[]
    per_episode=int(np.ceil(samples/episodes))
    for episode in range(episodes):
        env=RigRLEnvV4(cfg=cfg,seed=seed+1009*episode);obs,_=env.reset()
        # Use the same episode plant model as the LQR controller that anchors
        # this rollout, including any domain-randomized parameter draw.
        teacher=ConstrainedMPC(env.p,env.cp,MPCConfig(horizon=horizon,
                    physics_dt=cfg.physics_dt,control_dt=cfg.control_dt,
                    residual_accel_limit=cfg.residual_accel_limit,
                    actuator_mode=cfg.actuator_mode,step_dir=cfg.step_dir,
                    command_delay=cfg.command_delay,command_jitter=cfg.command_jitter))
        for _ in range(min(per_episode,cfg.max_steps)):
            action=teacher.action(
                env.loop.state,reference_preview(env,horizon),
                command_queue=env.command_queue,step_dir_actuator=env.step_dir_actuator,
                time_s=env.t,kick_torque=env.kick_torque,kick_at=env.kick_at,
                kick_duration=env.kick_duration)
            X.append(obs.copy());Y.append([action]);groups.append(episode)
            valid.append(teacher.diagnostics['success']);diagnostics.append(dict(teacher.diagnostics))
            obs,_,te,tr,_=env.step([action])
            if len(X)%50==0:
                print('EVENT '+json.dumps({'phase':'teacher','episode':episode+1,'samples':len(X),'successful':sum(valid)}),flush=True)
            if te or tr or len(X)>=samples:break
        print('EVENT '+json.dumps({'phase':'teacher','episode':episode+1,'samples':len(X),'successful':sum(valid)}),flush=True)
        if len(X)>=samples:break
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(groups),np.asarray(valid),diagnostics


def distill(X,Y,groups,valid,cfg,epochs,seed):
    torch.manual_seed(seed)
    unique=np.unique(groups);heldout=unique[-max(1,len(unique)//4):]
    train=valid & ~np.isin(groups,heldout);validation=valid & np.isin(groups,heldout)
    if train.sum()<8 or validation.sum()<4:raise RuntimeError('Too few successful MPC labels in training/held-out episodes')
    model=ActorCriticV4(X.shape[1],1,128);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    x=torch.from_numpy(X);target=torch.from_numpy(Y);indices=torch.from_numpy(np.flatnonzero(train))
    signs=np.tile(FRAME_SIGNS,cfg.history_length)
    losses=[];best_validation=float("inf");best_state=None
    for epoch in range(epochs):
        order=indices[torch.randperm(len(indices))]
        for batch in order.split(128):
            pred=model.deterministic(x[batch]);mirrored=model.deterministic(reflect_observation(x[batch],signs))
            loss=(pred-target[batch]).square().mean()+.025*(pred+mirrored).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
        with torch.no_grad():
            tr=float((model.deterministic(x[train])-target[train]).abs().mean())
            va=float((model.deterministic(x[validation])-target[validation]).abs().mean())
        if va<best_validation:
            best_validation=va;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        losses.append({'epoch':epoch+1,'train_action_mae':tr,'heldout_action_mae':va})
        print('EVENT '+json.dumps({'phase':'distill',**losses[-1]}),flush=True)
    model.load_state_dict(best_state)
    return model,{'best_heldout_action_mae':best_validation,'losses':losses,'train_samples':int(train.sum()),'validation_samples':int(validation.sum()),
                  'heldout_episodes':heldout.tolist(),'excluded_fallback_labels':int((~valid).sum())}


def train_ppo(model,cfg,args,outdir):
    pc=PPOConfigV2(hidden_size=128,learning_rate=args.learning_rate,n_envs=args.envs,n_steps=args.rollout,
                   anchor_kl_coef=args.anchor_kl,symmetry_coef=.025,epochs=3,gae_lambda=.98,entropy_coef=.0001,seed=args.seed)
    anchor=type(model)(model.obs_dim,1,128);anchor.load_state_dict(model.state_dict());anchor.eval()
    for p in anchor.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=pc.learning_rate,eps=1e-5)
    env=VectorRigEnvV4(pc.n_envs,args.seed,cfg);obs=torch.from_numpy(env.reset());steps=0;logs=[]
    while steps<args.steps:
        buffer=RolloutBufferV2(pc.n_steps,pc.n_envs,model.obs_dim,1,'cpu');infos_all=[]
        for t in range(pc.n_steps):
            with torch.no_grad():action,lp,_,value=model.sample(obs)
            no,reward,done,infos=env.step(action.numpy())
            buffer.obs[t]=obs;buffer.actions[t]=action;buffer.logp[t]=lp;buffer.values[t]=value
            buffer.rewards[t]=torch.from_numpy(reward);buffer.dones[t]=torch.from_numpy(done)
            for i,info in enumerate(infos):
                if info['truncated']:
                    buffer.truncated[t,i]=1.
                    with torch.no_grad():buffer.final_values[t,i]=model.value(torch.from_numpy(info['final_observation']).unsqueeze(0))[0]
            obs=torch.from_numpy(no);steps+=pc.n_envs;infos_all.extend(infos)
        with torch.no_grad():last=model.value(obs)
        buffer.compute_gae(last,pc.gamma,pc.gae_lambda)
        loss=ppo_update_v2(model,optimizer,buffer,pc,anchor,env.reflection_signs)
        returns=buffer.returns.flatten().numpy()
        pre_update_values=buffer.values.flatten().numpy()
        explained_pre=1-float(np.var(returns-pre_update_values))/max(float(np.var(returns)),1e-8)
        with torch.no_grad():
            updated_values=model.value(buffer.obs.reshape(-1,model.obs_dim)).numpy()
        explained_post=1-float(np.var(returns-updated_values))/max(float(np.var(returns)),1e-8)
        with torch.no_grad():
            mean_actions=model.deterministic(buffer.obs.reshape(-1,model.obs_dim)).numpy()
        diagnostics=rollout_diagnostics(buffer.actions.numpy(),mean_actions,infos_all,cfg)
        log={'steps':steps,**loss,'explained_variance':explained_post,
             'explained_variance_pre_update':explained_pre,
             'mean_reward':float(buffer.rewards.mean()),
             'projection_fraction':float(np.mean([abs(i['requested_residual_accel']-i['effective_residual_accel'])>1e-6 for i in infos_all])),
             **diagnostics}
        logs.append(log);print('PPO',json.dumps(log),flush=True)
        save_checkpoint_v2(outdir/f'policy_{steps:08d}.pt',model,pc,asdict(cfg),metadata(steps,'ppo',cfg=cfg))
    return pc,logs,steps


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['teacher','ppo'])
    ap.add_argument('--dataset',type=Path,help='Re-fit an existing teacher_runs.npz without recollecting');ap.add_argument('--outdir',type=Path,default=Path('runs/v4'));ap.add_argument('--seed',type=int,default=173)
    ap.add_argument('--samples',type=int,default=2400);ap.add_argument('--episodes',type=int,default=8)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--horizon',type=int,default=8)
    ap.add_argument('--history',type=int,default=8);ap.add_argument('--linear-encoder',action='store_true')
    ap.add_argument('--oracle-state',action='store_true');ap.add_argument('--no-preview',action='store_true')
    ap.add_argument('--init',type=Path)
    ap.add_argument('--actuator-mode',choices=['torque','step_dir'])
    ap.add_argument('--step-dir-max-velocity',type=float)
    ap.add_argument('--step-dir-max-acceleration',type=float)
    ap.add_argument('--steps',type=int,default=65536);ap.add_argument('--envs',type=int,default=8)
    ap.add_argument('--rollout',type=int,default=256);ap.add_argument('--learning-rate',type=float,default=3e-4)
    ap.add_argument('--anchor-kl',type=float,default=.001)
    ap.add_argument('--energy-gate-weight',type=float,default=0.)
    ap.add_argument('--energy-gate-step-scale-mjs',type=float,default=1.)
    ap.add_argument('--eval-seeds',nargs='*',type=int,default=[101,202,303]);ap.add_argument('--skip-evaluation',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.epochs<1 or args.steps<1 or args.envs<1 or args.rollout<2:raise ValueError('Training counts must be positive')
    out=args.outdir;out.mkdir(parents=True,exist_ok=True);started=time.time()
    step_dir=StepDirParams(
        max_velocity_rad_s=(55.0 if args.step_dir_max_velocity is None else args.step_dir_max_velocity),
        max_acceleration_rad_s2=(300.0 if args.step_dir_max_acceleration is None else args.step_dir_max_acceleration),
    )
    cfg=RLEnvConfigV4(history_length=args.history,linear_encoder=args.linear_encoder,
                      oracle_state=args.oracle_state,preview_enabled=not args.no_preview,
                      actuator_mode=args.actuator_mode or 'torque',
                      step_dir=step_dir)
    if args.mode=='teacher':
        if args.dataset:
            data=np.load(args.dataset,allow_pickle=False)
            source=json.loads((args.dataset.parent/'teacher_config.json').read_text())
            cfg=RLEnvConfigV4(**source['env_config'])
            if args.actuator_mode is not None and cfg.actuator_mode != args.actuator_mode:
                raise ValueError('Requested actuator mode does not match the saved teacher dataset')
            if (args.step_dir_max_velocity is not None and
                    not np.isclose(cfg.step_dir.max_velocity_rad_s,args.step_dir_max_velocity)):
                raise ValueError('Requested STEP/DIR speed limit does not match the saved teacher dataset')
            if (args.step_dir_max_acceleration is not None and
                    not np.isclose(cfg.step_dir.max_acceleration_rad_s2,args.step_dir_max_acceleration)):
                raise ValueError('Requested STEP/DIR acceleration limit does not match the saved teacher dataset')
            X,Y,groups,valid=[data[k] for k in ('observations','actions','episode','successful')]
            diagnostics=json.loads((args.dataset.parent/'teacher_solver.json').read_text())
        else:
            X,Y,groups,valid,diagnostics=collect_teacher(cfg,args.samples,args.episodes,args.seed,args.horizon)
        teacher_config=source if args.dataset else {'env_config':asdict(cfg),'seed':args.seed,
                    'mpc_config':asdict(MPCConfig(horizon=args.horizon,physics_dt=cfg.physics_dt,
                        control_dt=cfg.control_dt,residual_accel_limit=cfg.residual_accel_limit,
                        actuator_mode=cfg.actuator_mode,step_dir=cfg.step_dir,
                        command_delay=cfg.command_delay,command_jitter=cfg.command_jitter))}
        (out/'teacher_config.json').write_text(json.dumps(teacher_config,indent=2))
        np.savez_compressed(out/'teacher_runs.npz',observations=X,actions=Y,episode=groups,successful=valid)
        (out/'teacher_solver.json').write_text(json.dumps(diagnostics,indent=2))
        model,fit=distill(X,Y,groups,valid,cfg,args.epochs,args.seed)
        (out/'teacher_fit.json').write_text(json.dumps(fit,indent=2))
        pc=PPOConfigV2(hidden_size=128,seed=args.seed);steps=len(X)
    else:
        if args.init:model,cfg,_=load_policy(args.init)
        else:model=ActorCriticV4(26*cfg.history_length,1,128)
        if args.init and args.actuator_mode is not None:
            cfg=replace(cfg,actuator_mode=args.actuator_mode)
        if args.init and args.step_dir_max_velocity is not None:
            cfg=replace(cfg,step_dir=replace(cfg.step_dir,
                                            max_velocity_rad_s=args.step_dir_max_velocity))
        if args.init and args.step_dir_max_acceleration is not None:
            cfg=replace(cfg,step_dir=replace(cfg.step_dir,
                                            max_acceleration_rad_s2=args.step_dir_max_acceleration))
        cfg=replace(cfg,energy_gate_weight=args.energy_gate_weight,
                    energy_gate_step_scale_mJs=args.energy_gate_step_scale_mjs)
        pc,logs,steps=train_ppo(model,cfg,args,out)
        (out/'training.json').write_text(json.dumps(logs,indent=2))
    path=out/'policy_candidate.pt'
    revision=source_revision()
    elapsed=time.time()-started
    run_metadata=metadata(steps,args.mode,cfg=cfg,wall_seconds=elapsed,
                          source_revision=revision,seed=args.seed)
    (out/'run_metadata.json').write_text(json.dumps({
        **run_metadata,
        'arguments':{key:(str(value) if isinstance(value,Path) else value)
                     for key,value in vars(args).items()},
        'python_version':__import__('platform').python_version(),
        'numpy_version':np.__version__,
        'torch_version':torch.__version__,
    },indent=2))
    save_checkpoint_v2(path,model,pc,asdict(cfg),run_metadata)
    if not args.skip_evaluation:
        print('EVENT '+json.dumps({'phase':'evaluation'}),flush=True)
        rows=evaluate(model,cfg,args.eval_seeds)
        decision=promotion(rows)
        result={'benchmark_version':'SEDP-V4-100HZ','config':asdict(cfg),'rows':rows,'promotion':decision}
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))
        if decision['accepted'] and not cfg.oracle_state:
            save_checkpoint_v2(out/'policy_accepted.pt',model,pc,asdict(cfg),metadata(steps,args.mode,cfg=cfg,promotion=decision))
        print('promotion',decision,flush=True)
    print('candidate',path,flush=True)

if __name__=='__main__':main()

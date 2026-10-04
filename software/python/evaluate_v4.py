"""Paired deterministic and randomized evaluation through the training loop."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from active_vibration_rig_2d import Trajectory, MotionParams
from benchmark_suite import SCENARIOS
from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4
from constrained_mpc import ConstrainedMPC, MPCConfig


def reference_preview(env,horizon):
    preview=getattr(env.reference,'preview',env.reference.sample)
    return [preview(env.t+k*env.cfg.control_dt) for k in range(horizon+1)]


def run_case(controller, scenario=None, *, seed=0, cfg=None, model=None, horizon=40, randomized=False):
    cfg=replace(cfg or RLEnvConfigV4(),domain_randomization=randomized,
                initial_theta_std=np.radians(1.) if randomized else 0.,
                initial_theta_dot_std=np.radians(5.) if randomized else 0.,
                kick_probability=1. if randomized else 0.,
                episode_seconds=scenario.duration if scenario is not None else (cfg or RLEnvConfigV4()).episode_seconds)
    env=RigRLEnvV4(cfg=cfg,seed=seed);obs,_=env.reset()
    if scenario is not None:
        obs=env.set_reference(Trajectory(scenario.trajectory,MotionParams()))
    teacher=ConstrainedMPC(env.base_params,env.cp,MPCConfig(horizon=horizon,
                          physics_dt=cfg.physics_dt,control_dt=cfg.control_dt,
                          residual_accel_limit=cfg.residual_accel_limit)) if controller=='mpc' else None
    theta=[];track=[];rail=[];energy=[];sat=[];xerr=[];solves=[];fallback=0;projected=0;terminated=False
    for _ in range(cfg.max_steps):
        if teacher is not None:
            action=teacher.action(env.loop.state,reference_preview(env,horizon))
            solves.append(teacher.diagnostics['solve_seconds']);fallback+=not teacher.diagnostics['success']
        elif controller=='policy':
            import torch
            with torch.no_grad():action=float(model.deterministic(torch.tensor(obs).unsqueeze(0))[0,0])
        elif controller=='lqr': action=0.
        else: raise ValueError(controller)
        obs,_,te,tr,info=env.step([action])
        theta.append(np.degrees(env.y[4]));track.append(1000*(env.y[2]-env.reference.sample(env.t)[0]))
        rail.append(info['interval_peak_rail_fraction']);energy.append(info['energy']);xerr.append(1000*info['estimator_position_error'])
        sat.append(info['interval_saturation'])
        projected+=abs(info['requested_residual_accel']-info['effective_residual_accel'])>1e-6
        if te:terminated=True;break
        if tr:break
    count=len(theta)
    return {'controller':controller,'scenario':scenario.name if scenario else 'random_reference',
            'seed':seed,'randomized':randomized,'peak_angle_deg':float(np.max(np.abs(theta))),
            'angle_rms_deg':float(np.sqrt(np.mean(np.square(theta)))),
            'position_rmse_mm':float(np.sqrt(np.mean(np.square(track)))),
            'peak_rail_fraction':float(max(rail)),'saturation_fraction':float(np.mean(sat)),
            'stop_contact_fraction':float(np.mean(np.array(rail)>=1.)),
            'energy_integral_mJs':float(np.sum(energy)*cfg.control_dt*1000),
            'estimator_position_rmse_mm':float(np.sqrt(np.mean(np.square(xerr)))),
            'projection_fraction':projected/count,'terminated':terminated,
            'mpc_fallback_fraction':fallback/count if teacher else 0.,
            'solve_p50_ms':float(np.median(solves)*1000) if solves else 0.,
            'solve_p95_ms':float(np.quantile(solves,.95)*1000) if solves else 0.,'samples':count}


def evaluate(model=None,cfg=None,seeds=(101,202,303),include_mpc=False,horizon=40):
    rows=[]
    cases=[(sc,0,False) for sc in SCENARIOS]+[(None,int(seed),True) for seed in seeds]
    for scenario,seed,randomized in cases:
        for ctrl in ['lqr']+(['mpc'] if include_mpc else [])+(['policy'] if model is not None else []):
            rows.append(run_case(ctrl,scenario,seed=seed,cfg=cfg,model=model,horizon=horizon,randomized=randomized))
    return rows


def promotion(rows):
    """Require every gate and a paired aggregate gain without peak regression."""
    base={(r['scenario'],r['seed'],r['randomized']):r for r in rows if r['controller']=='lqr'}
    candidate=[r for r in rows if r['controller']=='policy'];reasons=[];ratios=[]
    if len(candidate)!=len(base):reasons.append('candidate coverage does not match paired baseline cases')
    caps={s.name:s.position_rmse_cap_mm for s in SCENARIOS}
    for r in candidate:
        key=(r['scenario'],r['seed'],r['randomized']);b=base[key]
        metrics=('peak_angle_deg','position_rmse_mm','peak_rail_fraction','saturation_fraction','stop_contact_fraction','energy_integral_mJs')
        if not all(np.isfinite(row[k]) and row[k]>=0 for row in (r,b) for k in metrics):
            reasons.append(f'{key}: invalid metrics');ratios.append(float('inf'));continue
        if r['terminated'] or r['peak_rail_fraction']>.78 or r['saturation_fraction']>.08 or r['stop_contact_fraction']>0:
            reasons.append(f'{key}: safety gate')
        if r['position_rmse_mm']>caps.get(r['scenario'],max(8.,b['position_rmse_mm']*1.02)):
            reasons.append(f'{key}: tracking gate')
        if r['peak_angle_deg']>b['peak_angle_deg']*1.02+1e-3:
            reasons.append(f'{key}: peak angle regression')
        if r['position_rmse_mm']>b['position_rmse_mm']*1.02+.05:
            reasons.append(f'{key}: paired tracking regression')
        ratios.append(r['energy_integral_mJs']/max(b['energy_integral_mJs'],1e-8))
    mean=float(np.mean(ratios)) if ratios else float('inf')
    if mean>=.95:reasons.append('less than 5% mean energy improvement over paired LQR')
    return {'accepted':bool(candidate) and not reasons,'mean_energy_ratio':mean,'reasons':reasons}


def load_policy(path):
    from ppo_agent_v2 import load_checkpoint_v2
    model,ck=load_checkpoint_v2(Path(path))
    if ck.get('extra',{}).get('observation_schema')!='sedp-v4-estimated-context-history':
        raise ValueError('Expected a v4 checkpoint; legacy checkpoints cannot be reused with this observation schema')
    cfg=RLEnvConfigV4(**ck['env_config'])
    if model.obs_dim!=26*cfg.history_length or model.action_dim!=1:
        raise ValueError('Checkpoint observation/action dimensions do not match v4 configuration')
    return model,cfg,ck


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--mpc',action='store_true')
    ap.add_argument('--horizon',type=int,default=40);ap.add_argument('--seeds',nargs='*',type=int,default=[101,202,303])
    ap.add_argument('--json',type=Path,required=True);args=ap.parse_args()
    model=None;cfg=RLEnvConfigV4()
    if args.model:model,cfg,_=load_policy(args.model)
    rows=evaluate(model,cfg,args.seeds,args.mpc,args.horizon)
    result={'benchmark_version':'SEDP-V4-100HZ','config':asdict(cfg),'horizon':args.horizon,
            'rows':rows,'promotion':promotion(rows) if model else None}
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(result,indent=2))
    for r in rows: print(r)
    print('promotion',result['promotion'])
    return 0

if __name__=='__main__':raise SystemExit(main())

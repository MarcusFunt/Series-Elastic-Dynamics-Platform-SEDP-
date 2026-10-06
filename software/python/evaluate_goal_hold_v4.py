"""Paired PPO/LQR/MPC evaluation on randomized, multi-second goal holds."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from constrained_mpc import ConstrainedMPC, MPCConfig
from evaluate_v4 import load_policy, reference_preview
from rig_rl_env_v4 import RigRLEnvV4


def run_case(controller, seed, cfg, model=None, horizon=8, tolerance_mm=5.):
    env=RigRLEnvV4(cfg=cfg,seed=seed)
    obs,_=env.reset()
    if cfg.reference_mode!='random_goal_hold':
        raise ValueError('Checkpoint/environment must use the random_goal_hold reference')
    teacher=None
    if controller=='mpc':
        teacher=ConstrainedMPC(env.p,env.cp,MPCConfig(
            horizon=horizon,physics_dt=cfg.physics_dt,control_dt=cfg.control_dt,
            residual_accel_limit=cfg.residual_accel_limit,
            actuator_mode=cfg.actuator_mode,step_dir=cfg.step_dir,
            command_delay=cfg.command_delay,command_jitter=cfg.command_jitter,
            angle_weight=3.2,angular_rate_weight=.8,action_weight=.04))

    rows=[]; terminated=False; fallback=0; solves=[]
    hold_runs={}; current_run={}; hold_targets={}
    for _ in range(cfg.max_steps):
        if controller=='lqr':
            action=0.
        elif controller=='mpc':
            action=teacher.action(env.loop.state,reference_preview(env,horizon),
                command_queue=env.command_queue,step_dir_actuator=env.step_dir_actuator,
                time_s=env.t,kick_torque=env.kick_torque,kick_at=env.kick_at,
                kick_duration=env.kick_duration)
            fallback+=int(not teacher.diagnostics['success'])
            solves.append(teacher.diagnostics['solve_seconds'])
        elif controller=='policy':
            with torch.no_grad():
                action=float(model.deterministic(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))[0,0])
        else:
            raise ValueError(controller)
        obs,_,te,tr,info=env.step([action])
        x=float(env.y[2]); theta=float(env.y[4]); t=float(env.t)
        ref=env.reference
        segment=int(ref.change_count); target=float(ref.x1)
        in_hold=(t-float(ref.t0) >= float(ref.Tmove)-1e-9)
        in_tolerance=abs(x-target) <= tolerance_mm/1000.
        hold_targets[segment]=target
        if in_hold:
            current_run[segment]=current_run.get(segment,0.)+cfg.control_dt if in_tolerance else 0.
            hold_runs[segment]=max(hold_runs.get(segment,0.),current_run[segment])
        else:
            current_run[segment]=0.
        rows.append((t,x,float(info['x_ref']),theta,float(info['energy']),
                     float(info['interval_peak_rail_fraction'])*env.p.rail_half_travel>
                     env.p.rail_half_travel-cfg.goal_edge_margin_m,
                     float(info['interval_saturation']),te or tr))
        if te:
            terminated=True
            break
        if tr:
            break

    a=np.asarray(rows,dtype=float)
    if len(a)==0:
        raise RuntimeError('No transitions were evaluated')
    targets=[v for k,v in sorted(hold_targets.items()) if k in hold_runs]
    completed=sum(v>=2.0-1e-9 for v in hold_runs.values())
    max_contiguous=max(hold_runs.values(),default=0.)
    energy=float(np.sum(a[:,4])*cfg.control_dt*1000.)
    return {
        'controller':controller,'seed':int(seed),'steps':len(a),'terminated':terminated,
        'integrated_resonator_energy_mJs':energy,
        'position_rmse_mm':float(np.sqrt(np.mean(np.square((a[:,1]-a[:,2])*1000.)))),
        'angle_rms_deg':float(np.degrees(np.sqrt(np.mean(np.square(a[:,3]))))),
        'peak_angle_deg':float(np.degrees(np.max(np.abs(a[:,3])))),
        'peak_rail_fraction':float(np.max(np.abs(a[:,1]))/env.p.rail_half_travel),
        'exclusion_zone_violations':int(np.sum(a[:,5])),
        'exclusion_zone_violation_fraction':float(np.mean(a[:,5])),
        'saturation_fraction':float(np.mean(a[:,6])),
        'goal_count':len(targets),'goal_targets_mm':[round(x*1000.,3) for x in targets],
        'goal_targets_respect_margin':all(abs(x)<=env.p.rail_half_travel-cfg.goal_edge_margin_m+1e-9 for x in targets),
        'completed_two_second_holds':int(completed),
        'two_second_hold_fraction':float(completed/max(len(targets),1)),
        'longest_contiguous_in_tolerance_hold_s':float(max_contiguous),
        'hold_tolerance_mm':float(tolerance_mm),
        'mpc_fallback_fraction':fallback/max(len(a),1) if controller=='mpc' else 0.,
        'mpc_solve_p95_ms':float(np.quantile(solves,.95)*1000.) if solves else 0.,
    }


def summarize(rows):
    by={(r['controller'],r['seed']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    result={}
    for ctrl in ('lqr','mpc','policy'):
        group=[by[(ctrl,s)] for s in seeds]
        result[ctrl]={
            'mean_integrated_resonator_energy_mJs':float(np.mean([
                r['integrated_resonator_energy_mJs'] for r in group])),
            'mean_position_rmse_mm':float(np.mean([r['position_rmse_mm'] for r in group])),
            'mean_angle_rms_deg':float(np.mean([r['angle_rms_deg'] for r in group])),
            'mean_peak_angle_deg':float(np.mean([r['peak_angle_deg'] for r in group])),
            'all_targets_respect_margin':all(r['goal_targets_respect_margin'] for r in group),
            'all_runs_avoid_exclusion_zone':all(r['exclusion_zone_violations']==0 for r in group),
            'all_runs_complete_two_second_holds':all(
                r['completed_two_second_holds']==r['goal_count'] for r in group),
            'mean_two_second_hold_fraction':float(np.mean([r['two_second_hold_fraction'] for r in group])),
        }
        if ctrl=='mpc':
            result[ctrl]['mean_fallback_fraction']=float(np.mean([r['mpc_fallback_fraction'] for r in group]))
            result[ctrl]['mean_solve_p95_ms']=float(np.mean([r['mpc_solve_p95_ms'] for r in group]))

    policy=[by[('policy',s)] for s in seeds]
    lqr=[by[('lqr',s)] for s in seeds]
    ratios=[p['integrated_resonator_energy_mJs']/max(b['integrated_resonator_energy_mJs'],1e-9)
            for p,b in zip(policy,lqr)]
    tracking_ok=all(p['position_rmse_mm']<=b['position_rmse_mm']*1.02+.05 for p,b in zip(policy,lqr))
    angle_ok=all(p['peak_angle_deg']<=b['peak_angle_deg']*1.02+.01 for p,b in zip(policy,lqr))
    safe=all(p['terminated'] is False and p['exclusion_zone_violations']==0 and
             p['saturation_fraction']<=.08 for p in policy)
    holds=all(p['completed_two_second_holds']==p['goal_count'] and p['goal_count']>0 for p in policy)
    mean_energy_ratio=float(np.mean(ratios))
    return {
        'policy_lqr_integrated_resonator_energy_ratio':mean_energy_ratio,
        'policy_mpc_integrated_resonator_energy_ratio':float(np.mean([
            by[('policy',s)]['integrated_resonator_energy_mJs']/max(
                by[('mpc',s)]['integrated_resonator_energy_mJs'],1e-9)
            for s in seeds])),
        'paired_tracking_gate':tracking_ok,'paired_peak_angle_gate':angle_ok,
        'safety_and_exclusion_gate':safe,'two_second_hold_gate':holds,
        'integrated_resonator_energy_improvement_at_least_5pct':mean_energy_ratio<.95,
        'accepted_vs_lqr':bool(safe and holds and tracking_ok and angle_ok and mean_energy_ratio<.95),
    },result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--seeds',nargs='*',type=int,default=[8011,8012,8013,8014,8015])
    parser.add_argument('--horizon',type=int,default=8)
    parser.add_argument('--tolerance-mm',type=float,default=5.)
    args=parser.parse_args()
    model,cfg,ck=load_policy(args.model)
    if cfg.reference_mode!='random_goal_hold':
        raise ValueError('Model checkpoint was not trained with --reference-mode random_goal_hold')
    cfg=replace(cfg,domain_randomization=True,initial_theta_std=math.radians(1.),
                initial_theta_dot_std=math.radians(5.),kick_probability=1.)
    rows=[]
    torch.set_num_threads(1)
    for seed in args.seeds:
        for controller in ('lqr','mpc','policy'):
            result=run_case(controller,seed,cfg,model,args.horizon,args.tolerance_mm)
            rows.append(result)
            print('EVENT '+json.dumps({'controller':controller,'seed':seed,
                                       'completed_holds':result['completed_two_second_holds'],
                                       'goals':result['goal_count'],
                                       'exclusion_violations':result['exclusion_zone_violations']}),flush=True)
    decision,summary=summarize(rows)
    output={'benchmark_version':'SEDP-V4-RANDOM-GOAL-HOLD','config':asdict(cfg),
            'checkpoint':str(args.model),'source_revision':ck.get('extra',{}).get('source_revision'),
            'hold_tolerance_mm':args.tolerance_mm,'rows':rows,'summary':summary,
            'promotion':decision}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(output,indent=2))
    print('promotion '+json.dumps(decision),flush=True)
    return 0 if decision['accepted_vs_lqr'] else 2


if __name__=='__main__':
    raise SystemExit(main())

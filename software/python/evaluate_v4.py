"""Paired deterministic and randomized evaluation through the training loop."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from active_vibration_rig_2d import Trajectory, MotionParams
from benchmark_suite import SCENARIOS
from rig_rl_env_v4 import RigRLEnvV4, RLEnvConfigV4
from constrained_mpc import ConstrainedMPC, MPCConfig
from legacy_constrained_mpc import LegacyConstrainedMPC, LegacyMPCConfig


MPC_TUNED_WEIGHTS = {'angle_weight': 3.2, 'angular_rate_weight': .8,
                     'action_weight': .04}


def reference_preview(env,horizon):
    preview=getattr(env.reference,'preview',env.reference.sample)
    return [preview(env.t+k*env.cfg.control_dt) for k in range(horizon+1)]


def run_case(controller, scenario=None, *, seed=0, cfg=None, model=None, horizon=8, randomized=False):
    cfg=replace(cfg or RLEnvConfigV4(),domain_randomization=randomized,
                initial_theta_std=np.radians(1.) if randomized else 0.,
                initial_theta_dot_std=np.radians(5.) if randomized else 0.,
                kick_probability=1. if randomized else 0.,
                episode_seconds=scenario.duration if scenario is not None else (cfg or RLEnvConfigV4()).episode_seconds)
    env=RigRLEnvV4(cfg=cfg,seed=seed);obs,_=env.reset()
    if scenario is not None:
        obs=env.set_reference(Trajectory(scenario.trajectory,MotionParams()))
    teacher=None
    if controller in ('mpc','mpc_baseline'):
        if controller == 'mpc_baseline':
            if cfg.actuator_mode != 'torque' or cfg.command_delay != 0 or cfg.command_jitter != 0:
                raise ValueError('The preserved pre-Phase-4 MPC baseline supports torque mode with zero command delay only')
            teacher=LegacyConstrainedMPC(env.base_params,env.cp,LegacyMPCConfig(
                horizon=horizon,physics_dt=cfg.physics_dt,control_dt=cfg.control_dt,
                residual_accel_limit=cfg.residual_accel_limit))
        else:
            mpc_options = MPC_TUNED_WEIGHTS
            # Match the plant model used by env.loop.controller. Under domain
            # randomization that controller is built from env.p, not the
            # nominal env.base_params; using nominal MPC parameters would give
            # the paired LQR baseline privileged model accuracy.
            teacher=ConstrainedMPC(env.p,env.cp,MPCConfig(
                horizon=horizon,physics_dt=cfg.physics_dt,control_dt=cfg.control_dt,
                residual_accel_limit=cfg.residual_accel_limit,actuator_mode=cfg.actuator_mode,
                step_dir=cfg.step_dir,command_delay=cfg.command_delay,
                command_jitter=cfg.command_jitter,**mpc_options))
    theta=[];track=[];rail=[];energy=[];sat=[];xerr=[];solves=[];fallback=0;projected=0;terminated=False
    soft_rail_violation=[];carriage_speed_violation=[];motor_speed_violation=[]
    mpc_outcomes={};mpc_reasons={};mpc_nonlinear_violation=0.;mpc_peak_predicted_rail=0.;mpc_torque_saturation_intervals=0
    for _ in range(cfg.max_steps):
        if teacher is not None:
            if controller == 'mpc_baseline':
                action=teacher.action(env.loop.state,reference_preview(env,horizon))
            else:
                action=teacher.action(
                    env.loop.state,reference_preview(env,horizon),
                    command_queue=env.command_queue,step_dir_actuator=env.step_dir_actuator,
                    time_s=env.t,kick_torque=env.kick_torque,kick_at=env.kick_at,
                    kick_duration=env.kick_duration)
            diag=teacher.diagnostics
            solves.append(diag['solve_seconds']);fallback+=not diag['success']
            mpc_outcomes[diag['outcome']]=mpc_outcomes.get(diag['outcome'],0)+1
            reason=diag.get('reason','') or 'success'
            mpc_reasons[reason]=mpc_reasons.get(reason,0)+1
            mpc_nonlinear_violation=max(mpc_nonlinear_violation,
                                        float(diag.get('nonlinear_max_constraint_violation',0.)))
            mpc_peak_predicted_rail=max(mpc_peak_predicted_rail,
                                        float(diag.get('nonlinear_peak_rail_fraction',0.)))
            mpc_torque_saturation_intervals+=int(diag.get('step_dir_torque_saturated_intervals',0))
        elif controller=='policy':
            import torch
            with torch.no_grad():action=float(model.deterministic(torch.tensor(obs).unsqueeze(0))[0,0])
        elif controller=='lqr': action=0.
        else: raise ValueError(controller)
        obs,_,te,tr,info=env.step([action])
        theta.append(np.degrees(env.y[4]));track.append(1000*(env.y[2]-env.reference.sample(env.t)[0]))
        rail.append(info['interval_peak_rail_fraction']);energy.append(info['energy']);xerr.append(1000*info['estimator_position_error'])
        sat.append(info['interval_saturation'])
        soft_limit=cfg.rail_soft_fraction*env.p.rail_half_travel
        soft_rail_violation.append(max(0.,abs(float(env.y[2]))-soft_limit)/max(env.p.rail_half_travel,1e-12))
        carriage_speed_violation.append(max(0.,abs(float(env.y[3]))-env.cp.max_speed))
        motor_speed_violation.append(max(0.,abs(float(env.y[1]))-env.cp.max_speed/max(env.p.pulley_radius,1e-12)))
        projected+=abs(info['requested_residual_accel']-info['effective_residual_accel'])>1e-6
        if te:terminated=True;break
        if tr:break
    count=len(theta)
    result={'controller':controller,'scenario':scenario.name if scenario else 'random_reference',
            'seed':seed,'randomized':randomized,'peak_angle_deg':float(np.max(np.abs(theta))),
            'angle_rms_deg':float(np.sqrt(np.mean(np.square(theta)))),
            'position_rmse_mm':float(np.sqrt(np.mean(np.square(track)))),
            'peak_rail_fraction':float(max(rail)),'saturation_fraction':float(np.mean(sat)),
            'stop_contact_fraction':float(np.mean(np.array(rail)>=1.)),
            'energy_integral_mJs':float(np.sum(energy)*cfg.control_dt*1000),
            'estimator_position_rmse_mm':float(np.sqrt(np.mean(np.square(xerr)))),
            'projection_fraction':projected/count,'terminated':terminated,
            'max_soft_rail_violation_fraction':float(max(soft_rail_violation)),
            'max_carriage_speed_violation_m_s':float(max(carriage_speed_violation)),
            'max_motor_speed_violation_rad_s':float(max(motor_speed_violation)),
            'mpc_fallback_fraction':fallback/count if teacher else 0.,
            'solve_p50_ms':float(np.median(solves)*1000) if solves else 0.,
            'solve_p95_ms':float(np.quantile(solves,.95)*1000) if solves else 0.,'samples':count,
            'actuator_mode':cfg.actuator_mode,'command_delay_seconds':cfg.command_delay,
            'command_jitter_seconds':cfg.command_jitter}
    if teacher:
        result.update({
            'mpc_outcome_counts':mpc_outcomes,
            'mpc_outcome_rates':{key:value/count for key,value in mpc_outcomes.items()},
            'mpc_reason_counts':mpc_reasons,
            'mpc_nonlinear_max_constraint_violation':mpc_nonlinear_violation,
            'mpc_nonlinear_peak_rail_fraction':mpc_peak_predicted_rail,
            'step_dir_torque_saturated_intervals':mpc_torque_saturation_intervals,
        })
    return result


def evaluate(model=None,cfg=None,seeds=(101,202,303),include_mpc=False,horizon=8,
             compare_mpc=False,scenarios=None):
    rows=[]
    selected_scenarios=SCENARIOS if scenarios is None else tuple(scenarios)
    cases=[(sc,0,False) for sc in selected_scenarios]+[(None,int(seed),True) for seed in seeds]
    mpc_controllers=(['mpc_baseline','mpc'] if compare_mpc else ['mpc']) if include_mpc or compare_mpc else []
    for scenario,seed,randomized in cases:
        for ctrl in ['lqr']+mpc_controllers+(['policy'] if model is not None else []):
            total_controllers=1+len(mpc_controllers)+int(model is not None)
            print('EVENT '+json.dumps({'phase':'evaluation','controller':ctrl,'case':scenario.name if scenario else 'random_reference','seed':seed,'completed_cases':len(rows),'total_cases':len(cases)*total_controllers}),flush=True)
            rows.append(run_case(ctrl,scenario,seed=seed,cfg=cfg,model=model,horizon=horizon,randomized=randomized))
    return rows


def mpc_comparison(rows):
    """Compare Phase 4 weights to the pre-Phase-4 MPC cost profile per pair."""
    key=lambda r:(r['scenario'],r['seed'],r['randomized'],r.get('actuator_mode'),
                  r.get('command_delay_seconds'),r.get('command_jitter_seconds'))
    reference={key(row):row for row in rows if row['controller']=='mpc_baseline'}
    candidate={key(row):row for row in rows if row['controller']=='mpc'}
    reasons=[];paired=[]
    if not reference or set(reference)!=set(candidate):
        reasons.append('MPC candidate coverage does not match the paired baseline cases')
    metrics=('angle_rms_deg','peak_angle_deg','position_rmse_mm','peak_rail_fraction',
             'saturation_fraction','stop_contact_fraction','max_soft_rail_violation_fraction',
             'max_carriage_speed_violation_m_s','max_motor_speed_violation_rad_s',
             'energy_integral_mJs','mpc_fallback_fraction')
    for pair in sorted(set(reference)&set(candidate),key=str):
        b=reference[pair];c=candidate[pair]
        changes={name:float(c[name]-b[name]) for name in metrics}
        for name in ('angle_rms_deg','peak_angle_deg','position_rmse_mm'):
            if c[name] > b[name]*1.02 + (0.05 if name=='position_rmse_mm' else .01):
                reasons.append(f'{pair}: {name} regressed beyond 2% paired bound')
        if c['peak_rail_fraction'] > b['peak_rail_fraction']+.01:
            reasons.append(f'{pair}: peak rail fraction regressed by more than 0.01')
        if c['saturation_fraction'] > b['saturation_fraction']+.02:
            reasons.append(f'{pair}: saturation fraction regressed by more than 0.02')
        if c['mpc_fallback_fraction'] > b['mpc_fallback_fraction']+.02:
            reasons.append(f'{pair}: MPC fallback fraction regressed by more than 0.02')
        if c['stop_contact_fraction'] > b['stop_contact_fraction']:
            reasons.append(f'{pair}: stop contact increased')
        if c['max_soft_rail_violation_fraction'] > b['max_soft_rail_violation_fraction']+.005:
            reasons.append(f'{pair}: soft rail constraint violation regressed by more than 0.005')
        if c['max_carriage_speed_violation_m_s'] > b['max_carriage_speed_violation_m_s']+1e-4:
            reasons.append(f'{pair}: carriage speed constraint violation regressed by more than 0.0001 m/s')
        if c['max_motor_speed_violation_rad_s'] > b['max_motor_speed_violation_rad_s']+1e-3:
            reasons.append(f'{pair}: motor speed constraint violation regressed by more than 0.001 rad/s')
        if c['terminated'] and not b['terminated']:
            reasons.append(f'{pair}: candidate terminated where baseline did not')
        paired.append({'scenario':pair[0],'seed':pair[1],'randomized':pair[2],
                       'actuator_mode':pair[3],'changes':changes,
                       'baseline_outcomes':b.get('mpc_outcome_counts',{}),
                       'candidate_outcomes':c.get('mpc_outcome_counts',{})})
    if paired:
        rms_ratio=float(np.mean([candidate[p]['angle_rms_deg']/max(reference[p]['angle_rms_deg'],1e-9)
                                 for p in candidate if p in reference]))
        energy_ratio=float(np.mean([candidate[p]['energy_integral_mJs']/max(reference[p]['energy_integral_mJs'],1e-9)
                                    for p in candidate if p in reference]))
        fallback_delta=float(np.mean([candidate[p]['mpc_fallback_fraction']-reference[p]['mpc_fallback_fraction']
                                      for p in candidate if p in reference]))
        if rms_ratio >= .98 and energy_ratio >= .98:
            reasons.append('neither paired vibration RMS nor energy improves by at least 2%')
        return {'accepted':bool(paired) and not reasons,
                'mean_angle_rms_ratio':rms_ratio,'mean_energy_ratio':energy_ratio,
                'mean_fallback_fraction_delta':fallback_delta,
                'per_case_regressions_checked':True,'reasons':reasons,'pairs':paired}
    return {'accepted':False,'mean_angle_rms_ratio':None,'mean_energy_ratio':None,
            'mean_fallback_fraction_delta':None,'per_case_regressions_checked':True,
            'reasons':reasons,'pairs':paired}


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
    ap.add_argument('--compare-mpc',action='store_true',help='Compare pre-Phase-4 and tuned objective weights')
    ap.add_argument('--actuator-mode',choices=['torque','step_dir'],default='torque')
    ap.add_argument('--horizon',type=int,default=8);ap.add_argument('--seeds',nargs='*',type=int,default=[101,202,303])
    ap.add_argument('--json',type=Path,required=True);args=ap.parse_args()
    model=None;cfg=RLEnvConfigV4(actuator_mode=args.actuator_mode)
    if args.model:model,cfg,_=load_policy(args.model)
    rows=evaluate(model,cfg,args.seeds,args.mpc,args.horizon,args.compare_mpc)
    result={'benchmark_version':'SEDP-V4-100HZ','config':asdict(cfg),'horizon':args.horizon,
            'rows':rows,'promotion':promotion(rows) if model else None,
            'mpc_comparison':mpc_comparison(rows) if args.compare_mpc else None}
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(result,indent=2))
    for r in rows: print(r)
    print('promotion',result['promotion'])
    return 0

if __name__=='__main__':raise SystemExit(main())

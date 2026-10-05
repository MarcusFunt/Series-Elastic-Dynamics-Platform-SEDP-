"""Reproduce the bounded Phase 4 MPC comparison.

Runs paired move, reversal, and stress cases for the preserved legacy MPC
and Phase 4 MPC on the torque baseline, then runs the Phase 4 MPC against LQR
with STEP/DIR and with command delay/jitter enabled.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from benchmark_suite import SCENARIOS
from evaluate_v4 import evaluate, mpc_comparison
from rig_rl_env_v4 import RLEnvConfigV4


def paired_rows(rows, left, right):
    key=lambda row:(row['scenario'],row['seed'],row['randomized'],row['actuator_mode'],
                    row['command_delay_seconds'],row['command_jitter_seconds'])
    a={key(row):row for row in rows if row['controller']==left}
    b={key(row):row for row in rows if row['controller']==right}
    result=[]
    for pair in sorted(set(a)&set(b),key=str):
        result.append({
            'scenario':pair[0],'seed':pair[1],'actuator_mode':pair[3],
            'command_delay_seconds':pair[4],'command_jitter_seconds':pair[5],
            'left':left,'right':right,
            'position_rmse_mm':{'left':a[pair]['position_rmse_mm'],'right':b[pair]['position_rmse_mm']},
            'angle_rms_deg':{'left':a[pair]['angle_rms_deg'],'right':b[pair]['angle_rms_deg']},
            'peak_angle_deg':{'left':a[pair]['peak_angle_deg'],'right':b[pair]['peak_angle_deg']},
            'peak_rail_fraction':{'left':a[pair]['peak_rail_fraction'],'right':b[pair]['peak_rail_fraction']},
            'stop_contact_fraction':{'left':a[pair]['stop_contact_fraction'],'right':b[pair]['stop_contact_fraction']},
            'max_soft_rail_violation_fraction':{
                'left':a[pair]['max_soft_rail_violation_fraction'],
                'right':b[pair]['max_soft_rail_violation_fraction']},
            'max_carriage_speed_violation_m_s':{
                'left':a[pair]['max_carriage_speed_violation_m_s'],
                'right':b[pair]['max_carriage_speed_violation_m_s']},
            'max_motor_speed_violation_rad_s':{
                'left':a[pair]['max_motor_speed_violation_rad_s'],
                'right':b[pair]['max_motor_speed_violation_rad_s']},
            'terminated':{'left':a[pair]['terminated'],'right':b[pair]['terminated']},
            'energy_integral_mJs':{'left':a[pair]['energy_integral_mJs'],'right':b[pair]['energy_integral_mJs']},
            'fallback_fraction':{'left':a[pair]['mpc_fallback_fraction'],
                                 'right':b[pair]['mpc_fallback_fraction']},
            'outcome_counts':{'left':a[pair].get('mpc_outcome_counts',{}),
                              'right':b[pair].get('mpc_outcome_counts',{})},
            'solve_p50_ms':{'left':a[pair]['solve_p50_ms'],'right':b[pair]['solve_p50_ms']},
            'solve_p95_ms':{'left':a[pair]['solve_p95_ms'],'right':b[pair]['solve_p95_ms']},
        })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--horizon',type=int,default=4)
    parser.add_argument('--duration',type=float,default=1.6)
    parser.add_argument('--command-delay',type=float,default=.015)
    parser.add_argument('--command-jitter',type=float,default=.003)
    args=parser.parse_args()
    scenarios=tuple(replace(s,duration=args.duration) for s in SCENARIOS)
    shared={'domain_randomization':False,'kick_probability':0.,
            'initial_theta_std':0.,'initial_theta_dot_std':0.,'sensor_noise':False}

    torque_cfg=RLEnvConfigV4(**shared)
    torque_rows=evaluate(cfg=torque_cfg,seeds=(),include_mpc=True,horizon=args.horizon,
                         compare_mpc=True,scenarios=scenarios)
    step_cfg=RLEnvConfigV4(**shared,actuator_mode='step_dir')
    step_rows=evaluate(cfg=step_cfg,seeds=(),include_mpc=True,horizon=args.horizon,
                       scenarios=scenarios)
    delayed_cfg=RLEnvConfigV4(**shared,command_delay=args.command_delay,
                              command_jitter=args.command_jitter)
    delayed_rows=evaluate(cfg=delayed_cfg,seeds=(),include_mpc=True,horizon=args.horizon,
                          scenarios=scenarios)
    rows=torque_rows+step_rows+delayed_rows
    result={
        'report':'SEDP Phase 4 constrained MPC comparison',
        'method':{
            'scenario_names':[s.name for s in scenarios],
            'trajectory_modes':{s.name:s.trajectory for s in scenarios},
            'duration_seconds':args.duration,'horizon':args.horizon,
            'physics_dt':torque_cfg.physics_dt,'control_dt':torque_cfg.control_dt,
            'initial_state':'zero state, identical for paired controllers',
            'randomization':False,'disturbance_torque':0.,
            'legacy_baseline':'preserved pre-Phase-4 ConstrainedMPC source and objective, torque mode with zero command latency',
            'tuned_objective_weights':{'position':1.8,'velocity':.12,'angle':3.2,
                                       'angular_rate':.8,'action':.04,'slew':.06},
            'tuned_scales':{'position':.006,'velocity':.35,'angle_rad':.0872664626,
                            'angular_rate_rad_s':1.745329252,'acceleration_residual':2.5},
            'step_dir_params':asdict(step_cfg.step_dir),
            'delayed_command':{'delay_seconds':args.command_delay,
                               'jitter_seconds':args.command_jitter},
        },
        'comparisons':{
            'phase4_weights_vs_preserved_mpc':mpc_comparison(torque_rows),
            'torque_mpc_vs_lqr':paired_rows(torque_rows,'lqr','mpc'),
            'step_dir_mpc_vs_lqr':paired_rows(step_rows,'lqr','mpc'),
            'delayed_mpc_vs_lqr':paired_rows(delayed_rows,'lqr','mpc'),
        },
        'rows':rows,
    }
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in result['comparisons'].items()
                      if key=='phase4_weights_vs_preserved_mpc'},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

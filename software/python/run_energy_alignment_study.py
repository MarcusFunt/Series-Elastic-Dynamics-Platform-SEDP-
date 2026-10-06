"""Resumable CPU study of MPC-initialized PPO with original and aligned rewards."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'runs/phase5/energy_alignment_study'
OUT=DEFAULT_OUT
TRAIN=ROOT/'software/python/train_residual_v4.py'
EVAL=ROOT/'software/python/evaluate_v4.py'
SEEDS=(173,271,389)
DEV_SEEDS=(1001,1002,1003)
FINAL_SEEDS=(5001,5002,5003)
PILOT_WEIGHTS=(.25,1.,4.)
PILOT_STEPS=65536
FULL_STEPS=1048576
CHECKPOINT_STEPS=(262144,524288,1048576)
# Median LQR per-step mJ*s in the three fixed and three random development cases.
ENERGY_STEP_SCALE_MJS=.0025861015040505603
ENERGY_ONLY_REASON='less than 5% mean energy improvement over paired LQR'
PPO_LEARNING_RATE=3e-4
ANCHOR_KL_COEF=.001
MPC_HORIZON=8
MPC_TEACHER_SAMPLES=2400
MPC_TEACHER_EPISODES=8
MPC_TEACHER_EPOCHS=30


def select_weight(observations):
    """Prioritize development safety/tracking compatibility, then energy."""
    def rank(weight):
        decisions=observations[weight]
        violations=sum(sum(ENERGY_ONLY_REASON not in reason for reason in d['reasons'])
                       for d in decisions)
        ratio=float(np.mean([d['mean_energy_ratio'] for d in decisions]))
        return violations,ratio,weight
    return min(observations,key=rank)


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def status(phase,**details):
    write_json(OUT/'status.json',{'phase':phase,'updated_utc':stamp(),**details})
    print(f'{stamp()} {phase} {json.dumps(details)}',flush=True)


def run(command,log_path,expected):
    if expected.exists():
        return
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w',encoding='utf-8') as output:
        result=subprocess.run([sys.executable,'-u',*map(str,command)],cwd=ROOT,
                              stdout=output,stderr=subprocess.STDOUT,check=False)
    if result.returncode or not expected.exists():
        raise RuntimeError(f'command failed ({result.returncode}): {command}; see {log_path}')


def train_teacher(seed):
    directory=OUT/'teacher'/f'seed{seed}'
    target=directory/'policy_candidate.pt'
    status('training_mpc_teacher',directory=str(directory),seed=seed,
           samples=MPC_TEACHER_SAMPLES,episodes=MPC_TEACHER_EPISODES,
           horizon=MPC_HORIZON)
    run([TRAIN,'teacher','--samples',MPC_TEACHER_SAMPLES,
         '--episodes',MPC_TEACHER_EPISODES,'--epochs',MPC_TEACHER_EPOCHS,
         '--horizon',MPC_HORIZON,'--seed',seed,'--history',8,
         '--outdir',directory,'--skip-evaluation'],directory/'teacher.log',target)
    return target


def train(directory,seed,steps,weight,teacher_checkpoint):
    target=directory/'policy_candidate.pt'
    status('training',directory=str(directory),seed=seed,steps=steps,weight=weight)
    run([TRAIN,'ppo','--steps',steps,'--envs',8,'--rollout',256,
         '--learning-rate',PPO_LEARNING_RATE,'--anchor-kl',ANCHOR_KL_COEF,'--history',8,
         '--init',teacher_checkpoint,
         '--seed',seed,'--energy-gate-weight',weight,
         '--energy-gate-step-scale-mjs',ENERGY_STEP_SCALE_MJS,
         '--outdir',directory,'--skip-evaluation'],directory/'train.log',target)
    return target


def evaluate_checkpoint(checkpoint,seeds,output):
    status('evaluating',checkpoint=str(checkpoint),seeds=seeds,output=str(output))
    run([EVAL,'--model',checkpoint,'--seeds',*seeds,'--json',output],
        output.with_suffix('.log'),output)
    return json.loads(output.read_text(encoding='utf-8'))


def evaluate_mpc(seeds,output):
    status('evaluating_mpc',seeds=seeds,output=str(output),horizon=MPC_HORIZON)
    run([EVAL,'--mpc','--horizon',MPC_HORIZON,'--seeds',*seeds,'--json',output],
        output.with_suffix('.log'),output)
    return json.loads(output.read_text(encoding='utf-8'))


def checkpoint_rank(decision):
    return (sum(ENERGY_ONLY_REASON not in reason for reason in decision['reasons']),
            decision['mean_energy_ratio'])


def paired_cases(rows,mpc_rows):
    key=lambda r:(r['scenario'],r['seed'],r['randomized'])
    base={key(r):r for r in rows if r['controller']=='lqr'}
    policy={key(r):r for r in rows if r['controller']=='policy'}
    mpc={key(r):r for r in mpc_rows if r['controller']=='mpc'}
    metrics=('energy_integral_mJs','position_rmse_mm','peak_angle_deg',
             'peak_rail_fraction','saturation_fraction','stop_contact_fraction',
             'terminated','samples')
    return [{'scenario':k[0],'seed':k[1],'randomized':k[2],
             'energy_ratio':policy[k]['energy_integral_mJs']/base[k]['energy_integral_mJs'],
             'energy_ratio_to_mpc':policy[k]['energy_integral_mJs']/mpc[k]['energy_integral_mJs'],
             'lqr':{m:base[k][m] for m in metrics},
             'mpc':{m:mpc[k][m] for m in metrics},
             'policy':{m:policy[k][m] for m in metrics}}
            for k in sorted(base,key=str) if k in policy and k in mpc]


def diagnostics(training):
    logs=json.loads((training/'training.json').read_text(encoding='utf-8'))
    tail=logs[-max(1,len(logs)//10):]
    fields=('mean_reward','explained_variance','kl','clipfrac','action_mean_abs',
            'action_spread','action_sample_std','executed_residual_accel_mean_abs',
            'projection_fraction','reward_reconstruction_error')
    return {k:float(np.mean([r[k] for r in tail])) for k in fields}


def create_report(protocol,selected):
    results=[]
    for condition in ('baseline','aligned'):
        for seed in SEEDS:
            directory=OUT/'full'/f'{condition}_seed{seed}'
            final=json.loads((directory/'evaluation_final.json').read_text(encoding='utf-8'))
            dev=json.loads((directory/'evaluation_dev_selected.json').read_text(encoding='utf-8'))
            mpc_final=json.loads((OUT/'mpc/evaluation_final.json').read_text(encoding='utf-8'))
            results.append({'condition':condition,'train_seed':seed,
                            'checkpoint':selected[f'{condition}_{seed}']['checkpoint'],
                            'development_promotion':dev['promotion'],
                            'final_promotion':final['promotion'],
                            'paired_cases':paired_cases(final['rows'],mpc_final['rows']),
                            'ppo_diagnostics_last_10pct':diagnostics(directory)})
    report={'protocol':protocol,'results':results}
    write_json(OUT/'comparison.json',report)
    lines=['# Energy-aligned PPO comparison','',
           f'Energy weight: {protocol["selected_energy_weight"]}; '
           f'normalization: {ENERGY_STEP_SCALE_MJS:.12g} mJ*s per step.',
           'Every PPO run was initialized from a per-seed distilled repository ConstrainedMPC teacher. '
           'The report pairs policy energy against both LQR and ConstrainedMPC.',
           'Development random seeds: '+str(DEV_SEEDS)+'. Final random seeds: '+str(FINAL_SEEDS)+'.',
           'Each full run used 1,048,576 CPU environment steps. Checkpoints were selected on development cases only.',
           'Fixed benchmark scenarios are shared by development and final evaluations; random seeds are separate.','',
           '| Condition | Train seed | Final mean energy ratio | Final gate | Dev ratio |',
           '|---|---:|---:|---|---:|']
    for r in results:
        final=r['final_promotion'];dev=r['development_promotion']
        lines.append(f'| {r["condition"]} | {r["train_seed"]} | '
                     f'{final["mean_energy_ratio"]:.4f} | '
                     f'{"PASS" if final["accepted"] else "FAIL"} | '
                     f'{dev["mean_energy_ratio"]:.4f} |')
    lines.extend(['','Promotion requires every existing energy, safety, and tracking gate to pass.',
                  'No candidate is promoted from development results or a failed final gate.',
                  'Per-case metrics, reasons, and PPO diagnostics are in comparison.json.',''])
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT,
                        help='directory for this run\'s checkpoints, logs, and reports')
    OUT=parser.parse_args().out_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    code_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (TRAIN,EVAL,ROOT/'software/python/rig_rl_env_v4.py',
                ROOT/'software/python/ppo_agent_v2.py',
                ROOT/'software/python/constrained_mpc.py',Path(__file__).resolve())}
    protocol={'started_utc':stamp(),'training_seeds':SEEDS,
              'development_random_seeds':DEV_SEEDS,'final_random_seeds':FINAL_SEEDS,
              'pilot_weights':PILOT_WEIGHTS,'pilot_steps':PILOT_STEPS,
              'full_steps':FULL_STEPS,'checkpoint_steps':CHECKPOINT_STEPS,
              'energy_step_scale_mJs':ENERGY_STEP_SCALE_MJS,
              'learning_rate':PPO_LEARNING_RATE,
              'anchor_kl_coef':ANCHOR_KL_COEF,
              'initialization':'per-training-seed ConstrainedMPC teacher distilled from repository implementation',
              'mpc_teacher':{'samples':MPC_TEACHER_SAMPLES,'episodes':MPC_TEACHER_EPISODES,
                             'distillation_epochs':MPC_TEACHER_EPOCHS,'horizon':MPC_HORIZON},
              'mpc_evaluation_horizon':MPC_HORIZON,
              'normalization_source':'LQR rows in runs/phase5/ppo_v5_cpu_long_seed173/evaluation.json',
              'selection_rule':'Fewest non-energy promotion-gate reasons, then lowest mean paired energy ratio',
              'code_sha256':code_hash,'device':'cpu','actuator_mode':'torque'}
    write_json(OUT/'protocol.json',protocol)
    teachers={seed:train_teacher(seed) for seed in SEEDS}
    mpc_dev=evaluate_mpc(DEV_SEEDS,OUT/'mpc/evaluation_dev.json')
    observations={}
    for weight in PILOT_WEIGHTS:
        decisions=[]
        for seed in SEEDS:
            directory=OUT/'pilot'/f'weight{weight:g}_seed{seed}'
            checkpoint=train(directory,seed,PILOT_STEPS,weight,teachers[seed])
            evaluation=evaluate_checkpoint(checkpoint,DEV_SEEDS,directory/'evaluation_dev.json')
            decisions.append(evaluation['promotion'])
        observations[weight]=decisions
    weight=select_weight(observations)
    protocol['selected_energy_weight']=weight
    protocol['pilot_decisions']={str(k):v for k,v in observations.items()}
    protocol['frozen_utc']=stamp()
    write_json(OUT/'protocol.json',protocol)
    status('weight_frozen',weight=weight)
    selected={}
    for condition,condition_weight in (('baseline',0.),('aligned',weight)):
        for seed in SEEDS:
            directory=OUT/'full'/f'{condition}_seed{seed}'
            train(directory,seed,FULL_STEPS,condition_weight,teachers[seed])
            dev_results=[]
            for step in CHECKPOINT_STEPS:
                checkpoint=directory/f'policy_{step:08d}.pt'
                evaluation=evaluate_checkpoint(
                    checkpoint,DEV_SEEDS,directory/f'evaluation_dev_{step}.json')
                dev_results.append((checkpoint,evaluation['promotion']))
            checkpoint,decision=min(dev_results,key=lambda item:checkpoint_rank(item[1]))
            selected[f'{condition}_{seed}']={'checkpoint':str(checkpoint),
                'development_decision':decision}
            write_json(OUT/'selected_checkpoints.json',selected)
            source=directory/f'evaluation_dev_{int(checkpoint.stem.split("_")[-1])}.json'
            (directory/'evaluation_dev_selected.json').write_bytes(source.read_bytes())
    status('candidates_frozen',checkpoints=selected)
    evaluate_mpc(FINAL_SEEDS,OUT/'mpc/evaluation_final.json')
    for condition in ('baseline','aligned'):
        for seed in SEEDS:
            checkpoint=Path(selected[f'{condition}_{seed}']['checkpoint'])
            directory=checkpoint.parent
            evaluate_checkpoint(checkpoint,FINAL_SEEDS,directory/'evaluation_final.json')
    create_report(protocol,selected)
    status('complete',report=str(OUT/'comparison.md'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error),traceback=traceback.format_exc())
        raise

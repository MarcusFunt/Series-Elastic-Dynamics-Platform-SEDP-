#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from active_vibration_rig_2d import PlantParams,ControllerParams
from benchmark_suite import benchmark,print_table,summarize
from rig_rl_policy_v3 import PPOPolicyAdapterV3

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,default=Path('../../models/ppo_v3/policy_accepted.pt'));args=ap.parse_args()
    pol=PPOPolicyAdapterV3(args.model,PlantParams(),ControllerParams(),.0005)
    rows=benchmark(['ppo'],ppo_policy=pol);print_table(rows);print('\nsummary:',summarize(rows))
    return 0 if summarize(rows).get('ppo',False) else 2
if __name__=='__main__':raise SystemExit(main())

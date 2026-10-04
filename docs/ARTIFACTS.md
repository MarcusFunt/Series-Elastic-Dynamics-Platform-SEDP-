# Generated artifacts and reproducibility

SEDP produces several binary artifacts during experiments:

- trained PyTorch policy checkpoints (`.pt`)
- rendered MP4 comparisons
- PNG screenshots/previews
- CSV training/evaluation logs
- OpenModelica result files

These files are intentionally **not part of the source bootstrap commit**. Git history is a poor place for repeatedly regenerated model checkpoints and videos; future published checkpoints should use **Git LFS or GitHub Releases** with the training configuration and benchmark report attached.

## Artifacts produced during the initial research pass

The development work that led to this repository produced:

- an LQR-imitation warm-start PPO checkpoint;
- a ~205k-step long-run PPO experiment;
- a v2 residual-PPO reversal-specialized checkpoint;
- browser-rendered carriage/spring previews;
- servo-vs-active-damping video;
- servo-vs-PPO comparison video.

The source code, formulations, benchmark results, and training workflow are committed here so those artifacts can be reproduced.

## Reproducing a v2 checkpoint

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd software/python

python pretrain_randomized_lqr_v2.py \
  --samples 30000 \
  --epochs 10 \
  --out ../../models/ppo_v2_anchor.pt

python train_stage_v2.py \
  --init ../../models/ppo_v2_anchor.pt \
  --anchor ../../models/ppo_v2_anchor.pt \
  --stage reversal \
  --steps 40000 \
  --out ../../models/ppo_v2_reversal.pt
```

Then run:

```bash
python evaluate_v2.py
python evaluate_randomized_v2.py
```

## Artifact publication policy

When a checkpoint is worth preserving:

1. keep the exact source commit SHA;
2. store the checkpoint in a GitHub Release or Git LFS;
3. attach the environment and PPO configuration;
4. attach deterministic and randomized benchmark CSVs;
5. document which trajectory family it was selected on.

This prevents the repository from presenting a narrowly optimized checkpoint as a universal controller.

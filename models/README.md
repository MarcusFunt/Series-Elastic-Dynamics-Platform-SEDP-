# Trained policy artifacts

Policy checkpoints are generated experiment artifacts and are intentionally not committed to normal Git history.

The current accepted v3 artifact is named `policy_accepted.pt`. It is a small residual-acceleration policy on top of constrained LQR, and it passed SEDP-B1 in the validated run recorded in `docs/benchmark_B1_fixed.json`.

To recreate it:

```bash
cd software/python
python pretrain_v3_damping.py --out ../../models/ppo_v3/heuristic_warmstart.pt
python train_ppo_v3.py --init ../../models/ppo_v3/heuristic_warmstart.pt --steps 8192 --outdir ../../models/ppo_v3
```

Stable binary checkpoints should be published through a GitHub Release or Git LFS together with the source commit and benchmark output.

The v2/legacy policies are historical only and must not be used for headline comparisons.

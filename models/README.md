# Trained policy artifacts

This directory is reserved for local/generated policy checkpoints.

The bootstrap repository intentionally does **not** commit binary `.pt` files. Train them locally using the scripts in `software/python/`, or publish stable checkpoints through Git LFS / GitHub Releases.

Historically useful artifacts from the research pass included:

- `ppo_lqr_warmstart.pt` — behavior-cloned LQR warm start;
- `ppo_longrun_best_nominal.pt` — best nominal-step checkpoint from the earlier long PPO run;
- a v2 reversal-specialized residual PPO policy;
- later robust-stage v2 checkpoints retained for comparison rather than assumed superior.

See `docs/RL_FORMULATION_V2.md`, `archive/rl_v1/LONG_RUN_RESULTS.md`, and `docs/ARTIFACTS.md`.

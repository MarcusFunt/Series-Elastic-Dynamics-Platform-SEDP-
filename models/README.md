# Trained policy artifacts

## Current policy

`ppo_v3/policy_accepted.pt` is the current benchmark-gated residual-acceleration policy. It is a small learned correction **on top of constrained LQR**, not an end-to-end torque controller.

Promotion rule: a policy must pass the immutable SEDP-B1 tracking, rail, saturation, stop-contact, peak-angle and RMS-angle gates. Reward alone is never sufficient.

See `docs/RL_FORMULATION_V3.md` and `docs/benchmark_B1_fixed.json`.

## Legacy

The `ppo_v2/` and `legacy/` artifacts are historical only. They must not be used for headline comparisons: the original full-state LQR / PPO v2 showcase exposed unacceptable aggressive-reversal behavior and a non-reproducible benchmark claim.

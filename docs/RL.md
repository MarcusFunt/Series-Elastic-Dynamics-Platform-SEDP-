# Learned control

The current path is **residual PPO**, not end-to-end motor control.

```text
tau_cmd = tau_servo + residual_policy(observation)
```

The deterministic servo handles trajectory tracking; the policy learns damping corrections and model-mismatch compensation.

## v2 improvements

- energy-removal shaping
- transition/settling rewards
- predicted stopping-distance rail barrier
- state-dependent action limiting near rail ends
- curriculum training
- domain randomization
- imitation/LQR warm start
- KL anchoring toward the stable prior
- left/right symmetry regularization
- bounded Beta-distribution actions

Selected policy:

`models/ppo_v2/ppo_v2_reversal2.pt`

It is selected for aggressive reversals rather than simple step settling. A future universal controller should validate every accepted checkpoint across step, reversal, sine, chirp, and randomized-plant tasks.

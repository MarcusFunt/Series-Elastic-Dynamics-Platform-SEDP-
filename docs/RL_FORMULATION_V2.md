# RL formulation v2 — active vibration rig

## Why v1 drifted

The first long PPO runs optimized a generic weighted state cost. They could improve one trajectory while sacrificing rail margin or another trajectory, and unconstrained fine-tuning could erase the useful LQR-imitation prior.

## Changes in v2

1. **Residual policy** remains: the learned action augments the deterministic carriage servo rather than replacing it.
2. **Potential-based energy shaping** rewards actual removal of resonator energy: `r_E = w_E * (E_prev - gamma * E_next)`.
3. **Settling/event rewards** reward ending each commanded move near zero angle/rate.
4. **Stopping-distance rail barrier** penalizes predicted rail occupancy before an end stop is reached.
5. **State-dependent residual scaling** attenuates outward residual torque near the rail boundary.
6. **Curriculum**: long-hold settling -> repeated reversals -> randomized robust operation.
7. **Domain randomization**: payload, spring stiffness/damping, belt stiffness, rail damping, motor torque, kicks, and observation noise.
8. **KL anchor to the imitation policy** reduces catastrophic forgetting of the known-stable controller.
9. **Left/right symmetry regularization** exploits approximate mechanical mirror symmetry: `pi(-s) ~= -pi(s)`.
10. **Bounded Beta policy** keeps residual actions intrinsically inside [-1,1].

## Chosen checkpoint

`models/ppo_v2/ppo_v2_reversal2.pt`

This checkpoint was selected before the hardest robust stage because it gave the best aggressive-reversal benchmark.

### Aggressive reversal, nominal plant

- Servo peak angle: ~19.7 deg
- Servo angle RMS: ~9.18 deg
- PPO peak angle: ~9.33 deg
- PPO angle RMS: ~3.01 deg
- Servo peak rail fraction: ~0.583
- PPO peak rail fraction: ~0.495

### Randomized reversal evaluation

Across 25 randomized episodes, the trained policy improved mean episode reward, mean absolute angle, mean peak angle, and peak rail use relative to the wrapped warm-start policy. Mean position error increased slightly, which remains an explicit tradeoff.

## Known limitation

The reversal-specialized policy is not the best step-settling policy. A future universal controller should use scenario-conditioned training or accept checkpoints only after validation on step, reversal, sine, chirp, and randomized plants.

## Research basis

- Johannink et al., *Residual Reinforcement Learning for Robot Control* (ICRA 2019 / arXiv 1812.03201)
- Cheng et al., *End-to-End Safe Reinforcement Learning through Barrier Functions* (AAAI 2019)
- imitation-learning + DRL active vibration control work in Engineering Structures (2026)
- safe RL using control-barrier / residual architectures

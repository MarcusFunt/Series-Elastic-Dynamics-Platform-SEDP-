# Random goal hold PPO result — 2026-10-06

## Task

The controller moves to randomized positions and holds each target. Episodes last 12 seconds. Each target uses a one-second minimum-jerk move followed by a hold of at least 2.5 seconds; the environment does not announce a new target unless its full hold fits in the episode. Three complete goals are evaluated per episode.

The assembly rail half-travel is 65 mm. The requested 15 mm exclusion band begins at ±50 mm. Targets are sampled within ±48 mm, giving a 2 mm control buffer, and the rail control barrier keeps the carriage out of the exclusion band.

## Training

- PPO fine-tuning: 65,536 steps, 8 environments, seed 1995.
- Initialized from [initial_checkpoint.pt](initial_checkpoint.pt).
- Learning rate 1e-4; anchor KL 0.001; energy gate weight 64, active only during goal holds.
- Randomized plant parameters, sensor noise, initial angle, and kick disturbances are enabled by the saved environment configuration.
- Code revision: `bf3e9239b1f2b7d56a6c830d20436e53288e73e0`.
- Final checkpoint SHA-256: `7A2757C5455686F533FABF1DA2D0D12E969087C830A01DB3048ABCF80F801C8C`.

## Paired confirmation

Eight independent seeds (9001–9008), with LQR and the repository MPC run on the same goal sequences and disturbances. A successful hold means the carriage remains within 5 mm of the target for at least two continuous seconds during its commanded dwell.

| Metric | PPO | LQR | MPC |
| --- | ---: | ---: | ---: |
| Integrated resonator energy (mJ·s) | 0.07363 | 0.07909 | 0.05809 |
| Position RMSE (mm) | 0.7638 | 0.9013 | 0.1544 |
| Angle RMS (deg) | 0.3154 | 0.3268 | 0.2811 |
| Peak angle (deg), mean across seeds | 1.3472 | 1.4501 | 1.0826 |

The policy reduced integrated resonator energy by **7.1% vs LQR** and improved tracking RMSE by **15.3% vs LQR**. It passed the paired peak-angle, tracking, safety, and 5% energy gates on both development seeds 8011–8015 and the independent confirmation seeds.

All 24 of 24 confirmation goals completed the two-second-in-tolerance hold. The policy had zero exclusion-zone incursions and zero terminations. The repo MPC also completed every hold and avoided the exclusion zone; it had lower integrated resonator energy and tracking RMSE than PPO on this task, so PPO is promoted against LQR only.

The reported energy is the time integral of the simulator's modeled resonator kinetic-plus-spring energy. It is a vibration-energy metric, not motor electrical consumption or total actuator energy.

## Artifacts

- [Policy checkpoint](policy_candidate.pt)
- [Development evaluation](evaluation_dev.json)
- [Independent confirmation evaluation](evaluation_confirmation.json)
- [Training metadata](run_metadata.json)
- [Per-update training metrics](training.json)

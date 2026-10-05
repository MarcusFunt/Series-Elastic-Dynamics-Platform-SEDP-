# V4 validation — 4 October 2026

The corrected training pipeline, estimated-state policy and MPC teacher work end to end. No candidate is accepted as a replacement controller. This is a formulation/architecture implementation with preliminary training evidence, not a claim of improved overall control.

## Checks

- 20 regression tests pass, including timeout bootstrapping, reflection, actor/critic separation, encoder wrapping, nominal-parameter isolation, checkpoint compatibility, runtime/training parity, MPC infeasibility fallback and promotion checks.
- Python compilation, the original core smoke test and unchanged archived SEDP-B1 LQR gates pass.
- The browser starts with a v4 candidate checkpoint and returns live `/api/state`.
- Final comparisons use all three deterministic trajectories and held-out randomized seeds 101/202/303. Every displayed candidate completes the episodes without termination, stop contact or torque saturation.

## Training actually run

Collected 1,024 teacher transitions across four randomized episodes, using a 40-step MPC horizon. 625 were successful solver/nonlinear-check labels; 399 fallback labels were excluded. Held-out validation uses complete episodes. A 40-epoch re-fit reaches normalized-action MAE 0.1438. The separate PPO test starts from the 30-epoch teacher candidate and runs 4,096 environment steps with four environments; it does not start from the later 40-epoch re-fit.

Only one training seed (173) was run. Three randomized evaluation seeds are not three independent training runs.

## Paired policy results

| Candidate | Mean per-case energy change vs estimated LQR | Accepted |
|---|---:|---|
| teacher_40_epochs | -8.16% | No |
| ppo_4096 | -4.58% | No |

The 40-epoch distilled policy improves step and square-stress vibration while worsening step tracking and aggressive/randomized peaks. Its step tracking RMSE exceeds the unchanged 8 mm cap. The shorter PPO check does not resolve these tradeoffs.

| Scenario | Estimated LQR RMS angle | Distilled RMS angle | LQR tracking RMSE | Distilled tracking RMSE |
|---|---:|---:|---:|---:|
| step | 0.427° | 0.240° | 6.30 mm | 8.55 mm |
| aggressive | 1.873° | 2.035° | 5.44 mm | 4.99 mm |
| square_stress | 1.815° | 1.478° | 29.22 mm | 28.75 mm |
| random_reference (101) | 2.089° | 2.163° | 12.49 mm | 13.68 mm |
| random_reference (202) | 1.893° | 2.031° | 14.63 mm | 16.33 mm |
| random_reference (303) | 2.318° | 2.359° | 13.73 mm | 15.25 mm |

## MPC reference result

The locally linearized MPC is currently a weak reference for aggressive reversals and hard square commands. It has frequent solver/nonlinear-feasibility fallback and does not uniformly beat LQR. Reported MPC behavior includes its explicit fallback to LQR. Its successful commands are usable teacher data, but the solver is not an established superior expert.

| Scenario | Fallback fraction | Median solve time | 95th percentile solve time |
|---|---:|---:|---:|
| step | 14.0% | 25.1 ms | 47.6 ms |
| aggressive | 54.5% | 33.9 ms | 65.6 ms |
| square_stress | 96.0% | 58.8 ms | 91.7 ms |
| random_reference (101) | 34.0% | 29.8 ms | 80.5 ms |
| random_reference (202) | 36.5% | 30.0 ms | 83.1 ms |
| random_reference (303) | 39.7% | 30.3 ms | 79.7 ms |

These pre-optimization timings exceed the 10 ms controller period on this
execution machine. They do not measure the later CPU-optimized implementation;
current default latency relative to the controller period remains unverified.
The simulation is offline and does not inject missed-deadline delay, so these
measurements do not establish real-time feasibility on the user’s PC or RP2350.

## Decision

Keep the accepted v3 default. Retain v4 as the reproducible experimental path. Before a large new training run, improve MPC prediction/feasibility and its vibration/tracking objective, train across more independent seeds, then test history/preview/estimator ablations. Hardware sensing, homing and timing still need actual calibration.

## Evidence

- [MPC reference](evidence/mpc_reference.json)
- [Distilled candidate evaluation](evidence/teacher_40_epochs.json)
- [PPO candidate evaluation](evidence/ppo_4096.json)
- [Teacher fit](evidence/teacher_fit.json)
- [PPO optimizer diagnostics](evidence/ppo_training.json)
- [Runtime versions and source hashes](evidence/source_manifest.json)

Commands and implementation details are in [RESIDUAL_V4.md](RESIDUAL_V4.md).

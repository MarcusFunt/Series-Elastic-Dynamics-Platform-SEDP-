# Phase 5 PPO Experiment Plan

## Question

Measure the effect of estimator information, observation history, and reference preview on residual PPO control, using the Phase 1–4 torque-mode plant and the same training/evaluation budgets for every configuration.

## Predeclared Matrix

This is a one-factor-at-a-time ablation. The baseline is estimated-state control with eight observation frames and reference preview enabled. Each other configuration changes exactly one factor:

| Configuration | State input | History frames | Reference preview |
|---|---|---:|---|
| Baseline | Estimated | 8 | Enabled |
| Oracle upper bound | True state | 8 | Enabled |
| Short history | Estimated | 1 | Enabled |
| No preview | Estimated | 8 | Disabled |

The oracle-state case is an information upper bound only and is not deployable. Preview-disabled observations retain the same 26-feature frame shape, with the nine preview features set to zero.

## Fixed Budgets

- Training seeds: `173`, `271`, `389`.
- Shared held-out random-reference evaluation seeds: `1001`, `1002`, `1003`.
- Training: `16,384` environment transitions per configuration and training seed, 8 environments, rollout length 256, 3 PPO epochs, learning rate `3e-5`, anchor-KL coefficient `0.005`, one Torch CPU thread.
- There is no v4 teacher checkpoint in the checkout. Each run starts from the same seeded actor-critic initialization scheme, with the constrained LQR residual path as the zero-action behavior. No teacher data is collected in this ablation.
- Evaluation: all fixed scenarios in `benchmark_suite.SCENARIOS` plus randomized references using the same three held-out seeds, paired against LQR.
- The matrix contains 12 independent training runs. Checkpoints, per-run stdout, training logs, run metadata, and raw evaluation rows are retained under the ignored `runs/phase5/ppo_v5` directory. A compact reproducibility/evidence JSON and report are also written under `docs/control/`.

The training budget was selected after profiling. A paired, alternating-order benchmark with 4 environments, 100 vector steps, and three repeats measured 123.9 transitions/s median for the scalar EKF and 194.4 transitions/s median for the batched EKF. The median of the three paired speed ratios was 1.57× (individual pairs: 1.57×, 1.62×, and 1.51×), a consistent gain beyond the observed run-to-run spread. The workload profile attributed most scalar-path time to recomputing the finite-difference EKF Jacobian through RK4. The PPO matrix budget is therefore 196,608 training transitions total, before evaluation.

## Measures and Promotion

For each trained seed, report tracking RMSE, angle RMS and peak angle, energy integral, rail and speed violations, saturation, completion, and estimator position error. Report means and sample standard deviations across training seeds, and preserve each case's raw paired result. Training seeds remain distinct from evaluation seeds.

Use the existing `evaluate_v4.promotion` gate for each candidate. No candidate is selected from training reward alone; oracle-state policies are never eligible for promotion. The report must state any per-case safety/performance regression and distinguish inconclusive low-budget comparisons from evidence of no effect.

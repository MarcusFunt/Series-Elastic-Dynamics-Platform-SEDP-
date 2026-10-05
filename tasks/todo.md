# Simulation Fidelity and Control Experiments — Checklist

## Phase 1: Physics Foundation

- [x] 1.1 Add Python two-spring geometry mode aligned with OpenModelica.
- [x] 1.2 Validate restoring torque and local tangent stiffness; document symmetry axes and defer torsion/geometric calibration pending physical measurements.
- [x] 1.3 Define total mechanical energy and verify conservative/dissipative energy behavior.
- [x] 1.4 Measure RK4 timestep convergence and document supported timestep range.
- [x] Review physics foundation results before controller comparisons.

## Phase 2: STEP/DIR Actuator

- [x] 2.1 Add selectable command-space actuator with discrete steps, limits, tracking lag, and diagnostics.
- [x] Verify zero-impairment torque-mode compatibility and actuator behavior.

## Phase 3: Timing and Sensing

- [x] 3.1 Add independent sensor rates, delay, jitter, and dropouts with acquisition/arrival timestamps.
- [x] 3.2 Add command delay and jitter with explicit pending-command semantics.
- [x] Verify synchronous and zero-delay parity.

## Phase 4: Constrained MPC

- [x] 4.1 Categorize infeasibility, numerical failure, timeout/iteration limit, nonlinear rejection, and success.
- [x] 4.2 Align MPC prediction with the active actuator/timing model and test prediction/rollout consistency.
- [x] 4.3 Tune tracking/vibration objective and compare per-case metrics and failure rates; reject the trial weights on declared paired bounds.
- [x] Review control-model results before teacher-data collection.

## Phase 5: PPO Experiments

- [ ] 5.1 Profile and optimize the dominant training cost while preserving seeded transition parity.
- [ ] 5.2 Predeclare ablation factors and budgets; compare estimator/history/preview across multiple independent training seeds.
- [ ] 5.3 Produce a reproducible report with shared held-out evaluations and uncertainty/spread.

## Final Verification

- [x] Run existing tests before committing implementation changes (full suite executed; see known environment/test-run issues below).
- [x] Confirm no dependencies, project configuration, or CI files changed without prior approval.

Full-suite result: 68 tests ran; the Phase 4, control, timing, physics, and STEP/DIR tests passed. Two training-web tests did not pass in this Windows run: the symlink fixture requires a privilege unavailable to the process (`WinError 1314`), and the real PPO job did not finish within the test's wait window. No repository configuration or dependency changes were made.

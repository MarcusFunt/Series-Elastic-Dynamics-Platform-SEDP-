# Phase 4: constrained MPC

## Solver outcomes and fallback

Each action records `outcome`, `reason`, elapsed solve time, iteration count,
constraint violation, actuator mode, and whether the action is a fallback. The
categories distinguish `success`, `infeasible_initial_bounds`,
`infeasible_constraints`, `numerical_solver_error`, `solver_failure`,
`iteration_limit`, `time_limit`, `nonlinear_actuator_rejection`, and
`nonlinear_state_rejection`. Invalid state/reference shapes remain caller
errors. Model linearization and solver exceptions return a categorized result.

Every failure returns normalized residual action `0.0`, which delegates the
command to the environment's bounded LQR baseline. This fallback is repeatable
for the same state and queue; it does not perturb the live command queue or its
jitter random generator. `MPCConfig.time_limit_seconds` can bound optimization
time and produces the distinct `time_limit` outcome.

The QP is sequentially linearized around an LQR nominal trajectory. At each
stage it enforces residual authority, projected acceleration bounds, soft rail,
carriage speed, motor speed, and (in torque mode) a conservative torque
envelope. A nonlinear rollout then applies the same queued-command projection,
torque transition or STEP/DIR interval, and known disturbance schedule as the
environment. A candidate that violates exact authority or state constraints is
rejected and uses the deterministic fallback. STEP/DIR torque saturation is an
actuator behavior and is counted separately from solver failure.

For delayed or jittered commands, MPC deep-copies the live FIFO queue, issues
distinct stage markers into that copy, and maps each predicted interval to its
issuing decision or to a command already pending at the current time. The
simulation queue and random stream remain untouched. The STEP/DIR prediction
state includes the actuator's lagged motor velocity and continuous step
position; pulse quantization and torque tracking are applied on each physics
substep.

## Objective scales

The stage state cost uses squared dimensionless errors. The scale values are
physical tolerances; the weights are dimensionless multipliers.

| Error | Scale | Baseline weight | Phase 4 trial weight |
| --- | ---: | ---: | ---: |
| Carriage position | 0.006 m (6 mm) | 1.8 | 1.8 |
| Carriage speed | 0.35 m/s | 0.12 | 0.12 |
| Lever angle | 0.08727 rad (5°) | 2.4 | 3.2 |
| Lever angular rate | 1.7453 rad/s (100°/s) | 0.55 | 0.8 |
| Residual acceleration | 2.5 m/s² | 0.03 | 0.04 |
| Acceleration-command slew | 2.5 m/s² per control update | 0.06 | 0.06 |

The final predicted state receives a multiplier of 2. The baseline weights are
the previous MPC profile; the trial increases vibration penalties while making
the residual action slightly more expensive. The run uses a horizon of four
control intervals (40 ms), a 1 ms physics step, and a 10 ms controller period.

## Paired evaluation

Reproduce the saved experiment from the repository root:

```powershell
python software/python/evaluate_mpc_phase4.py `
  --json docs/control/evidence/mpc_phase4.json --horizon 4 --duration 1.6
```

It uses identical zero initial states, sensor-noise-free observations, no
domain randomization, and no disturbance torque. The three 1.6 s reference
windows cover a nominal step move, minimum-jerk reversals, and the square-wave
stress case. The preserved pre-Phase-4 MPC is compared with the tuned MPC and
LQR in torque mode with zero command latency. STEP/DIR and 15 ms command delay
with ±3 ms jitter compare the Phase 4 MPC against LQR under the same actuator
and timing configuration. The legacy implementation did not support those
modes, so it is not treated as a paired reference there.

The per-case acceptance bounds are: no more than 2% plus 0.05 mm tracking
regression, no more than 2% plus 0.01° angle RMS or peak regression, at most
0.01 increase in peak rail fraction, at most 0.02 increase in saturation or
fallback fraction, no increase in stop contacts, at most 0.005 increase in
soft-rail violation fraction, at most 0.0001 m/s increase in carriage-speed
violation, at most 0.001 rad/s increase in motor-speed violation, and no new
termination. It must improve mean angle RMS or energy by at least 2%. Every
pair is checked before those aggregates are computed.

The trial weights are **not promoted**. Across the three torque cases, mean
angle RMS fell to 0.907 of the previous MPC and integrated energy to 0.867,
but the aggressive case exceeded the tracking and rail bounds, the stress case
exceeded the rail and fallback bounds, and the step case exceeded the tracking
and fallback bounds. The machine-readable reasons and per-case metrics are in
[`mpc_phase4.json`](evidence/mpc_phase4.json).

The recorded Phase 4 implementation had torque MPC p95 solve times from
41.5 ms to 228.7 ms, STEP/DIR times from 66.4 ms to 75.8 ms, and delayed-torque
times from 47.8 ms to 137.2 ms. Those measurements predate the later CPU
optimization and do not describe the current default implementation. The
repository does not yet contain a post-optimization latency result, so current
default latency relative to the 10 ms control period remains unverified.
STEP/DIR and delayed-command rows retain their constraint violations,
saturation, categorized outcomes, and historical solve-time distributions in
the JSON.

This is a bounded, deterministic comparison, not a broad robustness or
hardware-calibration claim. The rejected tuning remains available for further
cost and feasibility work. New teacher collection should wait for review of
the per-case failure rates and solve-time costs.

# Implementation Plan: SEDP Simulation Fidelity and Control Experiments

## Overview

Establish a validated simulation foundation, then extend it to represent the planned STEP/DIR actuator and asynchronous sensing. Improve constrained MPC on that plant before spending substantial compute on controlled PPO comparisons. Keep the existing accepted v3 path reproducible while v4 and new plant modes remain explicit experimental configurations.

## Current State

- `software/python/active_vibration_rig_2d.py` contains the seven-state `RigPlant`, RK4 integrator, torque-limited first-order actuator, and equivalent rotational spring law.
- `openmodelica/ActiveVibrationRig.mo` already contains an opt-in explicit two-spring geometry (`useGeometricSprings`); `docs/MATHEMATICAL_MODEL.md` documents the convention.
- `software/python/rig_rl_env_v3.py` and `rig_rl_env_v4.py` use a resonator-only small-angle energy measure for reward/metrics. That measure is useful for control scoring, but does not validate total plant energy conservation.
- `software/python/state_estimator.py` and the v4 residual loop receive synchronous measurements at the control cadence.
- `software/python/constrained_mpc.py` has a constrained SLSQP path and one generic fallback path; `docs/control/V4_VALIDATION.md` records high fallback rates and runtimes beyond the 10 ms control interval for several cases.
- Existing focused checks are in `tests/test_formulation.py` and `tests/test_control_v4.py`; `make test-control-v4` invokes the test suite. The original planning pass was read-only; Phase 1 verification results are recorded in `docs/control/PHYSICS_VALIDATION.md`.
- The first geometric torque comparison exposed an OpenModelica sign mismatch; the generalized-force sign is now used in both implementations. With placeholder geometry values, the upright geometric model has negative effective stiffness, so its physical stability is not established.

## Architecture Decisions

- Keep the torque actuator and equivalent torsional spring available as named baseline modes so existing checkpoints and comparisons remain reproducible.
- Add physical alternatives behind explicit model parameters or configuration, with matching Python and OpenModelica conventions where both implementations represent the same mechanism.
- Treat energy used for reward as distinct from total mechanical energy used for conservation checks.
- Preserve timestamps through sensor and command queues so the estimator and controller can reason about delayed data and actuation.
- Make MPC outcome categories machine-readable and log per-category rates in addition to aggregate fallback fractions.
- Define PPO ablations and training budgets before launching runs; vary estimator, history, and preview factors in controlled comparisons across multiple independent training seeds.

## Task List

### Phase 1: Validate Spring Geometry and Numerical Physics

#### Task 1.1: Add Python two-spring geometry mode

Implement the two spring anchor/shaft kinematics and axial spring forces in the Python plant, with parameters and sign conventions aligned to OpenModelica. Preserve equivalent-torsion mode and define a neutral/preload convention explicitly.

**Acceptance criteria:**
- [x] The explicit mode computes both spring lengths, length rates, forces, and pivot torque from geometry.
- [x] With symmetric anchors, spring torque is zero and reflection symmetry holds at absolute-angle symmetry poses (`alpha=0` or `pi`). A nonzero `theta_neutral_world` does not imply zero spring torque at `theta=0`; local restoring or destabilizing behavior is assessed from total potential curvature rather than assumed.
- [x] Equivalent-torsion behavior remains selectable and unchanged by default.

**Verification:** Add focused unit checks for geometry, symmetry, finite values, and agreement of torque with the negative derivative of spring potential. Compare shared parameter cases with the OpenModelica equations or a checked reference calculation.

**Dependencies:** None.

#### Task 1.2: Validate restoring torque and local stiffness

Measure torque across the intended angular range and verify that the effective small-angle tangent stiffness, including gravity, matches the numerical torque slope around the configured reference. Determine and document calibration status for the torsional baseline against the geometric mode; when physical parameters are placeholders, defer fitting until measurements are available.

**Acceptance criteria:**
- [x] Numerical torque slope at the configured reference agrees with the reported effective stiffness within a stated tolerance; when that reference is not an equilibrium, the value is identified as tangent stiffness.
- [x] The comparison includes the current default pose and at least one nontrivial deflection on each side.
- [x] Geometry and equivalent-torsion comparison is reported over a stated angular range, not only at one point; calibration is explicitly deferred pending measured rig parameters.

**Verification:** Deterministic tests and a small reproducible comparison report; no fitted parameters are silently substituted into the default plant.

**Dependencies:** Task 1.1.

#### Task 1.3: Add total mechanical energy and conservation checks

Define total mechanical energy for the undamped, unforced plant, including motor/carriage/resonator kinetic energy and enabled elastic/gravitational potentials. Keep the existing resonator-only energy score explicitly separate.

**Acceptance criteria:**
- [x] Energy terms match the plant’s mass matrix, belt law, selected spring law, gravity convention, and soft-stop potential where active.
- [x] An undamped, unforced trajectory has bounded energy drift over a documented horizon.
- [x] Damping or externally applied work changes energy in the expected direction/sign in controlled checks.

**Verification:** Add small deterministic tests for energy accounting and report relative drift; use separate cases to isolate spring, belt, and coupled kinetic terms.

**Dependencies:** Task 1.1.

#### Task 1.4: Establish timestep convergence

Compare RK4 trajectories and energy drift at successively halved physics timesteps against a fine-step reference while holding physical initial conditions and inputs constant.

**Acceptance criteria:**
- [x] The convergence comparison names each spring mode, parameter overrides, initial state, inputs, state/output error metrics, and time horizon.
- [x] Error decreases as the timestep is reduced for representative free decay and driven motion.
- [x] A supported maximum physics timestep is documented separately for both spring modes under the validated configurations.

**Verification:** Add deterministic convergence checks or a reproducible diagnostic command that emits the tested timestep/error table.

**Dependencies:** Tasks 1.1 and 1.3.

### Checkpoint: Physics Foundation

- [x] Geometry, restoring torque, energy accounting, and timestep results are reviewed before using the new model for controller comparisons.
- [x] Existing default torque-model behavior remains reproducible.

### Phase 2: Model the STEP/DIR Actuator

#### Task 2.1: Add command-space actuator and STEP/DIR execution model

Add a selectable actuator path that accepts bounded acceleration/velocity commands and converts them to discrete steps, accounting for command limits, actuator tracking lag, and documented simplifications. Keep the torque-input plant path available for existing studies.

**Acceptance criteria:**
- [x] Commands obey configured speed and acceleration limits and produce integer pulse/step increments.
- [x] Tracking lag and saturation are observable in state/diagnostics.
- [x] Command units and conversion to motor angle/carriage motion are documented.
- [x] The model can run in v3/v4 training environments without changing the torque-mode default.

**Verification:** Tests cover step quantization, direction/reversal, limits, lag, deterministic replay, v3/v4 environment integration, configuration serialization, and torque-mode transition regression.

**Dependencies:** Phase 1 checkpoint.

### Phase 3: Model Asynchronous and Imperfect Sensors/Commands

#### Task 3.1: Add per-channel sensor timing and data loss

Represent independent sensor rates, measurement delay, jitter, and dropouts with timestamps and reproducible random seeds. Update estimator integration to process measurements according to their acquisition time and arrival time.

**Acceptance criteria:**
- [x] Each sensor channel can use its own sample period, delay, jitter, and dropout probability.
- [x] Delayed measurements are inserted at acquisition time and replayed through buffered applied commands.
- [x] With impairment parameters set to zero, the current synchronous behavior is reproducible.

**Verification:** `tests/test_async_timing.py` covers channel schedules/dropouts, delayed EKF replay, acquisition/arrival timestamps, and zero-impairment parity. The sensor configuration and sampling constraints are documented in `docs/control/ASYNC_TIMING.md`.

**Dependencies:** Phase 2 actuator interface should expose command and simulation timing.

#### Task 3.2: Add command delay and jitter

Queue controller commands through configurable latency/jitter and define behavior while commands are pending, including safe command bounds and repeat/hold semantics.

**Acceptance criteria:**
- [x] Command issue time, arrival time, applied command, and saturation are observable.
- [x] Delay and jitter are independently configurable from sensor timing.
- [x] Zero-delay mode preserves prior command application semantics.

**Verification:** `tests/test_async_timing.py` checks FIFO order, pending hold behavior, activation timestamps, and immediate zero-delay application. Existing v4 and STEP/DIR regressions cover baseline transitions.

**Dependencies:** Task 2.1.

### Phase 4: Improve Constrained MPC

#### Task 4.1: Diagnose solver outcomes and improve feasibility handling

Separate infeasible initial bounds/constraints, numerical solver errors, iteration/time limits, nonlinear rollout rejection, and successful solves. Use the same actuator and timing model as the simulation being controlled.

**Acceptance criteria:**
- [x] Every returned action has a structured outcome category and reason.
- [x] Infeasible commands are distinguishable from numerical failures in logs and evaluation artifacts.
- [x] Fallback behavior remains bounded and deterministic for each failure class.
- [x] MPC prediction and nonlinear acceptance checks model the active actuator mode and delays.

**Verification:** Add targeted tests for each outcome category, fallback bounds, and prediction/rollout consistency.

**Dependencies:** Phases 1–3.

#### Task 4.2: Balance tracking and vibration objectives

Tune/reshape MPC costs and constraints using paired trajectory evaluations on the validated plant, reporting tracking, vibration/energy, constraint violations, solve time, and categorized failure rates.

**Acceptance criteria:**
- [x] Objective weights and scaling are documented in physical or normalized units.
- [x] Evaluation covers nominal moves, reversals, and stress trajectories with identical initial conditions and actuator model.
- [x] No aggregate improvement hides a regression beyond a declared per-metric acceptance bound; the trial was rejected on per-case bounds.

**Verification:** Produce a reproducible comparison against the current MPC and LQR reference; no candidate is promoted on solver success rate alone.

**Dependencies:** Task 4.1.

### Checkpoint: Control Model

- [x] MPC results use the validated plant, actuator, and timing model.
- [x] Failure categories and per-case metrics are reviewed before collecting new teacher data.

### Phase 5: Accelerate and Compare PPO Experiments

#### Task 5.1: Profile and accelerate the training path

Measure simulation and rollout throughput first, then optimize the dominant cost without changing environment semantics or observation/reward schemas implicitly.

**Acceptance criteria:**
- [x] A reproducible baseline throughput is recorded for fixed environment count and workload.
- [x] The selected optimization improves measured throughput and preserves seeded transition parity within defined tolerances.
- [x] Training artifacts record code/config versions and wall time.

**Verification:** Run parity checks and a fixed-size throughput comparison after the optimization.

**Dependencies:** Phases 1–4.

#### Task 5.2: Run controlled multi-seed PPO ablations

Compare estimator, history, and preview configurations with matched evaluation scenarios and budgets across multiple independent training seeds.

**Acceptance criteria:**
- [x] Ablation factors and training/evaluation budgets are fixed before runs begin.
- [x] Each configuration is trained with multiple independent seeds and evaluated on shared held-out seeds.
- [x] Results include uncertainty/spread across training seeds and report tracking, vibration, constraint, and completion metrics.
- [x] Checkpoints, configurations, source revisions, and raw evaluation results are retained.

**Verification:** Generate a reproducible comparison report and apply explicit promotion criteria; distinguish evaluation seeds from training seeds.

**Outcome:** All 12 candidates completed training and shared paired evaluation. No candidate met the 5% mean energy improvement gate; the remaining metric differences are too small to distinguish the configurations at this budget. See `docs/control/PPO_PHASE5.md` and `docs/control/evidence/ppo_ablation_v5.json`.

**Dependencies:** Task 5.1 and Control Model checkpoint.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Geometric springs and the equivalent torsional law represent different nonlinear behavior | Controller results may change for model reasons rather than controller improvements | Preserve both modes and report their torque/stiffness difference over the evaluation range |
| Energy bookkeeping omits a conservative or dissipative term | Conservation tests could pass for the wrong reason | Derive the energy function from the same mass matrix and force laws; isolate terms with targeted cases |
| Discrete steps and asynchronous timing enlarge the estimator/controller state | Existing checkpoints or performance may regress | Keep the original mode as a baseline and add mode-specific parity/performance checks |
| MPC is slower than the control interval | It may be useful offline but not as a real-time controller/teacher | Record solve-time distributions and distinguish simulation utility from real-time feasibility |
| Training conclusions depend on one seed or uncontrolled budgets | PPO rankings may be noise | Use multiple independent training seeds, shared evaluation seeds, and predeclared budgets |

## Open Questions

- Which physical spring dimensions/rates should be treated as the authoritative values when Python and OpenModelica parameter sets are compared? Current OpenModelica geometry values are labeled placeholders.
- Should the first STEP/DIR comparison use an ideal pulse follower with quantization and limits, or include a calibrated speed/torque pull-out envelope? The current project notes say hardware has not yet been physically validated.
- The initial post-profile Phase 5 budget was 16,384 transitions per run across 8 environments, 3 training seeds per configuration, and 3 shared held-out randomized seeds. Results were inconclusive and no candidate passed promotion, so a larger follow-up budget is still an open decision.

## Repository Constraints

- Do not add dependencies or change project configuration/CI without asking first.
- Do not run training sweeps until the preceding validation checkpoints are complete.
- Run the existing test suite before committing any implementation changes.

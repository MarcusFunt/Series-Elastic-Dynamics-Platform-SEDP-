# Estimated-state residual control v4

This implements the approved progression: correct training first, add estimated
state and observable motion context, then use constrained MPC as a reference
and teacher. The accepted v3 policy is not replaced. MPC and new v4 checkpoints
remain experimental until paired evaluation accepts a candidate.

## Formulation fixes

- Training and corrected v3 evaluation hold the complete torque command for
  10 ms, with 1 ms physics integration. `Simulator(control_dt=...)` supports
  this schedule. Archived B1 remains reproducible with its original default
  continuous-base timing; its thresholds and committed results are unchanged.
- Reflection maps preserve rail margin, position uncertainty and observation
  age. All signed state, action and reference-history features reverse sign.
- Timeouts retain the final observation, bootstrap its value and stop the GAE
  trace before reset. Safety termination does not bootstrap.
- The adverse-tracking penalty defaults to zero. Action and slew costs use the
  effective residual after acceleration projection; requested/effective
  residuals are logged separately. Torque saturation remains a separate
  actuator/acceptance diagnostic rather than being misreported as projection.
- v4 separates actor and critic feature networks and gradient clipping, scales
  all rewards by 0.02, uses GAE lambda 0.98, and starts with Beta concentration
  40 (normalized-action standard deviation approximately 0.156).
- Legacy v3 checkpoints retain requested-previous-action semantics when that
  configuration field is absent; newly trained v3 uses effective action.

## Observation and estimator

`MeasurementModel` exposes 12-bit motor/lever encoder angles, gyro and timestamp.
An optional linear carriage encoder is explicit (`--linear-encoder`). It never
supplies simulator carriage position, actuator torque or velocity as a default
measurement. `StateEstimator` is an eight-state EKF with the seven nominal plant
states plus gyro bias. It predicts using the same held-command nonlinear plant
and updates wrapped encoder innovations with a Joseph covariance update.

The motor must be homed. Carriage position inferred through the belt model is
model-dependent, not equivalent to a direct linear encoder. Nominal parameters
are copied; randomized true plant parameters are unavailable to the controller
and estimator. Ground truth is used for reward and metrics only.

Each 26-feature frame contains tracking errors, angle/rate, carriage position
and velocity estimates, nominal LQR acceleration, motor speed and torque
estimates, reference acceleration, previous effective action, rail margin, belt
extension estimate, reference position/velocity, three reference previews,
carriage uncertainty and observation age. Eight frames give 208 actor inputs.

Reference preview at 50/100/200 ms uses announced commands only. Random reference
preview holds the last announced destination beyond its segment and neither
advances the RNG nor exposes a future random target. The policy receives no
scenario-name labels. Sensor sampling is currently synchronous at 100 Hz;
explicit delay/jitter simulation is a subsequent experiment, not implemented
by the age feature alone.

`RigRLEnvV4` is the transition used by training and evaluation.
`EstimatedResidualPolicyV4` reuses its `ResidualControlLoop` in the browser.
A regression compares runtime and environment commands/states at identical
initial conditions and timing. `--oracle-state` is diagnostic only; oracle
checkpoints cannot be loaded into the runtime or auto-promoted.

## MPC reference and teacher

`ConstrainedMPC` re-linearizes the full seven-state held-torque transition at
each sample and solves a condensed convex quadratic objective with linear
constraints using SciPy SLSQP. The default horizon is 40 samples (0.40 seconds).
Constraints cover rail travel, carriage/motor speed, acceleration, residual
authority and a conservative speed-dependent torque envelope. The first command
also obeys the shared rail acceleration bounds. A nonlinear horizon rollout
checks rail/speed/torque and residual feasibility. Solver or rollout failure
returns zero residual; shared LQR and rail projection remain active.

This is a local-model reference, not a global nonlinear optimum or a hardware
safety guarantee. It is currently too slow and falls back too frequently to
recommend as the deployed 100 Hz controller. Timing/fallback rates are included
in results rather than hidden behind successful LQR behavior.

Teacher runs save observations, action labels, episode IDs, solver success,
solver diagnostics and configuration. Fallback labels are excluded. Fitting
splits entire episodes into training/validation and keeps the checkpoint with
lowest held-out action MAE. `--dataset` re-fits the same runs without collecting
new data. PPO can then initialize from the distilled candidate.

## Running from repository root

Install `requirements.txt`, including PyTorch, then:

```bash
make test-control-v4
python software/python/evaluate_v4.py --mpc --json runs/v4_reference/evaluation.json
python software/python/train_residual_v4.py teacher --samples 2400 --episodes 8 --epochs 40 --outdir runs/v4_teacher
python software/python/train_residual_v4.py ppo --init runs/v4_teacher/policy_candidate.pt --steps 65536 --outdir runs/v4_ppo
python software/python/evaluate_v4.py --model runs/v4_ppo/policy_candidate.pt --json runs/v4_ppo/recheck.json
python software/python/active_vibration_rig_web.py --ppo-v4-model runs/v4_ppo/policy_candidate.pt --controller ppo
```

The browser uses the v4 checkpoint's physics and controller periods. This is an
explicit experiment opt-in, not the default accepted-policy path. A longer PPO
run is provided as a command; this implementation pass only ran a 4,096-step
end-to-end PPO check. The original 524k experiment/checkpoint was not available
for re-evaluation beyond the supplied report.

## Promotion and experiment integrity

`SEDP-V4-100HZ` is separate from archived B1 because timing and sensing changed.
Evaluation pairs every candidate with zero-residual estimated-state LQR under
identical references, random parameters, disturbances and sensor noise. Defaults
include all three deterministic B1 trajectories and held-out random seeds
101/202/303. Rail maxima and saturation are checked at physics substeps.

Promotion requires complete finite paired metrics, rail <= 0.78, zero stop
contact/termination, saturation <= 8%, unchanged deterministic tracking caps
8/6/32 mm, no material peak/tracking regressions against paired LQR, and at least
5% mean per-case vibration-energy reduction. Random-reference tracking limits
are relative to the paired baseline because discontinuous random commands are
not the deterministic B1 trajectories. Rejected candidates are saved as
`policy_candidate.pt`, never as `policy_accepted.pt`.

See `V4_VALIDATION.md` and its JSON evidence for actual results. The next work
should improve the teacher's local prediction/feasibility and task balance,
then expand training/evaluation seeds. These experiments do not establish that
history or MPC alone improves overall control.

## Browser training workspace

Run `make browser`, open http://127.0.0.1:8765 and select **Training**.
The workspace starts the same v4 scripts in a separate Python process. One job
runs at a time; the simulator remains available. PPO settings include step
budget, environment count, rollout length, learning rate, anchor KL, seed,
history and the optional carriage encoder. Choose a previous candidate or
accepted checkpoint to initialize PPO; its stored history/sensor configuration
overrides those two form settings.

The workflows are PPO, MPC teacher plus distillation, checkpoint evaluation,
and MPC performance reference. Paired evaluation after training defaults on.
Turning it off produces an unevaluated candidate, never an accepted checkpoint.
A completed run shows metrics, evaluation gates, paired result rows, logs and
downloadable artifacts. **Test candidate in simulator** explicitly loads an
experimental candidate and resets the simulation with its checkpoint timing;
**Test accepted policy** loads the separately accepted artifact when present.

Jobs, configurations, logs, datasets, checkpoints and evaluations are saved in
`runs/web_training/<run-id>/`. Use `--training-dir PATH` to choose another
location. Closing a browser tab leaves training running. Cancelling a job or
stopping the server terminates its process; an uncleanly stopped run is marked
interrupted on the next server startup. Partial checkpoints remain downloadable.
The GUI currently lists runs created through this workspace; pre-existing CLI
runs and v3 checkpoints remain available through the CLI. PPO counts are rounded
up to a complete rollout, as in the existing trainer. Training runs on the
server's CPU using the existing NumPy dynamics, even if a GPU is installed.

For web/job integration tests, install `requirements-test.txt`, then run
`make test-control-v4`. Keep the default loopback host for local use; the GUI
is a local experiment tool without user authentication.

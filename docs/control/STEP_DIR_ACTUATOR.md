# STEP/DIR actuator model

The torque actuator remains the default. Set `RLEnvConfigV3.actuator_mode` to
`"step_dir"` to run either the v3 or v4 training environment through the
command-space model. The selected mode and all `StepDirParams` are part of the
serialized environment configuration, so checkpoints retain their actuator
assumptions.

## Command and pulse path

The controller still produces a bounded carriage-acceleration target
`a_total`. At each control boundary, the environment converts it to a
short-horizon carriage velocity target and then to motor-shaft velocity:

```text
v_carriage_target = clamp(v_carriage + a_total * control_dt,
                          -max_motor_speed * pulley_radius,
                           max_motor_speed * pulley_radius)
omega_motor_target = v_carriage_target / pulley_radius
```

`StepDirActuator.command(omega_motor_target, acceleration_limit)` also exposes
direct motor-velocity commands for non-RL callers. It clips speed and
acceleration limits, applies first-order command tracking lag, integrates the
limited motor velocity, and quantizes the resulting position to integer STEP
pulses. `direction` is -1, 0, or +1. The default pulse resolution is a 1/16
microstep of a 1.8 degree motor step.

The pulse position is tracked by a bounded PD torque command. The existing
motor speed-dependent torque envelope and first-order torque response then
drive the mechanical motor state in `RigPlant`; the rotor may lag behind the
commanded step position when torque saturates. Diagnostics report requested
and limited speeds, acceleration, pulse count and direction, step count,
motor-position tracking error, and speed, acceleration, and torque saturation.

Default `StepDirParams` are 55 rad/s maximum motor speed, 300 rad/s² maximum
motor acceleration, 15 ms command lag, 1/16 microstepping, and heuristic
tracking gains. These values are starting simulation settings, not hardware
calibration. The model does not simulate driver phase current, microstep
waveforms, pulse width, or pulse timing within a physics step, and it is not a
full TMC2209/NEMA17 electrical model.

## Training

The existing residual action remains in normalized acceleration units. The
STEP/DIR mode converts the resulting constrained acceleration into motor
velocity and emits pulses through the drive model. Torque mode continues to
use the prior transition unchanged. For example:

```powershell
python software/python/train_residual_v4.py ppo --actuator-mode step_dir `
  --step-dir-max-velocity 55 --step-dir-max-acceleration 300
```

PPO training, teacher collection, and evaluation use the configured actuator
mode. The Phase 4 MPC augments its prediction state with STEP/DIR velocity and
command position, clones the live actuator for each prediction, and runs the
same bounded pulse-tracking interval used by the environment. It also reads a
copy of the live command queue so delayed and jittered commands are predicted
without advancing the environment's random stream. Nonlinear STEP/DIR
saturation is modeled and reported; it is not treated as a solver failure.

The preserved pre-Phase-4 MPC remains available only as a torque-mode,
zero-command-latency comparison. For STEP/DIR or delayed-command runs, compare
the Phase 4 MPC against LQR on the same configured environment.

Focused verification:

```powershell
python -m unittest discover -s tests -p test_step_dir_actuator.py -v
```

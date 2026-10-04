# Series-Elastic Dynamics Platform (SEDP)

A reconfigurable benchtop mechatronics platform for **active vibration control, series-elastic force control, haptics, system identification, digital twins, and learned control**.

> **Current status:** simulation-first. The nonlinear Python plant, browser visualizer, OpenModelica model, constrained classical controller, benchmark-gated RL workflow, and project documentation exist. Physical RP2350/NEMA17 hardware is the next major validation step.

## What SEDP is

SEDP is a short-travel linear carriage carrying a deliberately compliant single-axis resonator. The carriage is driven aggressively so the resonator visibly rings. Feedback then moves the carriage to remove mechanical energy and make the same mechanism appear much more heavily damped.

The same compliant mechanism can also work as a **series-elastic force-feedback input device**: spring deformation becomes the force measurement, allowing programmable centering, damping, detents, virtual stops, vibration, and remote-force feedback.

- **Dynamics/output mode** — excite and actively suppress mechanical resonance.
- **Haptic/input mode** — measure user/load deflection and render controlled force feedback.

## Planned hardware

- ~150 mm linear travel, 2020 extrusion, 1–2 × MGN7 rails
- GT2 belt, NEMA 17, TMC2209
- RP2350 controller
- AS5600 motor encoder + AS5600 resonator encoder
- 6-axis SPI IMU
- two physical springs from the resonator shaft to carriage anchors
- interchangeable masses/springs

## Software

```text
software/python/
  active_vibration_rig_2d.py     nonlinear plant + trajectories + controllers
  active_vibration_rig_web.py    SVG/Plotly/FastAPI live visualizer
  benchmark_suite.py             immutable SEDP-B1 acceptance gates
  rig_rl_env_v3.py               residual-acceleration RL environment
  rig_rl_policy_v3.py            runtime PPO v3 adapter
  pretrain_v3_damping.py         physics-informed warm start
  train_ppo_v3.py                benchmark-gated PPO fine-tuning
  evaluate_v3.py                 PPO v3 evaluation
  ppo_agent_v2.py                shared bounded-Beta PPO core

openmodelica/
  ActiveVibrationRig.mo          equation-based digital twin
  simulate.mos
  PARAMETERS.csv
```

## Model

Reduced coordinates:

\[
q=[x,\theta]^T
\]

with a nonlinear base-excited resonator model of the form

\[
(M_c+m)\ddot x+ml\cos\theta\,\ddot\theta
-ml\sin\theta\,\dot\theta^2+b_x\dot x=F_x
\]

\[
J_p\ddot\theta+ml\cos\theta\,\ddot x
+c_\theta\dot\theta+\tau_{spring}(\theta)+\tau_g(\theta)=\tau_{ext}.
\]

The Python plant additionally models motor inertia, torque lag, speed-dependent torque limits, GT2 elasticity/damping, friction, rail stops, sensors, and disturbances.

See [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md).

## Controller correction after the first showcase

The first showcase exposed two genuine problems:

1. the original full-state torque LQR could reduce a simple-step oscillation but behaved badly on violent reversals, consuming excessive carriage travel and crossing the software rail;
2. PPO v2 acted in raw motor-torque residual space and could reduce angle by accepting poor tracking/rail behavior. Its old headline benchmark was not reproducible from the committed simulator/checkpoint.

Those results are **superseded**.

The current stack is:

1. motion-limited trajectory tracking in **carriage-acceleration space**;
2. reduced LQR on `[e_x, e_v, theta, theta_dot]`, producing a bounded residual acceleration;
3. a relative-degree-2 **rail control-barrier projection** applied before torque conversion;
4. optional PPO v3 residual acceleration **inside the same deterministic safety layer**.

PPO no longer controls motor torque directly:

\[
a_{cmd}=\Pi_{rail}(a_{track}+a_{LQR}+a_{RL}).
\]

The legacy controller remains available as `lqr_legacy` only for regression.

## SEDP-B1 validated results

Every candidate sees the same plant, target, motor envelope, rail limits, integration step, and initial condition. A controller is rejected if it violates tracking, rail, saturation, or stop-contact gates even if its angle metric looks better.

| Scenario | Controller | Peak angle | Angle RMS | Tracking RMSE | Peak rail | Accepted |
|---|---|---:|---:|---:|---:|---|
| Step | Safe servo | 3.05° | 0.62° | 5.51 mm | 0.539 | baseline |
| Step | Constrained LQR | **1.80°** | **0.40°** | 6.26 mm | 0.573 | **yes** |
| Aggressive smooth reversal | Safe servo | 5.08° | 2.87° | 0.17 mm | 0.544 | baseline |
| Aggressive smooth reversal | Constrained LQR | **3.95°** | **1.84°** | 5.30 mm | 0.718 | **yes** |
| Square-reversal stress | Safe servo | 6.39° | 3.29° | 27.01 mm | 0.539 | baseline |
| Square-reversal stress | Constrained LQR | **3.45°** | **1.76°** | 29.20 mm | 0.614 | **yes** |
| Aggressive smooth reversal | PPO v3 | **3.82°** | **1.80°** | 5.53 mm | 0.729 | **yes** |
| Square-reversal stress | PPO v3 | **3.34°** | **1.67°** | 28.96 mm | 0.635 | **yes** |

No accepted LQR/PPO run in this table contacts the rail or spends time torque-saturated. PPO v3 is deliberately described as a **small residual refinement**, not a universal LQR replacement.

See [docs/benchmark_B1_fixed.json](docs/benchmark_B1_fixed.json) and [docs/RL_FORMULATION_V3.md](docs/RL_FORMULATION_V3.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the constrained controller:

```bash
cd software/python
python active_vibration_rig_2d.py --headless 5 --controller lqr --trajectory aggressive
```

Run the browser:

```bash
python active_vibration_rig_web.py --controller lqr --trajectory aggressive
```

With a v3 checkpoint:

```bash
python active_vibration_rig_web.py \
  --ppo-model ../../models/ppo_v3/policy_accepted.pt \
  --controller ppo --trajectory aggressive
```

Run the acceptance benchmark:

```bash
python benchmark_suite.py --controllers lqr --fail-on-reject
python evaluate_v3.py --model /path/to/policy_accepted.pt
```

## OpenModelica

```bash
cd openmodelica
omc simulate.mos
```

The Modelica package includes motor/transmission dynamics, nonlinear carriage/resonator coupling, explicit two-spring geometry, simplified encoders/IMU, and baseline/active-damping examples.

## RP2350 architecture

```text
AS5600 #1 -- I2C0 --\
AS5600 #2 -- I2C1 ---+--> Core 0: timestamp, observer, force model,
IMU -------- SPI ----/              inner loop, rail safety, motor command
                                      |
                                      v
                                  PIO STEP/DIR --> TMC2209 --> NEMA17

Core 1: trajectory generation, LQR/MPC/RL, system ID, haptics, USB/logging
```

## Documentation

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Mathematical model](docs/MATHEMATICAL_MODEL.md)
- [Hardware](docs/HARDWARE.md)
- [Firmware](docs/FIRMWARE.md)
- [RL formulation v3](docs/RL_FORMULATION_V3.md)
- [Superseded RL v2](docs/RL_FORMULATION_V2.md)
- [OpenModelica](docs/OPENMODELICA.md)
- [System model YAML](docs/SYSTEM_MODEL.yaml)
- [Browser UI](docs/BROWSER_UI.md)
- [FigJam diagrams](docs/FIGMA.md)
- [Historical RL results](archive/rl_v1/)

## Current limitations

- No physical hardware has been validated yet.
- Spring, belt, friction, latency, and actuator parameters remain provisional until system identification.
- Sensor models are control-level approximations.
- OpenModelica is not yet calibrated against the real rig.
- PPO v3 passes the same gates as LQR but currently offers only a modest reversal-specific improvement.

## Next physical milestone

Build the instrumented rig, identify the plant, recalibrate the twins, then rerun the exact same SEDP-B1 acceptance gates against hardware-observable state.

### Estimated-state residual control v4

The new experimental pipeline fixes PPO timing, reflection and timeout handling,
adds an encoder/gyro EKF with motion context and eight-frame history, and uses a
constrained MPC reference to collect teacher demonstrations. See
[the v4 workflow](docs/control/RESIDUAL_V4.md) and
[validation results](docs/control/V4_VALIDATION.md). It does not replace the
accepted v3 policy: initial candidates improve some vibration metrics but fail
paired tracking/peak-angle promotion checks.

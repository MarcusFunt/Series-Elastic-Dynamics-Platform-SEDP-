# Series-Elastic Dynamics Platform (SEDP)

A reconfigurable benchtop mechatronics platform for **active vibration control, series-elastic force control, haptics, system identification, digital twins, and learned control**.

> **Current status:** simulation-first. The nonlinear Python plant, browser visualizer, OpenModelica model, classical controllers, and residual-PPO workflow exist. Physical RP2350/NEMA17 hardware is the next major validation step.

## What SEDP is

SEDP is a short-travel linear carriage carrying a deliberately compliant single-axis resonator. The carriage is driven aggressively so the resonator visibly rings. Software then moves the carriage in feedback to remove that mechanical energy and make the same mechanism appear much more heavily damped or dynamically stiff.

The same compliant mechanism can also work in the opposite direction as a **series-elastic force-feedback input device**: spring deformation becomes the force measurement, allowing the stepper-driven mechanism to render programmable forces, detents, damping, virtual springs, and remote telemetry forces.

That gives the platform two complementary roles:

- **Dynamics/output mode** — excite and actively suppress mechanical resonance.
- **Haptic/input mode** — measure user/load deflection and render controlled force feedback.

## Planned hardware

- ~150 mm linear travel
- 2020 extrusion
- 1–2 × MGN7 linear rails
- GT2 belt transmission
- NEMA 17 stepper
- TMC2209 driver
- RP2350 controller
- AS5600 motor-side encoder
- AS5600 resonator/pivot encoder
- 6-axis SPI IMU on the resonator
- two physical springs from the resonator shaft to carriage anchor points
- interchangeable resonator masses/springs

## Repository layout

```text
software/python/
  active_vibration_rig_2d.py     nonlinear plant + trajectories + classical control
  active_vibration_rig_web.py    SVG/Plotly/FastAPI live browser visualization
  rig_rl_env_v2.py               residual-RL environment
  ppo_agent_v2.py                bounded Beta-policy PPO
  rig_rl_policy_v2.py            runtime policy adapter
  pretrain_randomized_lqr_v2.py  LQR imitation warm start
  train_stage_v2.py              curriculum/domain-randomized PPO training
  evaluate_v2.py                 deterministic benchmarks
  evaluate_randomized_v2.py      robustness benchmarks

openmodelica/
  ActiveVibrationRig.mo          equation-based OpenModelica plant
  simulate.mos                   batch simulations
  PARAMETERS.csv                 parameter map

docs/
  PROJECT_OVERVIEW.md
  MATHEMATICAL_MODEL.md
  HARDWARE.md
  FIRMWARE.md
  OPENMODELICA.md
  RL_FORMULATION_V2.md
  SYSTEM_MODEL.yaml
  FIGMA.md
  BROWSER_UI.md

archive/rl_v1/
  README.md
  LONG_RUN_RESULTS.md
```

## Main mathematical model

The reduced generalized coordinates are

\[
q = [x,\theta]^T
\]

where `x` is carriage position and `theta` is resonator deflection. A useful nonlinear reduced form is

\[
(M_c+m)\ddot{x}+ml\cos(\theta)\ddot{\theta}
-ml\sin(\theta)\dot{\theta}^2+b_x\dot{x}=F_x
\]

\[
J_p\ddot{\theta}+ml\cos(\theta)\ddot{x}
+c_\theta\dot{\theta}+\tau_{spring}(\theta)+\tau_g(\theta)
=\tau_{ext}.
\]

The Python plant extends this with motor inertia, motor torque lag, speed-dependent torque limits, GT2 belt elasticity/damping, Coulomb/viscous friction, rail end stops, sensor models, and nonlinear spring behavior.

See [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md).

## Control stack

Implemented:

- plain trajectory servo — intentionally ignores resonator motion
- corrected discrete LQR active damping
- motor-position mode
- residual PPO
- LQR/imitation warm-start workflow
- curriculum + domain randomization
- predictive rail-limit barrier and state-dependent residual limiting

The learned controller does **not** replace the entire trajectory servo. It learns a bounded residual:

\[
\tau_{cmd}=\tau_{servo}+a_{RL}\,\tau_{res,max}.
\]

See [docs/RL_FORMULATION_V2.md](docs/RL_FORMULATION_V2.md).

## Demonstration result

On the nominal aggressive-reversal benchmark used in the v2 RL pass:

| Controller | Peak angle | Angle RMS | Peak rail fraction |
|---|---:|---:|---:|
| Servo baseline | ~19.7° | ~9.18° | ~0.583 |
| Residual PPO v2 | ~9.3° | ~3.01° | ~0.495 |

The v2 reversal-specialized policy was the best checkpoint from that experiment. Trained checkpoints and rendered videos are treated as generated artifacts rather than committed source; see [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Classical headless simulation:

```bash
cd software/python
python active_vibration_rig_2d.py --headless 5 --controller lqr --trajectory aggressive
```

Browser visualizer:

```bash
cd software/python
python active_vibration_rig_web.py --controller lqr --trajectory aggressive
```

Open `http://127.0.0.1:8765`.

To use PPO, first train or supply a compatible checkpoint, then:

```bash
python active_vibration_rig_web.py \
  --ppo-model /path/to/policy.pt \
  --controller ppo \
  --trajectory aggressive
```

## Train residual PPO

A typical sequence is:

```bash
cd software/python

python pretrain_randomized_lqr_v2.py \
  --out ../../models/ppo_v2_anchor.pt

python train_stage_v2.py \
  --init ../../models/ppo_v2_anchor.pt \
  --anchor ../../models/ppo_v2_anchor.pt \
  --stage reversal \
  --steps 40000 \
  --out ../../models/ppo_v2_reversal.pt
```

Then benchmark with `evaluate_v2.py` and `evaluate_randomized_v2.py`.

## OpenModelica

```bash
cd openmodelica
omc simulate.mos
```

The OpenModelica package contains the parametric two-spring geometry, motor/transmission dynamics, nonlinear carriage/lever coupling, simplified encoders/IMU, and baseline/active-damping examples.

## RP2350 architecture

```text
AS5600 #1 -- I2C0 --\
AS5600 #2 -- I2C1 ---+--> Core 0: timestamp, observer, spring/force model,
IMU -------- SPI ----/              inner loop, rail safety, motion command
                                      |
                                      v
                                  PIO STEP/DIR --> TMC2209 --> NEMA17

Core 1: trajectory generation, LQR/MPC/RL, system ID, haptics, USB/logging
```

PIO is intended for deterministic edge generation, not for spring/dynamics mathematics.

## Documentation

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Mathematical model](docs/MATHEMATICAL_MODEL.md)
- [Hardware plan](docs/HARDWARE.md)
- [Firmware architecture](docs/FIRMWARE.md)
- [RL formulation v2](docs/RL_FORMULATION_V2.md)
- [OpenModelica notes](docs/OPENMODELICA.md)
- [Machine-readable system model](docs/SYSTEM_MODEL.yaml)
- [Browser UI](docs/BROWSER_UI.md)
- [Artifacts and reproducibility](docs/ARTIFACTS.md)
- [Editable FigJam diagrams](docs/FIGMA.md)

## Current limitations

- No physical hardware has been validated yet.
- Spring, belt, friction, and actuator parameters are provisional until system identification is performed.
- The AS5600/IMU models are control-level approximations, not electronics-level timing/noise models.
- The selected PPO result is strongest on repeated aggressive reversals, not universally optimal across every trajectory.
- OpenModelica has not yet been compiler-validated in this environment.

## Project direction

1. build and instrument the physical rig;
2. identify spring, damping, belt, friction, latency, and motor parameters;
3. calibrate the Python/OpenModelica twins against unseen physical trajectories;
4. validate encoder + gyro/IMU state estimation on RP2350;
5. benchmark servo, active damping, LQR/LQG, MPC, and RL under identical constraints;
6. reuse the mechanism as a force-feedback/haptic controller.

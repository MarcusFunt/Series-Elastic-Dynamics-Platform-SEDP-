# Project overview

## Purpose

SEDP is deliberately designed around a visible compliant mode rather than around eliminating compliance mechanically. With compensation disabled, the upper mass should ring dramatically after fast carriage motion; with compensation enabled, the same mechanism should appear much more heavily damped.

The platform is intentionally small: a short MGN7/GT2 carriage rather than an industrial crane. This keeps aggressive experimentation practical while retaining the control concepts found in anti-sway, flexible-structure, and series-elastic-actuator research.

## Two operating modes

### Active-dynamics mode

The carriage performs fast reversals that intentionally excite the resonator. Feedback uses the measured resonator state to move the base in a phase that removes energy.

Primary state:

`[x, x_dot, theta, theta_dot]`

Extended state:

`[phi_m, omega_m, x, x_dot, theta, theta_dot, tau_act]`

### Series-elastic / haptic mode

The compliant element becomes the force sensor. The controller measures relative displacement, maps that through a calibrated force law, and shifts motor position until measured force equals requested force.

This supports programmable centering, damping, detents, virtual stops, vibration, trim, and remote-force feedback for RC/robot teleoperation.

## Mechanical concept

- 2020 extrusion base
- one or two MGN7 rails
- GT2 belt
- NEMA17 stepper
- ~150 mm physical rail
- carriage-mounted one-axis resonator
- two diagonal springs between shaft and carriage
- replaceable spring/mass geometry
- AS5600 on motor side
- AS5600 on resonator pivot
- 6-axis IMU on moving resonator

## Control philosophy

Input shaping is a comparison baseline, not the central goal. SEDP is meant to be moved aggressively and then actively remove mechanical energy using feedback.

Benchmark ladder:

1. plain trajectory servo
2. direct velocity/angle active damping
3. LQR/LQG
4. constrained MPC
5. adaptive/model-based control
6. residual RL
7. optional end-to-end learned policy as a research comparison

## Digital twin strategy

Use two models:

- **Reduced model** for observers, LQR, linear MPC, and analysis.
- **High-fidelity plant** for nonlinear simulation, domain randomization, MPC validation, and sim-to-real training.

CAD provides geometry, mass distribution, and nominal inertia. Real experiments must identify spring behavior, damping, friction, belt compliance, actuator limits, and sensor/command delay.

## Scientific question

Can adaptive or learned feedback outperform a strong fixed classical controller when payload, spring stiffness, damping, friction, belt compliance, saturation, nonlinearities, or latency change?

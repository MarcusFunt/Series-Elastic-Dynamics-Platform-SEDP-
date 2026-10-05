# ActiveVibrationRig — OpenModelica model

This package is a first-pass OpenModelica implementation of SEDP.

## What is modeled

- NEMA17/stepper as a dynamic torque source, not at electromagnetic phase level
- motor rotor inertia, viscous loss, and smoothed Coulomb friction
- speed-dependent torque ceiling and first-order torque/current response
- 20T GT2 conversion
- belt stiffness/damping and optional cubic nonlinearity
- carriage mass, rail losses, finite travel, and soft end stops
- nonlinear base-accelerated upright resonator coupling
- gravity with the mass above the pivot
- explicit two-spring geometric model or equivalent rotational spring/damper
- simplified 12-bit AS5600s
- idealized lever IMU
- smooth trajectories
- baseline and active-damping controllers

## Coordinates / states

- `phiMotor`, `omegaMotor`
- `x`, `v`
- `theta`, `thetaDot`
- `tauAct`

## Two-spring geometry

Relative to the moving carriage pivot:

- shaft spring point: `(rSpring*sin(theta), rSpring*cos(theta))`
- left anchor: `(-anchorHalfSpacing, anchorY)`
- right anchor: `(+anchorHalfSpacing, anchorY)`

Each spring gets an exact instantaneous length and rate. The force vectors are resolved into torque about the pivot.
With `x = rSpring*sin(theta)` and `y = rSpring*cos(theta)`, the generalized
torque is `F dot d(position)/d(theta)`. This sign makes spring torque the
negative derivative of spring potential. The current spring dimensions and
rates are placeholders; see [the physics validation report](../docs/control/PHYSICS_VALIDATION.md)
before interpreting the geometric model's stability.

## Run

Open `ActiveVibrationRig.mo` in OMEdit and simulate one of:

- `ActiveVibrationRig.Examples.BaselineAggressive`
- `ActiveVibrationRig.Examples.ActiveDampingAggressive`
- `ActiveVibrationRig.Examples.FreeDecay`
- `ActiveVibrationRig.Examples.EquivalentTorsionBaseline`

Or:

```bash
omc simulate.mos
```

## Current modeling limits

The current pass does not model individual stepper winding currents, microstep lookup tables, TMC2209 chopper cycles, magnetic detent torque, or discrete loss-of-synchronism events. Sensor models are control-level approximations rather than detailed I2C/filter/noise models.

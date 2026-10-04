# OpenModelica model

The package in `openmodelica/` models the machine at the mechanical-control level.

Included:

- dynamic stepper torque source
- rotor inertia/losses
- speed-dependent torque ceiling
- first-order actuator torque response
- 20T GT2 conversion
- elastic/damped belt
- carriage mass and rail losses
- finite travel and soft stops
- nonlinear carriage/resonator coupling
- upright gravity contribution
- explicit two-spring geometric model
- equivalent rotational spring/damper alternative
- simplified AS5600 models
- idealized lever IMU
- baseline and active-damping examples

Examples:

- `ActiveVibrationRig.Examples.BaselineAggressive`
- `ActiveVibrationRig.Examples.ActiveDampingAggressive`
- `ActiveVibrationRig.Examples.FreeDecay`
- `ActiveVibrationRig.Examples.EquivalentTorsionBaseline`

Run:

```bash
cd openmodelica
omc simulate.mos
```

The model does not yet simulate TMC2209 chopper cycles, A/B winding current, microstep magnetic torque ripple, or discrete missed-step events.

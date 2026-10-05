# Plant physics validation — 4 October 2026

Regenerate the comparison and convergence evidence with:

```powershell
python software/python/validate_plant_physics.py --out docs/control/evidence/physics_validation.json
```

## Spring geometry and restoring torque

The spring shaft point is `(r sin(theta), r cos(theta))`, so positive `theta`
moves from `+y` toward `+x`. The generalized force is
`Q_theta = F · d(position)/d(theta)`. This is the negative derivative of the
spring potential. The previous OpenModelica expression used the opposite
planar cross-product sign; that sign was corrected in
`openmodelica/ActiveVibrationRig.mo` and the Python model.

The current OpenModelica spring dimensions and rates are explicitly marked as
placeholders. With those values, the geometric model has an effective
small-angle stiffness of **-0.5491 N·m/rad**, including gravity. The existing
equivalent torsion model has **+0.3082 N·m/rad**. At +5 degrees, their spring
torques are +0.03809 N·m and -0.03665 N·m respectively. The geometric
placeholder setup therefore pushes away from upright; this is a parameter
result, not evidence that the physical rig is unstable. The existing Python
default remains the equivalent torsion model.

For symmetric anchors, zero spring torque occurs at the absolute-angle
symmetry axes (`alpha=0` or `pi`). `theta_neutral_world` sets the reference for
the incremental gravity term; it does not force the geometric spring pair to
have zero torque at `theta=0` for other reference angles. No calibration has
been fitted between the geometric and equivalent-torsion laws: the geometric
dimensions, rates, and free length are placeholders, and measured rig values
are not yet available. In particular, their opposite effective-stiffness signs
at upright show that treating them as interchangeable would be unsupported.

The executable comparison reports torques from -15 to +15 degrees. Unit
checks compare each geometric spring's lengths, length rates, force vectors,
and generalized torque against an independent calculation, check left/right
symmetry, and verify torque equals the negative potential gradient. The
reported effective stiffness is also checked against the numerical slope of
the total spring-plus-gravity torque.

## Energy and integration

`RigPlant.total_mechanical_energy()` includes rotor and coupled
carriage/resonator kinetic energy, belt and resonator spring potentials,
incremental gravity potential, and soft-stop potential. It excludes actuator
electrical storage and dissipative terms. The existing RL energy reward stays
as a resonator-only control score; it is not used as a conservation measure.

The undamped, unforced geometric-mode check conserves total energy to less
than **2×10⁻⁵ relative drift** over 0.2 seconds at a 0.25 ms timestep.
The damped default-mode check decreases total mechanical energy.

The convergence run covers both the equivalent-torsion and explicit two-spring
models. Each uses a 120 ms horizon, a 0.125 ms RK4 reference, the same initial
seven-state vector, and a free trajectory plus a held 0.08 N·m actuator torque
command. Dissipation and Coulomb friction are set to zero; the complete final
parameter values and explicit overrides for each model are recorded in the
JSON evidence. State error is the maximum over the horizon of the normalized
RMS error across all seven states. The scales and per-state final errors are
also included.

With limits of 10⁻⁴ for normalized state error and free-case relative energy
drift, the largest passing timestep in the tested set is **1 ms for each
spring mode**. At 1 ms, equivalent-torsion state error is 2.76×10⁻⁵ (free) and
8.29×10⁻⁵ (driven), with free-case energy drift 1.16×10⁻⁵. The geometric
model's values are 2.78×10⁻⁵ (free), 8.29×10⁻⁵ (driven), and 8.99×10⁻⁸ energy
drift. The 2 ms driven case exceeds the state-error limit for both models.

These timestep results apply to the recorded initial condition, held torque
inputs, parameter sets, and 120 ms horizon. They do not establish a timestep
bound for every randomized or hardware-calibrated configuration.

## Verification

Run the focused checks with:

```powershell
python -m unittest discover -s tests -p test_plant_physics.py -v
```

The Python checks and evidence generation pass. The OpenModelica source change
was not compiled in this environment because the `omc` executable is
unavailable.

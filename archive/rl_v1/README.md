# Active Vibration Rig — RL + Classical Control Toolkit

This archived first RL pass extended the nonlinear active-vibration / series-elastic rig simulator with residual PPO and fixed an important classical-control bug.

## Important bug fixed

The old LQR mode silently fell back to the normal servo because the custom DARE iteration did not converge. That is why early visualizations could look identical with and without active damping.

The corrected implementation uses `scipy.linalg.solve_discrete_are` first, with the original iterative solver retained as a fallback.

Nominal step benchmark from that pass:

- Servo peak angle: **9.62 deg**
- LQR peak angle: **1.21 deg**
- Servo settling time: **1.02 s**
- LQR settling time: **0.234 s**

## RL architecture

The first workflow used residual PPO:

```text
motor torque command = ordinary position servo + PPO residual torque
```

The policy acted at 100 Hz while the nonlinear plant integrated at 2 kHz. The 11-D observation contained motor-side position/velocity, carriage state, resonator angle/rate, actuator torque, reference position/velocity/acceleration, and the previous policy action.

Domain randomization covered resonator mass, spring stiffness/damping, belt stiffness, carriage damping, available motor torque, initial resonator state, external kicks, and aggressive target reversals.

## Lessons

Behavior cloning from corrected LQR was a much better starting point than pure random exploration. Longer PPO training improved a narrow nominal step objective for a while but then exploited reward trade-offs and did not become a universally better aggressive-reversal controller.

That result motivated the v2 formulation in `docs/RL_FORMULATION_V2.md`: energy-removal shaping, predictive rail barriers, KL anchoring, symmetry regularization, stronger curriculum/domain randomization, and bounded Beta actions.

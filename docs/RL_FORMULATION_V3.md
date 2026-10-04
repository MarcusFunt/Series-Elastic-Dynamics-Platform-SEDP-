# RL formulation v3 — safe residual acceleration

## Why v3 exists

The original showcase exposed two regressions: the legacy full-state LQR could consume excessive rail on aggressive reversals, and torque-residual PPO v2 could trade tracking/rail use for lower angular RMS. The old v2 headline result was also not reproducible from the committed simulator/checkpoint.

v3 changes both the deterministic controller and learned action space.

## Deterministic base controller

The current LQR acts on

\[
z=[e_x,\dot e_x,\theta,\dot\theta]^T
\]

and outputs **residual carriage acceleration** rather than raw motor torque.

\[
a_{track}=a_r+k_p(x_r-x)+k_d(\dot x_r-\dot x)
\]

\[
a_{LQR}=-Kz.
\]

Both terms go through the same rail-aware acceleration projection.

### Rail control barrier

For right boundary `h_R=L-x` and left boundary `h_L=L+x`, the controller enforces the relative-degree-2 condition

\[
\ddot h+2\zeta\omega\dot h+\omega^2h\ge0.
\]

This generates an admissible interval

\[
a_{min}(x,\dot x)\le a\le a_{max}(x,\dot x)
\]

intersected with the global acceleration limit. Active damping can therefore no longer “win” by driving into the rail.

## PPO action

PPO outputs only a bounded residual acceleration:

\[
a_{RL}\in[-a_{res,max},a_{res,max}]
\]

and

\[
a_{cmd}=\Pi_{rail}(a_{track}+a_{LQR}+a_{RL}).
\]

The policy cannot bypass the deterministic rail safety layer.

## Observation

The 12-D normalized observation contains:

- position and velocity tracking error
- resonator angle and angular velocity
- absolute carriage position / rail fraction
- carriage velocity
- deterministic base acceleration
- motor speed
- actuator torque state
- reference acceleration
- previous learned action
- remaining rail margin

## Reward

v3 penalizes tracking error, angle/rate, absolute resonator energy, action magnitude/slew, adverse corrections that accelerate away from an existing tracking error, and predicted stopping-distance rail risk. It rewards actual energy removal. Episodes terminate on rail or extreme-angle violations.

Domain randomization covers payload, rotational stiffness/damping, belt stiffness, rail damping, motor torque, initial resonator state, and external kicks.

## Warm start

Before PPO, the actor behavior-clones a transparent physical residual

\[
a_{teacher}\approx0.5\dot\theta+1.0\theta
\]

clipped to the residual budget. PPO fine-tunes from that prior.

## Checkpoint acceptance

Reward alone can never promote a policy. Every publishable checkpoint must pass **SEDP-B1** under the exact same plant/reference/limits as the classical controller, including:

- peak rail fraction
- stop contact
- torque saturation fraction
- tracking RMSE
- peak angle
- angle RMS

## Current result

| Scenario | Controller | Peak angle | RMS angle | Tracking RMSE | Peak rail |
|---|---|---:|---:|---:|---:|
| step | constrained LQR | 1.80° | 0.40° | 6.26 mm | 0.573 |
| step | PPO v3 | 1.76° | 0.42° | 6.33 mm | 0.591 |
| aggressive | constrained LQR | 3.95° | 1.84° | 5.30 mm | 0.718 |
| aggressive | PPO v3 | 3.82° | 1.80° | 5.53 mm | 0.729 |
| square stress | constrained LQR | 3.45° | 1.76° | 29.20 mm | 0.614 |
| square stress | PPO v3 | 3.34° | 1.67° | 28.96 mm | 0.635 |

PPO is **not** claimed to universally beat LQR. It gives a small reversal-specific improvement while remaining inside the same gates.

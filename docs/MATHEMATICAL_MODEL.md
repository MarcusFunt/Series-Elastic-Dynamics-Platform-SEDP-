# Mathematical model

## Coordinates

Reduced generalized coordinates:

\[
q=[x,\theta]^T
\]

Extended coordinates:

\[
q_f=[\phi_m,x,\theta]^T
\]

The Python plant state is

\[
y=[\phi_m,\dot\phi_m,x,\dot x,\theta,\dot\theta,\tau_{act}]^T.
\]

## Motor + GT2 transmission

\[
\delta_b=r_p\phi_m-x,\qquad \dot\delta_b=r_p\dot\phi_m-\dot x
\]

\[
F_b=k_b\delta_b+c_b\dot\delta_b+k_{b3}\delta_b^3.
\]

Motor dynamics:

\[
J_m\ddot\phi_m=
\tau_{act}-b_m\dot\phi_m-\tau_{c,m}\operatorname{sgn}_\epsilon(\dot\phi_m)-r_pF_b.
\]

The commanded torque is filtered through a speed-dependent limit and first-order actuator response.

## Coupled carriage/resonator dynamics

Let `alpha = theta_neutral + theta`. The coupled mass matrix is

\[
\begin{bmatrix}
M & ml\cos\alpha\\
ml\cos\alpha & J_p
\end{bmatrix}
\begin{bmatrix}
\ddot x\\
\ddot\theta
\end{bmatrix}
=
\begin{bmatrix}
R_x\\R_\theta
\end{bmatrix}.
\]

The implementation includes rail damping/friction, soft stops, centrifugal coupling, nonlinear stiffness, gravity increment about a preloaded neutral pose, and external disturbances.

A reduced interpretation is

\[
J_p\ddot\theta+c_\theta\dot\theta+k_{eff}\theta=-\Gamma\ddot{x}+\tau_{ext}.
\]

This is the core active-damping mechanism: carriage acceleration can add or remove energy depending on phase.

## Two-spring geometric model

Relative to the carriage pivot,

\[
r_s=
\begin{bmatrix}
r\sin\theta\\r\cos\theta
\end{bmatrix},
\quad
a_L=[-a,h]^T,
\quad
a_R=[+a,h]^T.
\]

For each spring,

\[
d_i=r_s-a_i,\qquad L_i=\|d_i\|,
\]

\[
f_i=k_s(L_i-L_0)+c_s\dot L_i,
\]

\[
F_i=-f_i\frac{d_i}{L_i}.
\]

The pivot torque is

\[
\tau_s=r_s\times(F_L+F_R).
\]

This exact geometric model is implemented in OpenModelica. The fast Python control/RL plant currently uses an equivalent rotational law.

## Sensors

Intended measurements:

- motor AS5600: `phi_m`
- resonator AS5600: `theta`
- gyro: approximately `theta_dot + bias + noise`
- accelerometer: base + tangential + centripetal + gravity terms

The intended RP2350 observer uses the encoder as absolute low-frequency truth and the gyro as the high-bandwidth angular-rate channel.

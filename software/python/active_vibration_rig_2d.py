#!/usr/bin/env python3
"""
Active vibration / series-elastic bench rig — parametric 2D dynamics renderer
=============================================================================

This is a self-contained, code-first description of the rig discussed in the
ChatGPT design session.  It deliberately models the mechanism as a *base-
excited compliant rotary oscillator on a motor-driven linear carriage*, not as
an inverted-pendulum balancing problem and not primarily as an input-shaping
problem.

GENERALIZED COORDINATES
-----------------------
    q = [phi_m, x, theta]^T

    phi_m : motor/pulley shaft angle [rad]
    x     : carriage position on the MGN7/GT2 axis [m]
    theta : resonator deflection relative to its mechanically neutral angle [rad]

The physical lever neutral axis has world angle

    alpha = theta_0 + theta

where theta_0=0 means upright, theta_0=pi means hanging downward.  A constant
spring preload is assumed to cancel the *static* gravity torque at theta=0;
therefore gravity contributes only the incremental term about the neutral pose.

MOTOR / BELT MODEL
------------------
Pulley displacement and belt extension:

    x_p     = r_p * phi_m
    delta_b = r_p*phi_m - x

Belt force (linear + optional cubic elasticity):

    F_b = k_b*delta_b + c_b*delta_b_dot + k_b3*delta_b^3

Motor dynamics:

    J_m*phi_ddot = tau_act
                   - b_m*phi_dot
                   - tau_c,m*tanh(phi_dot/v_eps,m)
                   - r_p*F_b

A first-order actuator lag is included:

    tau_act_dot = (sat(tau_cmd, +/-tau_limit(phi_dot)) - tau_act) / tau_tau

with a phenomenological speed-dependent torque limit

    tau_limit(omega) = tau_hold / sqrt(1 + (|omega|/omega_corner)^2)

This is intentionally a compact *mechanical* stepper model; it is not a full
phase-current/electromagnetic model of a TMC2209 + NEMA17.

CARRIAGE + RESONATOR MODEL
--------------------------
Let

    M = m_c + m_r
    B(theta) = m_r*l*cos(alpha)
    J_p = J_com + m_r*l^2

The coupled equations are

    M*x_ddot + B*theta_ddot
      - m_r*l*sin(alpha)*theta_dot^2
      + b_x*x_dot + F_c,x*tanh(x_dot/v_eps,x)
      = F_b + F_ext + F_stop

    J_p*theta_ddot + B*x_ddot
      + c_theta*theta_dot
      + tau_c,theta*tanh(theta_dot/v_eps,theta)
      + k_theta*theta + k_theta3*theta^3
      - m_r*g*l*(sin(theta_0+theta)-sin(theta_0))
      = tau_ext

or, at every integration step,

    [ M   B ] [x_ddot    ] = [rhs_x    ]
    [ B  Jp ] [theta_ddot]   [rhs_theta]

The gravity expression is incremental about theta=0 for different mounting
orientations. In equivalent-torsion mode that pose is statically balanced; in
geometric-spring mode the selected spring pair must also have zero torque
there, or a separate preload-compensation torque must be modeled. For the
intended upright resonator use theta_0=0. In equivalent-torsion mode, the
small-angle effective rotational stiffness is approximately

    k_eff ~= k_theta - m_r*g*l

so the spring must exceed m_r*g*l for the upright neutral pose to remain
passively stable.

SENSOR MODEL
------------
The renderer also exposes physically useful synthetic sensor signals:
    * 12-bit motor AS5600 angle (wrapped + quantized)
    * 12-bit lever AS5600 angle (wrapped + quantized)
    * lever gyro rate
    * two-axis ideal IMU specific force at a configurable radius on the lever

The IMU point acceleration is computed from rigid-body kinematics; this makes
this file a useful source of truth for later Kalman/EKF work.

CONTROL / MOTION TOOLING
------------------------
Interactive mode provides:
    * baseline trajectory-only servo (rings freely after aggressive motion)
    * automatically linearized discrete LQR active damping
    * hold / step / sine / chirp / aggressive reversal / manual trajectories
    * live sliders for key mechanical parameters
    * click-on-rail or arrow-key manual target control
    * disturbance torque "kick"
    * play/pause/reset and CSV export

Headless mode can be used for scripted sweeps and CI-style simulation.

Dependencies: numpy, matplotlib.  No scipy/control package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
try:
    from scipy.linalg import solve_discrete_are
except Exception:
    solve_discrete_are = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def smooth_sign(v: float, eps: float) -> float:
    eps = max(abs(eps), 1e-9)
    return math.tanh(v / eps)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quantize_angle_12bit(a: float) -> float:
    q = round(((a % (2.0 * math.pi)) / (2.0 * math.pi)) * 4096.0) % 4096
    return q * (2.0 * math.pi / 4096.0)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class PlantParams:
    # Linear carriage / resonator
    carriage_mass: float = 0.45          # kg, carriage excluding resonator
    resonator_mass: float = 0.12         # kg
    lever_com_distance: float = 0.095    # m, pivot -> resonator COM
    lever_inertia_com: float = 1.8e-4    # kg m^2, about resonator COM
    theta_neutral_world: float = 0.0     # rad; 0=upright, pi=downward

    # Equivalent rotational compliance at the resonator pivot
    k_theta: float = 0.42                # N m / rad
    k_theta3: float = 0.0                # N m / rad^3
    c_theta: float = 0.006               # N m s / rad
    use_geometric_springs: bool = False
    spring_anchor_half_spacing: float = 0.035  # m; symmetric carriage anchors
    spring_anchor_y: float = -0.015      # m; anchor coordinate relative to pivot
    spring_shaft_radius: float = 0.040   # m; spring attachment radius
    spring_k: float = 500.0              # N/m per spring
    spring_d: float = 1.0                # N s/m per spring
    spring_free_length: float = 0.010    # m per spring
    theta_coulomb: float = 0.0015        # N m, smooth Coulomb friction
    theta_friction_eps: float = 0.05     # rad/s

    # Carriage losses
    b_x: float = 1.2                     # N s / m
    x_coulomb: float = 0.30              # N
    x_friction_eps: float = 0.015        # m/s

    # Stepper / pulley
    motor_inertia: float = 5.0e-5        # kg m^2
    motor_viscous: float = 2.0e-4        # N m s / rad
    motor_coulomb: float = 0.007          # N m
    motor_friction_eps: float = 0.5       # rad/s
    pulley_teeth: int = 20
    belt_pitch: float = 0.002             # m

    # Belt / transmission compliance
    belt_stiffness: float = 6500.0        # N/m
    belt_damping: float = 18.0            # N s/m
    belt_cubic: float = 0.0               # N/m^3

    # Phenomenological motor torque envelope and actuator lag
    motor_hold_torque: float = 0.48       # N m
    motor_omega_corner: float = 55.0      # rad/s
    motor_torque_time_constant: float = 0.0015  # s

    # Travel and soft-stop model
    rail_half_travel: float = 0.065       # m usable +/- range
    stop_stiffness: float = 18000.0       # N/m outside software travel
    stop_damping: float = 120.0           # N s/m

    # Environment
    gravity: float = 9.81                 # m/s^2

    # Synthetic sensor placement/noise
    imu_radius: float = 0.075             # m from pivot
    gyro_bias: float = math.radians(0.08) # rad/s
    gyro_noise_std: float = math.radians(0.03)
    accel_noise_std: float = 0.015        # m/s^2

    @property
    def pulley_radius(self) -> float:
        # pitch circumference = tooth_count * belt_pitch
        return self.pulley_teeth * self.belt_pitch / (2.0 * math.pi)

    @property
    def resonator_inertia_pivot(self) -> float:
        return self.lever_inertia_com + self.resonator_mass * self.lever_com_distance**2

    @property
    def effective_small_angle_stiffness(self) -> float:
        """Return the negative net-torque slope at the configured theta=0 pose.

        This is an equilibrium restoring stiffness only if net torque at that
        pose is zero. For an offset geometric-spring pose with residual spring
        torque, it is the local tangent stiffness around that reference.
        """
        p = self
        if p.use_geometric_springs:
            radius = p.spring_shaft_radius
            angle = p.theta_neutral_world
            shaft_x = radius * math.sin(angle)
            shaft_y = radius * math.cos(angle)
            tangent_x = radius * math.cos(angle)
            tangent_y = -radius * math.sin(angle)
            curvature_x = -radius * math.sin(angle)
            curvature_y = -radius * math.cos(angle)
            spring_stiffness = 0.0
            for anchor_x in (-p.spring_anchor_half_spacing, p.spring_anchor_half_spacing):
                dx = shaft_x - anchor_x
                dy = shaft_y - p.spring_anchor_y
                length = math.hypot(dx, dy)
                dot = dx * tangent_x + dy * tangent_y
                length_rate_per_angle = dot / max(length, 1e-12)
                length_accel_per_angle = (
                    (tangent_x**2 + tangent_y**2 + dx * curvature_x + dy * curvature_y)
                    / max(length, 1e-12)
                    - dot**2 / max(length, 1e-12) ** 3
                )
                spring_stiffness += p.spring_k * (
                    length_rate_per_angle**2
                    + (length - p.spring_free_length) * length_accel_per_angle
                )
        else:
            spring_stiffness = p.k_theta
        gravity_slope = p.resonator_mass * p.gravity * p.lever_com_distance * math.cos(p.theta_neutral_world)
        return spring_stiffness - gravity_slope

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PlantParams":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass(frozen=True)
class StepDirParams:
    """Command, pulse-resolution, and mechanical tracking parameters.

    Speeds use motor-shaft radians per second. ``step_angle_rad`` is the
    commanded shaft rotation represented by one STEP pulse (full step divided
    by the selected microstep factor).
    """

    step_angle_rad: float = math.radians(1.8) / 16.0
    max_velocity_rad_s: float = 55.0
    max_acceleration_rad_s2: float = 300.0
    tracking_time_constant_s: float = 0.015
    position_gain_nm_per_rad: float = 20.0
    velocity_gain_nms_per_rad: float = 0.0015

    def __post_init__(self) -> None:
        values = (self.step_angle_rad, self.max_velocity_rad_s,
                  self.max_acceleration_rad_s2, self.tracking_time_constant_s,
                  self.position_gain_nm_per_rad, self.velocity_gain_nms_per_rad)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('STEP/DIR parameters must be finite')
        if any(value <= 0.0 for value in values):
            raise ValueError('STEP/DIR resolution, limits, lag, and gains must be positive')


class StepDirActuator:
    """Bounded STEP/DIR pulse planner with a lagged, torque-limited rotor tracker.

    ``command`` accepts a requested motor angular velocity and optional
    acceleration limit. ``advance`` updates the bounded velocity trajectory,
    quantizes its integrated position into integer pulse counts, then returns a
    bounded motor torque command that tracks the pulse position in RigPlant.
    This is a mechanical command model; it does not simulate phase current or
    microstep current waveforms.
    """

    def __init__(self, params: Optional[StepDirParams] = None):
        self.params = params or StepDirParams()
        self.reset()

    def reset(self, motor_position_rad: float = 0.0) -> None:
        if not math.isfinite(motor_position_rad):
            raise ValueError('Initial motor position must be finite')
        self.requested_velocity_rad_s = 0.0
        self.velocity_command_rad_s = 0.0
        self.acceleration_limit_rad_s2 = self.params.max_acceleration_rad_s2
        self.requested_acceleration_limit_rad_s2 = self.acceleration_limit_rad_s2
        self.velocity_rad_s = 0.0
        self.acceleration_rad_s2 = 0.0
        self.position_rad = float(motor_position_rad)
        self.step_count = int(round(self.position_rad / self.params.step_angle_rad))
        self.position_rad = self.step_count * self.params.step_angle_rad
        self.step_phase_rad = 0.0
        self.velocity_saturated = False
        self.acceleration_limit_saturated = False

    def command(self, target_velocity_rad_s: float,
                acceleration_limit_rad_s2: Optional[float] = None) -> Dict[str, object]:
        if not math.isfinite(target_velocity_rad_s):
            raise ValueError('STEP/DIR velocity command must be finite')
        requested_accel = (self.params.max_acceleration_rad_s2 if acceleration_limit_rad_s2 is None
                           else float(acceleration_limit_rad_s2))
        if not math.isfinite(requested_accel) or requested_accel < 0.0:
            raise ValueError('STEP/DIR acceleration limit must be finite and nonnegative')
        self.requested_velocity_rad_s = float(target_velocity_rad_s)
        self.velocity_command_rad_s = clamp(
            self.requested_velocity_rad_s,
            -self.params.max_velocity_rad_s,
            self.params.max_velocity_rad_s,
        )
        self.requested_acceleration_limit_rad_s2 = requested_accel
        self.acceleration_limit_rad_s2 = min(requested_accel, self.params.max_acceleration_rad_s2)
        self.velocity_saturated = not math.isclose(
            self.velocity_command_rad_s, self.requested_velocity_rad_s,
            rel_tol=0.0, abs_tol=1e-12,
        )
        self.acceleration_limit_saturated = requested_accel > self.params.max_acceleration_rad_s2
        return {
            'requested_velocity_rad_s': self.requested_velocity_rad_s,
            'velocity_command_rad_s': self.velocity_command_rad_s,
            'requested_acceleration_limit_rad_s2': requested_accel,
            'acceleration_limit_rad_s2': self.acceleration_limit_rad_s2,
            'velocity_saturated': self.velocity_saturated,
            'acceleration_limit_saturated': self.acceleration_limit_saturated,
        }

    def advance(self, dt: float, actual_phi_rad: float, actual_omega_rad_s: float,
                torque_limit_nm: float) -> Dict[str, object]:
        values = (dt, actual_phi_rad, actual_omega_rad_s, torque_limit_nm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('STEP/DIR update values must be finite')
        if dt <= 0.0 or torque_limit_nm < 0.0:
            raise ValueError('STEP/DIR timestep must be positive and torque limit nonnegative')

        response = -math.expm1(-dt / self.params.tracking_time_constant_s)
        lagged_delta = (self.velocity_command_rad_s - self.velocity_rad_s) * response
        accel_delta_limit = self.acceleration_limit_rad_s2 * dt
        applied_delta = clamp(lagged_delta, -accel_delta_limit, accel_delta_limit)
        old_velocity = self.velocity_rad_s
        self.velocity_rad_s = clamp(
            old_velocity + applied_delta,
            -self.params.max_velocity_rad_s,
            self.params.max_velocity_rad_s,
        )
        self.acceleration_rad_s2 = (self.velocity_rad_s - old_velocity) / dt
        dynamic_accel_limited = not math.isclose(
            applied_delta, lagged_delta, rel_tol=0.0, abs_tol=1e-12,
        )
        self.acceleration_limit_saturated = self.acceleration_limit_saturated or dynamic_accel_limited

        self.position_rad += self.velocity_rad_s * dt
        previous_count = self.step_count
        self.step_count = int(round(self.position_rad / self.params.step_angle_rad))
        self.step_phase_rad = self.position_rad - self.step_count * self.params.step_angle_rad
        step_delta = self.step_count - previous_count
        step_position = self.step_count * self.params.step_angle_rad
        direction = (1 if step_delta > 0 else -1 if step_delta < 0 else 0)

        torque_request = (
            self.params.position_gain_nm_per_rad * (step_position - actual_phi_rad)
            + self.params.velocity_gain_nms_per_rad * (self.velocity_rad_s - actual_omega_rad_s)
        )
        torque_command = clamp(torque_request, -torque_limit_nm, torque_limit_nm)
        return {
            'requested_velocity_rad_s': self.requested_velocity_rad_s,
            'velocity_command_rad_s': self.velocity_command_rad_s,
            'step_velocity_rad_s': self.velocity_rad_s,
            'step_acceleration_rad_s2': self.acceleration_rad_s2,
            'velocity_tracking_error_rad_s': self.velocity_command_rad_s - self.velocity_rad_s,
            'requested_acceleration_limit_rad_s2': self.requested_acceleration_limit_rad_s2,
            'acceleration_limit_rad_s2': self.acceleration_limit_rad_s2,
            'step_pulses': abs(int(step_delta)),
            'step_delta': int(step_delta),
            'step_count': int(self.step_count),
            'direction': direction,
            'step_position_rad': step_position,
            'step_phase_rad': self.step_phase_rad,
            'motor_tracking_error_rad': step_position - actual_phi_rad,
            'torque_request_nm': torque_request,
            'torque_command_nm': torque_command,
            'velocity_saturated': self.velocity_saturated,
            'acceleration_limit_saturated': self.acceleration_limit_saturated,
            'torque_saturated': not math.isclose(torque_command, torque_request,
                                                 rel_tol=0.0, abs_tol=1e-12),
        }


@dataclass
class ControllerParams:
    # Legacy force-domain servo. Kept as the deliberately under-damped baseline.
    x_kp: float = 900.0
    x_kd: float = 45.0

    # Motion-limited acceleration-domain tracker used by modern controllers.
    track_kp: float = 95.0
    track_kd: float = 18.0
    max_accel: float = 11.0
    max_speed: float = 0.80

    # Relative-degree-2 rail control-barrier filter.
    rail_soft_fraction: float = 0.80
    rail_cbf_omega: float = 15.0
    rail_cbf_zeta: float = 1.0

    # Energy-shaping diagnostic residual acceleration.
    energy_theta_gain: float = 2.5
    energy_omega_gain: float = 1.45
    energy_residual_limit: float = 6.0

    # Reduced 4-state LQR on [e_x, e_v, theta, theta_dot], control=carriage acceleration.
    lqr_q_x: float = 80.0
    lqr_q_v: float = 8.0
    lqr_q_theta: float = 1500.0
    lqr_q_theta_dot: float = 60.0
    lqr_r_accel: float = 2.5
    lqr_residual_limit: float = 6.0
    lqr_design_dt: float = 0.01

    # Legacy full-state motor-torque LQR retained only for regression.
    q_phi: float = 0.03
    q_omega_m: float = 0.002
    q_x: float = 1600.0
    q_v: float = 22.0
    q_theta: float = 900.0
    q_theta_dot: float = 55.0
    q_tau: float = 0.05
    r_tau_cmd: float = 0.30

@dataclass
class MotionParams:
    amplitude: float = 0.035
    step_time: float = 0.50
    sine_frequency: float = 1.2
    aggressive_period: float = 0.75
    aggressive_move_fraction: float = 0.72
    chirp_f0: float = 0.4
    chirp_f1: float = 7.0
    chirp_duration: float = 8.0


# ---------------------------------------------------------------------------
# Plant model
# ---------------------------------------------------------------------------

class RigPlant:
    """Nonlinear 7-state model.

    State y = [phi_m, omega_m, x, v, theta, omega, tau_act]
    Input u = commanded motor torque tau_cmd [N m]
    """

    IDX_PHI = 0
    IDX_WM = 1
    IDX_X = 2
    IDX_V = 3
    IDX_TH = 4
    IDX_W = 5
    IDX_TAU = 6

    def __init__(self, p: PlantParams):
        self.p = p

    def motor_torque_limit(self, omega_m: float) -> float:
        p = self.p
        return p.motor_hold_torque / math.sqrt(1.0 + (abs(omega_m) / max(p.motor_omega_corner, 1e-6))**2)

    def belt_force(self, y: np.ndarray) -> Tuple[float, float, float]:
        p = self.p
        phi, wm, x, v = float(y[0]), float(y[1]), float(y[2]), float(y[3])
        delta = p.pulley_radius * phi - x
        delta_dot = p.pulley_radius * wm - v
        force = p.belt_stiffness * delta + p.belt_damping * delta_dot + p.belt_cubic * delta**3
        return force, delta, delta_dot

    def soft_stop_force(self, x: float, v: float) -> float:
        p = self.p
        L = p.rail_half_travel
        if x > L:
            return -p.stop_stiffness * (x - L) - p.stop_damping * max(v, 0.0)
        if x < -L:
            return -p.stop_stiffness * (x + L) - p.stop_damping * min(v, 0.0)
        return 0.0

    def spring_geometry(self, theta: float, theta_dot: float = 0.0) -> Dict[str, object]:
        """Return the explicit two-spring geometry in OpenModelica coordinates.

        The shaft point is ``(r*sin(alpha), r*cos(alpha))``, where
        ``alpha = theta_neutral_world + theta``, and the two
        carriage anchors are ``(-a, anchor_y)`` and ``(+a, anchor_y)``.
        Springs act in tension or compression. For symmetric anchors their
        torque is zero at the absolute-angle symmetry axes (alpha=0 or pi),
        but a different theta_neutral_world does not make theta=0 a spring
        equilibrium. Static gravity compensation remains a separate
        incremental-gravity term used by the torsional model.
        """
        p = self.p
        alpha = p.theta_neutral_world + theta
        shaft_x = p.spring_shaft_radius * math.sin(alpha)
        shaft_y = p.spring_shaft_radius * math.cos(alpha)
        shaft_vx = p.spring_shaft_radius * math.cos(alpha) * theta_dot
        shaft_vy = -p.spring_shaft_radius * math.sin(alpha) * theta_dot
        lengths = []
        length_rates = []
        forces = []
        torque = 0.0
        potential = 0.0
        for anchor_x in (-p.spring_anchor_half_spacing, p.spring_anchor_half_spacing):
            dx = shaft_x - anchor_x
            dy = shaft_y - p.spring_anchor_y
            length = math.hypot(dx, dy)
            safe_length = max(length, 1e-12)
            length_rate = (dx * shaft_vx + dy * shaft_vy) / safe_length
            magnitude = p.spring_k * (length - p.spring_free_length) + p.spring_d * length_rate
            force_x = -magnitude * dx / safe_length
            force_y = -magnitude * dy / safe_length
            lengths.append(length)
            length_rates.append(length_rate)
            forces.append((force_x, force_y))
            # theta grows from +y toward +x, so Q_theta = F dot d(position)/d(theta).
            torque += force_x * p.spring_shaft_radius * math.cos(alpha)
            torque -= force_y * p.spring_shaft_radius * math.sin(alpha)
            potential += 0.5 * p.spring_k * (length - p.spring_free_length) ** 2
        return {
            'lengths': tuple(lengths),
            'length_rates': tuple(length_rates),
            'forces': tuple(forces),
            'torque': float(torque),
            'potential': float(potential),
        }

    def spring_torque(self, theta: float, theta_dot: float = 0.0) -> float:
        """Return spring torque for the selected geometric or equivalent model."""
        p = self.p
        if p.use_geometric_springs:
            return float(self.spring_geometry(theta, theta_dot)['torque'])
        return -p.k_theta * theta - p.k_theta3 * theta**3 - p.c_theta * theta_dot

    def spring_potential(self, theta: float) -> float:
        """Return conservative spring potential, excluding spring damping."""
        p = self.p
        if p.use_geometric_springs:
            return float(self.spring_geometry(theta)['potential'])
        return 0.5 * p.k_theta * theta**2 + 0.25 * p.k_theta3 * theta**4

    def total_mechanical_energy(self, y: np.ndarray) -> float:
        """Return plant kinetic plus conservative mechanical potential energy.

        The energy includes rotor and coupled carriage/resonator kinetic energy,
        belt elasticity, the selected resonator spring potential, incremental
        gravity, and soft-stop potential. It excludes actuator electrical
        storage and dissipative terms, so conservation applies to unforced
        trajectories with damping disabled and zero actuator torque.
        """
        p = self.p
        state = np.asarray(y, dtype=float)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError('Mechanical energy requires a finite seven-state plant vector')
        phi, wm, x, v, theta, omega, _ = map(float, state)
        alpha = p.theta_neutral_world + theta
        total_mass = p.carriage_mass + p.resonator_mass
        coupling = p.resonator_mass * p.lever_com_distance * math.cos(alpha)
        kinetic = (
            0.5 * p.motor_inertia * wm**2
            + 0.5 * total_mass * v**2
            + coupling * v * omega
            + 0.5 * p.resonator_inertia_pivot * omega**2
        )
        delta = p.pulley_radius * phi - x
        belt_potential = 0.5 * p.belt_stiffness * delta**2 + 0.25 * p.belt_cubic * delta**4
        gravity_potential = p.resonator_mass * p.gravity * p.lever_com_distance * (
            math.cos(alpha) - math.cos(p.theta_neutral_world) + theta * math.sin(p.theta_neutral_world)
        )
        stop_excess = max(abs(x) - p.rail_half_travel, 0.0)
        stop_potential = 0.5 * p.stop_stiffness * stop_excess**2
        return float(kinetic + belt_potential + self.spring_potential(theta)
                     + gravity_potential + stop_potential)

    def accelerations(
        self,
        y: np.ndarray,
        tau_cmd: float,
        external_force: float = 0.0,
        external_torque: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """Return phi_dd, x_dd, theta_dd, tau_dot, belt_force, torque_limit."""
        p = self.p
        phi, wm, x, v, th, w, tau_act = map(float, y)
        Fb, _, _ = self.belt_force(y)

        tau_lim = self.motor_torque_limit(wm)
        tau_sat = clamp(tau_cmd, -tau_lim, tau_lim)
        tau_dot = (tau_sat - tau_act) / max(p.motor_torque_time_constant, 1e-6)

        phi_dd = (
            tau_act
            - p.motor_viscous * wm
            - p.motor_coulomb * smooth_sign(wm, p.motor_friction_eps)
            - p.pulley_radius * Fb
        ) / max(p.motor_inertia, 1e-9)

        alpha = p.theta_neutral_world + th
        m = p.resonator_mass
        l = p.lever_com_distance
        M = p.carriage_mass + m
        Jp = p.resonator_inertia_pivot
        B = m * l * math.cos(alpha)

        rhs_x = (
            Fb
            + external_force
            + self.soft_stop_force(x, v)
            - p.b_x * v
            - p.x_coulomb * smooth_sign(v, p.x_friction_eps)
            + m * l * math.sin(alpha) * w * w
        )

        # Incremental gravity about a spring-preloaded neutral pose.
        gravity_increment = m * p.gravity * l * (
            math.sin(alpha) - math.sin(p.theta_neutral_world)
        )
        rhs_theta = (
            external_torque
            - p.theta_coulomb * smooth_sign(w, p.theta_friction_eps)
            + self.spring_torque(th, w)
            + gravity_increment
        )

        det = M * Jp - B * B
        if det <= 1e-12:
            raise RuntimeError("Mass matrix became singular; check mass/inertia parameters.")

        x_dd = (Jp * rhs_x - B * rhs_theta) / det
        th_dd = (-B * rhs_x + M * rhs_theta) / det
        return phi_dd, x_dd, th_dd, tau_dot, Fb, tau_lim

    def derivative(
        self,
        y: np.ndarray,
        tau_cmd: float,
        external_force: float = 0.0,
        external_torque: float = 0.0,
    ) -> np.ndarray:
        phi_dd, x_dd, th_dd, tau_dot, _, _ = self.accelerations(
            y, tau_cmd, external_force, external_torque
        )
        return np.array([y[1], phi_dd, y[3], x_dd, y[5], th_dd, tau_dot], dtype=float)

    def rk4(
        self,
        y: np.ndarray,
        tau_cmd: float,
        dt: float,
        external_force: float = 0.0,
        external_torque: float = 0.0,
    ) -> np.ndarray:
        f = lambda s: self.derivative(s, tau_cmd, external_force, external_torque)
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        out = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return out

    def sensor_values(self, y: np.ndarray, tau_cmd: float, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Synthetic encoder + IMU readings from the current state."""
        p = self.p
        rng = rng or np.random.default_rng(0)
        phi, wm, x, v, th, w, _ = map(float, y)
        _, xdd, thdd, _, _, _ = self.accelerations(y, tau_cmd)

        alpha = p.theta_neutral_world + th
        r = p.imu_radius

        # World coordinates: x right, y up.  Lever vector = [r sin(alpha), r cos(alpha)].
        ax = xdd + r * math.cos(alpha) * thdd - r * math.sin(alpha) * w*w
        ay = -r * math.sin(alpha) * thdd - r * math.cos(alpha) * w*w

        # Accelerometer measures specific force a - g_world; g_world=[0,-g].
        fx = ax
        fy = ay + p.gravity

        # Body axes: longitudinal along lever, tangential in positive theta direction.
        e_long = np.array([math.sin(alpha), math.cos(alpha)])
        e_tan = np.array([math.cos(alpha), -math.sin(alpha)])
        f_world = np.array([fx, fy])
        a_long = float(np.dot(f_world, e_long))
        a_tan = float(np.dot(f_world, e_tan))

        return {
            "motor_as5600": quantize_angle_12bit(phi),
            "lever_as5600": quantize_angle_12bit(th),
            "gyro": w + p.gyro_bias + float(rng.normal(0.0, p.gyro_noise_std)),
            "accel_long": a_long + float(rng.normal(0.0, p.accel_noise_std)),
            "accel_tan": a_tan + float(rng.normal(0.0, p.accel_noise_std)),
            "x_ddot": xdd,
            "theta_ddot": thdd,
        }


# ---------------------------------------------------------------------------
# Motion generators
# ---------------------------------------------------------------------------

class Trajectory:
    def __init__(self, mode: str, mp: MotionParams):
        self.mode = mode
        self.mp = mp
        self.manual_target = 0.0

    def sample(self, t: float) -> Tuple[float, float, float]:
        m = self.mp
        A = m.amplitude
        if self.mode == "hold":
            return 0.0, 0.0, 0.0
        if self.mode == "manual":
            return self.manual_target, 0.0, 0.0
        if self.mode == "step":
            return (0.0, 0.0, 0.0) if t < m.step_time else (A, 0.0, 0.0)
        if self.mode == "sine":
            om = 2.0 * math.pi * m.sine_frequency
            return A*math.sin(om*t), A*om*math.cos(om*t), -A*om*om*math.sin(om*t)
        if self.mode == "aggressive":
            # Repeatable aggressive but feasible point-to-point reversals.
            # Minimum-jerk quintic prevents the benchmark from being dominated by
            # impossible discontinuous position commands.
            P = max(m.aggressive_period, 1e-4)
            k = int(max(t, 0.0) / P)
            local = max(t, 0.0) - k*P
            x0 = 0.0 if k == 0 else (A if ((k-1) % 2 == 0) else -A)
            x1 = A if (k % 2 == 0) else -A
            Tm = max(P * clamp(m.aggressive_move_fraction, 0.2, 0.95), 1e-4)
            if local >= Tm:
                return x1, 0.0, 0.0
            z = clamp(local / Tm, 0.0, 1.0)
            q = 10*z**3 - 15*z**4 + 6*z**5
            qd = (30*z**2 - 60*z**3 + 30*z**4) / Tm
            qdd = (60*z - 180*z**2 + 120*z**3) / (Tm*Tm)
            dx = x1 - x0
            return x0 + dx*q, dx*qd, dx*qdd
        if self.mode == "aggressive_square":
            # Historical discontinuous stress case retained for regression.
            k = int(max(t, 0.0) / max(m.aggressive_period, 1e-4))
            return (A if (k % 2 == 0) else -A), 0.0, 0.0
        if self.mode == "chirp":
            T = max(m.chirp_duration, 1e-3)
            tt = min(max(t, 0.0), T)
            k = (m.chirp_f1 - m.chirp_f0) / T
            phase = 2.0 * math.pi * (m.chirp_f0*tt + 0.5*k*tt*tt)
            om = 2.0 * math.pi * (m.chirp_f0 + k*tt)
            # Approximate acceleration neglecting d(omega)/dt term in amplitude derivative.
            return A*math.sin(phase), A*om*math.cos(phase), -A*om*om*math.sin(phase)
        raise ValueError(f"Unknown trajectory mode: {self.mode}")


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def discrete_linearize(plant: RigPlant, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically linearize one RK4 sample about y=0,u=0."""
    n = 7
    y0 = np.zeros(n)
    u0 = 0.0
    eps_state = np.array([1e-5,1e-4,1e-6,1e-5,1e-6,1e-5,1e-5])
    eps_u = 1e-5
    f0 = plant.rk4(y0, u0, dt)
    A = np.zeros((n,n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps_state[i]
        fp = plant.rk4(y0+e, u0, dt)
        fm = plant.rk4(y0-e, u0, dt)
        A[:,i] = (fp-fm)/(2*eps_state[i])
    fp = plant.rk4(y0, u0+eps_u, dt)
    fm = plant.rk4(y0, u0-eps_u, dt)
    B = ((fp-fm)/(2*eps_u)).reshape(n,1)
    return A, B


def solve_dare_iterative(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int = 5000, tol: float = 1e-11) -> np.ndarray:
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""
    P = Q.copy()
    for _ in range(max_iter):
        BT_P = B.T @ P
        S = R + BT_P @ B
        K = np.linalg.solve(S, BT_P @ A)
        Pn = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(Pn-P)) < tol:
            P = Pn
            break
        P = Pn
    else:
        raise RuntimeError("DARE iteration did not converge")
    return np.linalg.solve(R + B.T@P@B, B.T@P@A)


class Controller:
    """Controller collection with a shared acceleration-domain safety layer.

    Modern controllers separate trajectory tracking, vibration damping, and
    safety projection.  This prevents an active controller from reducing angle
    by simply spending excessive carriage travel.
    """

    def __init__(self, plant: RigPlant, cp: ControllerParams, dt: float):
        self.plant = plant
        self.cp = cp
        self.dt = dt
        self.mode = "servo"
        self.K_legacy: Optional[np.ndarray] = None
        self.K_reduced: Optional[np.ndarray] = None
        self.rl_policy = None
        self.recompute_lqr()

    def recompute_lqr(self) -> None:
        # Only the reduced constrained controller is designed eagerly.
        self._recompute_lqr_reduced()
        self.K_legacy = None

    def _recompute_lqr_legacy(self) -> None:
        A, B = discrete_linearize(self.plant, self.dt)
        cp = self.cp
        Q = np.diag([
            cp.q_phi, cp.q_omega_m, cp.q_x, cp.q_v,
            cp.q_theta, cp.q_theta_dot, cp.q_tau,
        ])
        R = np.array([[cp.r_tau_cmd]])
        try:
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(A, B, Q, R)
            self.K_legacy = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        except Exception:
            try:
                self.K_legacy = solve_dare_iterative(A, B, Q, R)
            except Exception:
                self.K_legacy = None

    def _recompute_lqr_reduced(self) -> None:
        p, cp = self.plant.p, self.cp
        J = max(p.resonator_inertia_pivot, 1e-9)
        Bcoup = p.resonator_mass * p.lever_com_distance * math.cos(p.theta_neutral_world)
        k = p.effective_small_angle_stiffness
        c = p.c_theta
        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -k/J, -c/J],
        ])
        Bc = np.array([[0.0], [1.0], [0.0], [-Bcoup/J]])
        dt = max(cp.lqr_design_dt, 1e-4)
        try:
            from scipy.signal import cont2discrete
            from scipy.linalg import solve_discrete_are
            C = np.eye(4); D = np.zeros((4,1))
            Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, C, D), dt, method="zoh")
            Q = np.diag([cp.lqr_q_x, cp.lqr_q_v, cp.lqr_q_theta, cp.lqr_q_theta_dot])
            R = np.array([[cp.lqr_r_accel]])
            P = solve_discrete_are(Ad, Bd, Q, R)
            self.K_reduced = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        except Exception:
            Ad = np.eye(4) + Ac*dt
            Bd = Bc*dt
            Q = np.diag([cp.lqr_q_x, cp.lqr_q_v, cp.lqr_q_theta, cp.lqr_q_theta_dot])
            R = np.array([[cp.lqr_r_accel]])
            try:
                self.K_reduced = solve_dare_iterative(Ad, Bd, Q, R)
            except Exception:
                self.K_reduced = None

    def feedforward_torque(self, a_ref: float, v_ref: float) -> float:
        p = self.plant.p
        equiv = p.motor_inertia / max(p.pulley_radius,1e-9) + p.pulley_radius * (p.carriage_mass+p.resonator_mass)
        return equiv * a_ref + p.pulley_radius * p.b_x * v_ref

    def tracking_accel(self, y: np.ndarray, ref: Tuple[float,float,float]) -> float:
        cp = self.cp
        xr, vr, ar = ref
        a = ar + cp.track_kp*(xr-float(y[2])) + cp.track_kd*(vr-float(y[3]))
        return clamp(a, -cp.max_accel, cp.max_accel)

    def rail_accel_bounds(self, y: np.ndarray) -> Tuple[float,float]:
        p, cp = self.plant.p, self.cp
        x, v = float(y[2]), float(y[3])
        L = p.rail_half_travel * clamp(cp.rail_soft_fraction, 0.2, 0.99)
        amax = abs(cp.max_accel)
        if x >= L:
            return -amax, -amax
        if x <= -L:
            return amax, amax
        w = max(cp.rail_cbf_omega, 1e-3)
        z = max(cp.rail_cbf_zeta, 0.0)
        lo = -w*w*(L + x) - 2.0*z*w*v
        hi =  w*w*(L - x) - 2.0*z*w*v
        lo = max(lo, -amax)
        hi = min(hi, +amax)
        if lo > hi:
            a = -amax if (x > 0 or (abs(x) < 1e-9 and v > 0)) else amax
            return a, a
        return lo, hi

    def project_accel(self, y: np.ndarray, a_des: float) -> float:
        lo, hi = self.rail_accel_bounds(y)
        return clamp(a_des, lo, hi)

    def torque_from_accel(self, y: np.ndarray, a_cmd: float) -> float:
        p = self.plant.p
        r = max(p.pulley_radius, 1e-9)
        reflected_mass = p.motor_inertia/(r*r)
        M_eff = p.carriage_mass + p.resonator_mass + reflected_mass
        v, wm = float(y[3]), float(y[1])
        F = M_eff*a_cmd + p.b_x*v + p.x_coulomb*smooth_sign(v, p.x_friction_eps)
        tau = r*F + p.motor_viscous*wm + p.motor_coulomb*smooth_sign(wm, p.motor_friction_eps)
        return float(tau)

    def energy_residual_accel(self, y: np.ndarray) -> float:
        cp = self.cp
        a = cp.energy_theta_gain*float(y[4]) + cp.energy_omega_gain*float(y[5])
        return clamp(a, -cp.energy_residual_limit, cp.energy_residual_limit)

    def lqr_residual_accel(self, y: np.ndarray, ref: Tuple[float,float,float]) -> float:
        if self.K_reduced is None:
            return self.energy_residual_accel(y)
        xr, vr, _ = ref
        z = np.array([float(y[2])-xr, float(y[3])-vr, float(y[4]), float(y[5])])
        a = -float((self.K_reduced @ z.reshape(-1,1))[0,0])
        return clamp(a, -self.cp.lqr_residual_limit, self.cp.lqr_residual_limit)

    def modern_accel(self, y: np.ndarray, ref: Tuple[float,float,float], damping: str = "none") -> float:
        a = self.tracking_accel(y, ref)
        if damping == "energy":
            a += self.energy_residual_accel(y)
        elif damping == "lqr":
            a += self.lqr_residual_accel(y, ref)
        elif damping != "none":
            raise ValueError(damping)
        return self.project_accel(y, a)

    def command(self, y: np.ndarray, ref: Tuple[float,float,float]) -> float:
        p, cp = self.plant.p, self.cp
        xr, vr, ar = ref

        if self.mode == "servo":
            F = cp.x_kp*(xr-y[2]) + cp.x_kd*(vr-y[3]) + (p.carriage_mass+p.resonator_mass)*ar
            return p.pulley_radius * F

        if self.mode == "safe_servo":
            return self.torque_from_accel(y, self.modern_accel(y, ref, "none"))

        if self.mode == "energy":
            return self.torque_from_accel(y, self.modern_accel(y, ref, "energy"))

        if self.mode == "lqr":
            return self.torque_from_accel(y, self.modern_accel(y, ref, "lqr"))

        if self.mode == "lqr_legacy":
            if self.K_legacy is None:
                self._recompute_lqr_legacy()
            if self.K_legacy is None:
                return self.command_servo_fallback(y, ref)
            r = p.pulley_radius
            yref = np.array([xr/r, vr/r, xr, vr, 0.0, 0.0, self.feedforward_torque(ar, vr)])
            e = y - yref
            return float(self.feedforward_torque(ar, vr) - (self.K_legacy @ e.reshape(-1,1))[0,0])

        if self.mode == "motor_position":
            phi_ref = xr / p.pulley_radius
            wm_ref = vr / p.pulley_radius
            return 0.22*(phi_ref-y[0]) + 0.004*(wm_ref-y[1]) + self.feedforward_torque(ar,vr)

        if self.mode == "ppo":
            if self.rl_policy is None:
                return self.torque_from_accel(y, self.modern_accel(y, ref, "lqr"))
            if hasattr(self.rl_policy, "residual_accel"):
                base = self.modern_accel(y, ref, "lqr")
                residual = float(self.rl_policy.residual_accel(y, ref, base))
                a = self.project_accel(y, base + residual)
                if hasattr(self.rl_policy, "record_effective_action"):
                    self.rl_policy.record_effective_action(a-base)
                return self.torque_from_accel(y, a)
            return float(self.rl_policy.command(y, ref))

        raise ValueError(self.mode)

    def set_rl_policy(self, policy) -> None:
        self.rl_policy = policy
        if self.rl_policy is not None and hasattr(self.rl_policy, "reset"):
            self.rl_policy.reset()

    def command_servo_fallback(self, y: np.ndarray, ref: Tuple[float,float,float]) -> float:
        old = self.mode
        self.mode = "servo"
        u = self.command(y, ref)
        self.mode = old
        return u


# ---------------------------------------------------------------------------
# Simulator / logging
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, p: PlantParams, cp: ControllerParams, mp: MotionParams, dt: float = 0.0005, control_dt: Optional[float] = None):
        self.p = p
        self.cp = cp
        self.mp = mp
        self.dt = dt
        self.control_dt = control_dt if control_dt is not None else dt
        if dt <= 0: raise ValueError("dt must be positive")
        ratio = self.control_dt/dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("control_dt must be a positive integer multiple of dt")
        self.control_steps = int(round(ratio))
        self._physics_steps = 0
        self._held_command = 0.0
        self.plant = RigPlant(p)
        self.controller = Controller(self.plant, cp, dt)
        self.trajectory = Trajectory("step", mp)
        self.y = np.zeros(7)
        self.t = 0.0
        self.external_torque = 0.0
        self.kick_until = -1.0
        self.kick_torque = 0.025
        self.rng = np.random.default_rng(3)
        self.history = []
        self.log_period = 0.002
        self._next_log_t = 0.0

    def reset(self) -> None:
        self.y[:] = 0.0
        self.t = 0.0
        self.external_torque = 0.0
        self.kick_until = -1.0
        self._physics_steps = 0
        self._held_command = 0.0
        if self.controller.rl_policy is not None:
            self.controller.rl_policy.reset()
        self.history.clear()
        self._next_log_t = 0.0

    def kick(self, torque: float = 0.025, duration: float = 0.06) -> None:
        self.kick_torque = torque
        self.kick_until = self.t + duration

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            ref = self.trajectory.sample(self.t)
            if self._physics_steps % self.control_steps == 0:
                self._held_command = self.controller.command(self.y, ref)
            u = self._held_command
            self._physics_steps += 1
            ext_tau = self.kick_torque if self.t < self.kick_until else 0.0
            self.y = self.plant.rk4(self.y, u, self.dt, external_torque=ext_tau)
            self.t += self.dt
            if self.t + 1e-12 >= self._next_log_t:
                sensors = self.plant.sensor_values(self.y, u, self.rng)
                Fb, delta, _ = self.plant.belt_force(self.y)
                self.history.append({
                    "t": self.t,
                    "phi_m": self.y[0], "omega_m": self.y[1],
                    "x": self.y[2], "x_dot": self.y[3],
                    "theta": self.y[4], "theta_dot": self.y[5],
                    "tau_act": self.y[6], "tau_cmd": u,
                    "belt_force": Fb, "belt_extension": delta,
                    **sensors,
                })
                self._next_log_t += self.log_period

    def export_csv(self, path: Path) -> None:
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)


# ---------------------------------------------------------------------------
# Interactive 2D renderer
# ---------------------------------------------------------------------------

def run_interactive(sim: Simulator) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.widgets import Button, RadioButtons, Slider

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("Active vibration rig — parametric 2D model")

    ax = fig.add_axes([0.05, 0.46, 0.60, 0.49])
    ax_th = fig.add_axes([0.69, 0.69, 0.28, 0.25])
    ax_x = fig.add_axes([0.69, 0.39, 0.28, 0.25])

    p = sim.p
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.13, 0.13)
    ax.set_ylim(-0.035, 0.18)
    ax.set_xlabel("horizontal position [m]")
    ax.set_ylabel("height [m]")
    ax.grid(True, alpha=0.2)

    # Static geometry
    rail_y = 0.0
    ax.plot([-p.rail_half_travel-0.015, p.rail_half_travel+0.015], [rail_y,rail_y], lw=5, alpha=0.45)
    ax.plot([-p.rail_half_travel, p.rail_half_travel], [-0.012,-0.012], lw=2, alpha=0.35)
    ax.axvline(-p.rail_half_travel, ls="--", lw=1)
    ax.axvline(+p.rail_half_travel, ls="--", lw=1)

    motor_center = (-0.105, 0.035)
    motor = Circle(motor_center, 0.018, fill=False, lw=2)
    ax.add_patch(motor)
    ax.text(motor_center[0], motor_center[1], "NEMA\n17", ha="center", va="center", fontsize=9)
    pulley = Circle((-0.082, 0.015), 0.010, fill=False, lw=2)
    idler = Circle((0.082, 0.015), 0.010, fill=False, lw=2)
    ax.add_patch(pulley); ax.add_patch(idler)
    ax.plot([-0.082,0.082],[0.025,0.025], lw=1.8)
    ax.plot([-0.082,0.082],[0.005,0.005], lw=1.8)
    ax.text(-0.105, 0.063, "AS5600", ha="center", fontsize=8)

    carriage_w, carriage_h = 0.028, 0.015
    carriage = Rectangle((-carriage_w/2, rail_y), carriage_w, carriage_h, fill=False, lw=2)
    ax.add_patch(carriage)
    pivot_dot, = ax.plot([], [], "o", ms=6)
    lever_line, = ax.plot([], [], lw=3)
    mass_circle = Circle((0,0), 0.010, fill=False, lw=2)
    ax.add_patch(mass_circle)
    imu_rect = Rectangle((0,0), 0.012,0.007, fill=False, lw=1.5)
    ax.add_patch(imu_rect)
    spring_line, = ax.plot([], [], lw=1.5)
    target_line = ax.axvline(0.0, ls=":", lw=1.5)
    info = ax.text(0.02,0.97,"",transform=ax.transAxes,va="top",fontsize=9,family="monospace")

    # Traces
    th_line, = ax_th.plot([], [], label="theta [deg]")
    ref0 = ax_th.axhline(0, lw=1, alpha=0.5)
    ax_th.set_ylabel("theta [deg]"); ax_th.set_xlabel("t [s]"); ax_th.grid(True,alpha=0.25)
    x_line, = ax_x.plot([], [], label="x [mm]")
    xr_line, = ax_x.plot([], [], ls="--", label="x_ref [mm]")
    ax_x.set_ylabel("x [mm]"); ax_x.set_xlabel("t [s]"); ax_x.grid(True,alpha=0.25); ax_x.legend(loc="upper right",fontsize=8)

    # Controls
    radio_traj_ax = fig.add_axes([0.69,0.08,0.12,0.26])
    radio_ctrl_ax = fig.add_axes([0.83,0.08,0.13,0.18])
    radio_traj = RadioButtons(radio_traj_ax, ("hold","step","sine","chirp","aggressive","aggressive_square","manual"), active=1)
    radio_ctrl = RadioButtons(radio_ctrl_ax, ("servo","safe_servo","energy","lqr","lqr_legacy","motor_position"), active=0)
    radio_traj_ax.set_title("trajectory",fontsize=10)
    radio_ctrl_ax.set_title("controller",fontsize=10)

    slider_specs = [
        ("k_theta", 0.15, 1.20, p.k_theta),
        ("c_theta", 0.0, 0.035, p.c_theta),
        ("m_res", 0.04, 0.30, p.resonator_mass),
        ("belt_k", 1000.0, 16000.0, p.belt_stiffness),
        ("amp", 0.005, min(0.055,p.rail_half_travel*0.90), sim.mp.amplitude),
    ]
    sliders = {}
    y0 = 0.31
    for i,(name,lo,hi,val) in enumerate(slider_specs):
        sa = fig.add_axes([0.08, y0-i*0.052, 0.49, 0.028])
        sliders[name] = Slider(sa,name,lo,hi,valinit=val)

    btn_pause_ax = fig.add_axes([0.59,0.31,0.075,0.04])
    btn_reset_ax = fig.add_axes([0.59,0.255,0.075,0.04])
    btn_kick_ax = fig.add_axes([0.59,0.200,0.075,0.04])
    btn_csv_ax = fig.add_axes([0.59,0.145,0.075,0.04])
    b_pause = Button(btn_pause_ax,"Pause")
    b_reset = Button(btn_reset_ax,"Reset")
    b_kick = Button(btn_kick_ax,"Kick")
    b_csv = Button(btn_csv_ax,"CSV")

    paused = {"v":False}

    def refresh_params(_=None):
        sim.p.k_theta = float(sliders["k_theta"].val)
        sim.p.c_theta = float(sliders["c_theta"].val)
        sim.p.resonator_mass = float(sliders["m_res"].val)
        sim.p.belt_stiffness = float(sliders["belt_k"].val)
        sim.mp.amplitude = float(sliders["amp"].val)
        sim.controller.recompute_lqr()

    for s in sliders.values(): s.on_changed(refresh_params)

    def set_traj(label):
        sim.trajectory.mode = label
    def set_ctrl(label):
        sim.controller.mode = label
    radio_traj.on_clicked(set_traj)
    radio_ctrl.on_clicked(set_ctrl)

    def pause(_):
        paused["v"] = not paused["v"]
        b_pause.label.set_text("Play" if paused["v"] else "Pause")
    b_pause.on_clicked(pause)
    b_reset.on_clicked(lambda _:(sim.reset(), None))
    b_kick.on_clicked(lambda _:sim.kick())

    def export(_):
        path = Path(f"rig_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        sim.export_csv(path)
        print(f"wrote {path.resolve()}")
    b_csv.on_clicked(export)

    def on_click(event):
        if event.inaxes is ax and event.xdata is not None:
            sim.trajectory.mode = "manual"
            sim.trajectory.manual_target = clamp(float(event.xdata), -sim.p.rail_half_travel*0.9, sim.p.rail_half_travel*0.9)
            radio_traj.set_active(5)
    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_key(event):
        if event.key == " ": pause(None)
        elif event.key in ("r","R"): sim.reset()
        elif event.key == "left":
            sim.trajectory.mode="manual"; sim.trajectory.manual_target=clamp(sim.trajectory.manual_target-0.005,-sim.p.rail_half_travel*0.9,sim.p.rail_half_travel*0.9)
        elif event.key == "right":
            sim.trajectory.mode="manual"; sim.trajectory.manual_target=clamp(sim.trajectory.manual_target+0.005,-sim.p.rail_half_travel*0.9,sim.p.rail_half_travel*0.9)
        elif event.key in ("k","K"): sim.kick()
    fig.canvas.mpl_connect("key_press_event", on_key)

    def draw_spring(x0,y0,x1,y1,n=7,amp=0.004):
        # zigzag spring in arbitrary orientation
        dx,dy=x1-x0,y1-y0; L=math.hypot(dx,dy)
        if L<1e-9: return [x0,x1],[y0,y1]
        tx,ty=dx/L,dy/L; nx,ny=-ty,tx
        pts=[]
        for i in range(n*2+1):
            s=i/(n*2)
            off=0 if i in (0,n*2) else (amp if i%2 else -amp)
            pts.append((x0+s*dx+off*nx,y0+s*dy+off*ny))
        return [q[0] for q in pts],[q[1] for q in pts]

    def update(_frame):
        if not paused["v"]:
            # ~60 Hz display, 0.5 ms integration -> 33 substeps/frame
            sim.step(max(1,int(round((1/60)/sim.dt))))

        p = sim.p; y=sim.y
        phi,wm,x,v,th,w,tau_act = y
        ref = sim.trajectory.sample(sim.t)
        alpha = p.theta_neutral_world + th

        carriage.set_xy((x-carriage_w/2,rail_y))
        pivot=(x,rail_y+carriage_h)
        tip=(pivot[0]+p.lever_com_distance*math.sin(alpha), pivot[1]+p.lever_com_distance*math.cos(alpha))
        pivot_dot.set_data([pivot[0]],[pivot[1]])
        lever_line.set_data([pivot[0],tip[0]],[pivot[1],tip[1]])
        mass_circle.center=tip
        imu_r=p.imu_radius
        imu_pt=(pivot[0]+imu_r*math.sin(alpha),pivot[1]+imu_r*math.cos(alpha))
        imu_rect.set_xy((imu_pt[0]-0.006,imu_pt[1]-0.0035))
        sx,sy=draw_spring(x-0.012,rail_y+carriage_h*0.75, pivot[0]+0.045*math.sin(alpha), pivot[1]+0.045*math.cos(alpha))
        spring_line.set_data(sx,sy)
        target_line.set_xdata([ref[0],ref[0]])

        sensor=sim.plant.sensor_values(y, sim.controller.command(y,ref), sim.rng)
        info.set_text(
            f"t={sim.t:6.3f} s\n"
            f"x={x*1e3:+7.2f} mm   v={v:+6.3f} m/s\n"
            f"theta={math.degrees(th):+7.2f} deg   omega={math.degrees(w):+7.1f} deg/s\n"
            f"tau_act={tau_act:+6.3f} N m\n"
            f"k_eff={p.effective_small_angle_stiffness:+6.3f} N m/rad\n"
            f"gyro={math.degrees(sensor['gyro']):+7.2f} deg/s\n"
            f"IMU tan={sensor['accel_tan']:+6.2f} m/s2"
        )

        if sim.history:
            h=sim.history
            t0=max(0.0,sim.t-8.0)
            rows=[r for r in h if r["t"]>=t0]
            ts=np.array([r["t"] for r in rows])
            ths=np.degrees([r["theta"] for r in rows])
            xs=1000*np.array([r["x"] for r in rows])
            xrs=1000*np.array([sim.trajectory.sample(float(tt))[0] for tt in ts])
            th_line.set_data(ts,ths); x_line.set_data(ts,xs); xr_line.set_data(ts,xrs)
            ax_th.set_xlim(t0,max(t0+0.2,sim.t)); ax_x.set_xlim(t0,max(t0+0.2,sim.t))
            thlim=max(3.0,float(np.max(np.abs(ths)))+2.0); ax_th.set_ylim(-thlim,thlim)
            xlim=max(10.0,float(np.max(np.abs(np.r_[xs,xrs])))+5.0); ax_x.set_ylim(-xlim,xlim)
        return carriage,pivot_dot,lever_line,mass_circle,imu_rect,spring_line,target_line,info,th_line,x_line,xr_line

    ani = FuncAnimation(fig, update, interval=1000/60, blit=False, cache_frame_data=False)
    fig._rig_animation = ani  # keep alive
    plt.show()


# ---------------------------------------------------------------------------
# Headless simulation
# ---------------------------------------------------------------------------

def run_headless(sim: Simulator, duration: float, csv_path: Optional[Path], frame_path: Optional[Path]) -> None:
    steps = int(math.ceil(duration/sim.dt))
    for _ in range(steps): sim.step(1)
    if csv_path:
        sim.export_csv(csv_path)
        print(f"wrote {csv_path}")
    if frame_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p=sim.p; y=sim.y
        fig,ax=plt.subplots(figsize=(10,5))
        ax.set_aspect("equal"); ax.grid(True,alpha=0.2)
        ax.set_xlim(-0.13,0.13); ax.set_ylim(-0.035,0.18)
        rail_y=0.0
        ax.plot([-p.rail_half_travel,p.rail_half_travel],[rail_y,rail_y],lw=5,alpha=0.4)
        x,th=y[2],y[4]; alpha=p.theta_neutral_world+th
        cw,ch=0.028,0.015
        ax.add_patch(plt.Rectangle((x-cw/2,rail_y),cw,ch,fill=False,lw=2))
        pivot=(x,rail_y+ch)
        tip=(pivot[0]+p.lever_com_distance*math.sin(alpha),pivot[1]+p.lever_com_distance*math.cos(alpha))
        ax.plot([pivot[0],tip[0]],[pivot[1],tip[1]],lw=3)
        ax.add_patch(plt.Circle(tip,0.010,fill=False,lw=2))
        ax.set_title(f"t={sim.t:.3f}s, x={x*1e3:.1f}mm, theta={math.degrees(th):.1f}deg")
        fig.savefig(frame_path,dpi=180,bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {frame_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params",type=Path,help="JSON file overriding PlantParams")
    ap.add_argument("--dump-defaults",type=Path,help="write default parameter JSON and exit")
    ap.add_argument("--controller",choices=["servo","safe_servo","energy","lqr","lqr_legacy","motor_position","ppo"],default="servo")
    ap.add_argument("--trajectory",choices=["hold","step","sine","chirp","aggressive","aggressive_square","manual"],default="step")
    ap.add_argument("--dt",type=float,default=0.0005,help="integration timestep [s]")
    ap.add_argument("--headless",type=float,metavar="SECONDS",help="run without GUI for N seconds")
    ap.add_argument("--csv",type=Path,help="CSV output path in headless mode")
    ap.add_argument("--frame",type=Path,help="PNG final-frame output path in headless mode")
    return ap.parse_args()


def main() -> None:
    args=parse_args()
    p=PlantParams()
    if args.params:
        p=PlantParams.from_dict(json.loads(args.params.read_text()))
    if args.dump_defaults:
        args.dump_defaults.write_text(json.dumps(asdict(p),indent=2))
        print(f"wrote {args.dump_defaults}")
        return
    cp=ControllerParams(); mp=MotionParams()
    sim=Simulator(p,cp,mp,dt=args.dt)
    sim.controller.mode=args.controller
    sim.trajectory.mode=args.trajectory
    if args.headless is not None:
        run_headless(sim,args.headless,args.csv,args.frame)
    else:
        run_interactive(sim)


if __name__ == "__main__":
    main()

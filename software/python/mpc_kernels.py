"""Optional Numba CPU kernels for repeated MPC plant rollouts."""
import math

import numpy as np

try:
    from numba import njit
except ImportError:  # Keep the NumPy implementation available without Numba.
    njit = None


NUMBA_AVAILABLE = njit is not None


def plant_kernel_parameters(plant_params):
    """Pack the plant constants used by the scalar RK4 kernel."""
    p = plant_params
    values = np.asarray((
        p.carriage_mass,
        p.resonator_mass,
        p.lever_com_distance,
        p.resonator_inertia_pivot,
        p.theta_neutral_world,
        p.k_theta,
        p.k_theta3,
        p.c_theta,
        p.theta_coulomb,
        p.theta_friction_eps,
        p.b_x,
        p.x_coulomb,
        p.x_friction_eps,
        p.motor_inertia,
        p.motor_viscous,
        p.motor_coulomb,
        p.motor_friction_eps,
        p.pulley_radius,
        p.belt_stiffness,
        p.belt_damping,
        p.belt_cubic,
        p.motor_hold_torque,
        p.motor_omega_corner,
        p.motor_torque_time_constant,
        p.rail_half_travel,
        p.stop_stiffness,
        p.stop_damping,
        p.gravity,
        p.spring_anchor_half_spacing,
        p.spring_anchor_y,
        p.spring_shaft_radius,
        p.spring_k,
        p.spring_d,
        p.spring_free_length,
    ), dtype=np.float64)
    return values, bool(p.use_geometric_springs)


def _derivative_scalar(state, torque_command, external_force, external_torque,
                       p, use_geometric_springs):
    phi, motor_speed, x, velocity, theta, angular_rate, motor_torque = state
    pulley_radius = p[17]
    delta = pulley_radius * phi - x
    delta_rate = pulley_radius * motor_speed - velocity
    belt_force = p[18] * delta + p[19] * delta_rate + p[20] * delta**3

    torque_limit = p[21] / math.sqrt(
        1.0 + (abs(motor_speed) / max(p[22], 1e-6))**2)
    saturated_torque = min(max(torque_command, -torque_limit), torque_limit)
    motor_torque_rate = (saturated_torque - motor_torque) / max(p[23], 1e-6)
    motor_friction = p[15] * math.tanh(motor_speed / max(abs(p[16]), 1e-9))
    motor_accel = (motor_torque - p[14] * motor_speed - motor_friction
                   - pulley_radius * belt_force) / max(p[13], 1e-9)

    if x > p[24]:
        stop_force = -p[25] * (x - p[24]) - p[26] * max(velocity, 0.0)
    elif x < -p[24]:
        stop_force = -p[25] * (x + p[24]) - p[26] * min(velocity, 0.0)
    else:
        stop_force = 0.0

    carriage_friction = p[11] * math.tanh(velocity / max(abs(p[12]), 1e-9))
    alpha = p[4] + theta
    coupling = p[1] * p[2] * math.cos(alpha)
    rhs_x = (belt_force + external_force + stop_force - p[10] * velocity
             - carriage_friction + p[1] * p[2] * math.sin(alpha) * angular_rate**2)
    angular_friction = p[8] * math.tanh(angular_rate / max(abs(p[9]), 1e-9))
    gravity_increment = p[1] * p[27] * p[2] * (
        math.sin(alpha) - math.sin(p[4]))

    if use_geometric_springs:
        shaft_x = p[30] * math.sin(alpha)
        shaft_y = p[30] * math.cos(alpha)
        shaft_vx = p[30] * math.cos(alpha) * angular_rate
        shaft_vy = -p[30] * math.sin(alpha) * angular_rate
        spring_torque = 0.0
        for anchor_x in (-p[28], p[28]):
            dx = shaft_x - anchor_x
            dy = shaft_y - p[29]
            length = math.hypot(dx, dy)
            safe_length = max(length, 1e-12)
            length_rate = (dx * shaft_vx + dy * shaft_vy) / safe_length
            magnitude = p[31] * (length - p[33]) + p[32] * length_rate
            force_x = -magnitude * dx / safe_length
            force_y = -magnitude * dy / safe_length
            spring_torque += (force_x * p[30] * math.cos(alpha)
                              - force_y * p[30] * math.sin(alpha))
    else:
        spring_torque = -p[5] * theta - p[6] * theta**3 - p[7] * angular_rate

    rhs_theta = external_torque - angular_friction + spring_torque + gravity_increment
    total_mass = p[0] + p[1]
    determinant = total_mass * p[3] - coupling**2
    if determinant <= 1e-12:
        raise RuntimeError('Mass matrix became singular; check MPC kernel parameters.')
    carriage_accel = (p[3] * rhs_x - coupling * rhs_theta) / determinant
    angular_accel = (-coupling * rhs_x + total_mass * rhs_theta) / determinant
    return np.asarray((motor_speed, motor_accel, velocity, carriage_accel,
                       angular_rate, angular_accel, motor_torque_rate))


def _rk4_batch_kernel(states, torque_commands, parameters, use_geometric_springs,
                      physics_dt, substeps, time_s, kick_torque, kick_at,
                      kick_duration):
    y = states.copy()
    batch_size = len(y)
    for step in range(substeps):
        now = time_s + step * physics_dt
        ext_torque = (kick_torque if kick_at >= 0.0
                      and kick_at <= now < kick_at + kick_duration else 0.0)
        k1 = np.empty_like(y)
        for row in range(batch_size):
            k1[row] = _derivative_scalar(
                y[row], torque_commands[row], 0.0, ext_torque,
                parameters, use_geometric_springs)
        stage = y + 0.5 * physics_dt * k1
        k2 = np.empty_like(y)
        for row in range(batch_size):
            k2[row] = _derivative_scalar(
                stage[row], torque_commands[row], 0.0, ext_torque,
                parameters, use_geometric_springs)
        stage = y + 0.5 * physics_dt * k2
        k3 = np.empty_like(y)
        for row in range(batch_size):
            k3[row] = _derivative_scalar(
                stage[row], torque_commands[row], 0.0, ext_torque,
                parameters, use_geometric_springs)
        stage = y + physics_dt * k3
        k4 = np.empty_like(y)
        for row in range(batch_size):
            k4[row] = _derivative_scalar(
                stage[row], torque_commands[row], 0.0, ext_torque,
                parameters, use_geometric_springs)
        y += physics_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


if NUMBA_AVAILABLE:
    _derivative_scalar = njit(cache=True)(_derivative_scalar)
    _rk4_batch_kernel = njit(cache=True)(_rk4_batch_kernel)


def torque_transition_batch(states, torque_commands, parameters,
                            use_geometric_springs, physics_dt, substeps,
                            time_s, kick_torque, kick_at, kick_duration):
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is not available')
    return _rk4_batch_kernel(
        np.ascontiguousarray(states, dtype=np.float64),
        np.ascontiguousarray(torque_commands, dtype=np.float64),
        parameters, use_geometric_springs, physics_dt, substeps,
        time_s, kick_torque, kick_at, kick_duration)

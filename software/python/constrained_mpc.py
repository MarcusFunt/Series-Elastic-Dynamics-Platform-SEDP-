"""Actuator- and command-timing-aware constrained MPC reference.

The QP is a sequential local linearization of the active plant and command
queue. A nonlinear rollout through the same torque or STEP/DIR model accepts
the first action. Solver failures use a bounded zero-residual LQR fallback;
this remains a simulation reference, not a certified hardware safety control.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy import sparse

from active_vibration_rig_2d import (
    Controller, RigPlant, StepDirActuator, StepDirParams, clamp,
)
from state_estimator import held_transition, numerical_jacobian
from timing_models import CommandDelayQueue
from mpc_kernels import NUMBA_AVAILABLE, plant_kernel_parameters, torque_transition_batch
try:
    import osqp
except ImportError:
    osqp = None


@dataclass
class MPCConfig:
    # Eight 10 ms samples target the controller period; current default latency
    # has not been measured after the later CPU optimization.
    # Longer horizons remain available for offline/reference use.
    horizon: int = 8
    physics_dt: float = .001
    control_dt: float = .01
    residual_accel_limit: float = 2.5
    max_iterations: int = 60
    qp_max_iterations: int = 1000
    solver_backend: str = 'auto'
    linearization_stride: int = 1
    time_limit_seconds: float | None = None
    tracking_scale: float = .006
    velocity_scale: float = .35
    angle_scale: float = math.radians(5.)
    angular_rate_scale: float = math.radians(100.)
    position_weight: float = 1.8
    velocity_weight: float = .12
    angle_weight: float = 2.4
    angular_rate_weight: float = .55
    terminal_multiplier: float = 2.
    action_weight: float = .03
    slew_weight: float = .06
    actuator_mode: str = 'torque'
    step_dir: StepDirParams = None
    command_delay: float = 0.
    command_jitter: float = 0.
    command_seed: int = 0

    def __post_init__(self):
        if (self.horizon < 2 or self.residual_accel_limit <= 0
                or self.max_iterations < 1 or self.qp_max_iterations < 1
                or self.linearization_stride < 1):
            raise ValueError('Invalid MPC configuration')
        if self.solver_backend not in ('auto', 'slsqp', 'osqp'):
            raise ValueError("solver_backend must be 'auto', 'slsqp', or 'osqp'")
        if (self.physics_dt <= 0 or self.control_dt < self.physics_dt
                or not np.isclose(self.control_dt / self.physics_dt,
                                  round(self.control_dt / self.physics_dt))):
            raise ValueError('MPC timing must have an integer substep ratio')
        if (self.time_limit_seconds is not None
                and (not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds < 0)):
            raise ValueError('time_limit_seconds must be finite, nonnegative, or None')
        if self.actuator_mode not in ('torque', 'step_dir'):
            raise ValueError("actuator_mode must be 'torque' or 'step_dir'")
        if isinstance(self.step_dir, dict):
            self.step_dir = StepDirParams(**self.step_dir)
        elif self.step_dir is None:
            self.step_dir = StepDirParams()
        elif not isinstance(self.step_dir, StepDirParams):
            raise ValueError('step_dir must be StepDirParams or a mapping')
        if (not math.isfinite(self.command_delay) or not math.isfinite(self.command_jitter)
                or min(self.command_delay, self.command_jitter) < 0):
            raise ValueError('command delay and jitter must be finite and nonnegative')
        weights = (self.position_weight, self.velocity_weight, self.angle_weight,
                   self.angular_rate_weight, self.terminal_multiplier,
                   self.action_weight, self.slew_weight)
        if not all(math.isfinite(v) and v >= 0 for v in weights):
            raise ValueError('MPC objective weights must be finite and nonnegative')
        scales=(self.tracking_scale,self.velocity_scale,self.angle_scale,self.angular_rate_scale,
                self.residual_accel_limit)
        if not all(math.isfinite(v) and v > 0 for v in scales):
            raise ValueError('MPC objective scales and residual authority must be finite and positive')


class _DeadlineReached(Exception):
    pass


class ConstrainedMPC:
    def __init__(self, params, controller_params, cfg=None):
        self.p = params
        self.cp = controller_params
        self.cfg = cfg or MPCConfig()
        self.plant = RigPlant(params)
        self.controller = Controller(self.plant, controller_params, self.cfg.physics_dt)
        self.previous_accel = 0.
        self.warm = np.zeros(self.cfg.horizon)
        self.diagnostics = {}
        self._kernel_parameters, self._kernel_geometric_springs = plant_kernel_parameters(params)
        self._numba_kernel_enabled = False
        self._numba_kernel_error = ''
        if NUMBA_AVAILABLE and self.cfg.actuator_mode == 'torque':
            try:
                torque_transition_batch(
                    np.zeros((1, 7)), np.zeros(1), self._kernel_parameters,
                    self._kernel_geometric_springs, self.cfg.physics_dt,
                    round(self.cfg.control_dt / self.cfg.physics_dt),
                    0., 0., -1., 0.)
            except Exception as exc:
                self._numba_kernel_error = str(exc)
            else:
                self._numba_kernel_enabled = True

    def _integrate_torque_batch(self, states, torques, *, time_s=0.,
                                kick_torque=0., kick_at=-1., kick_duration=0.,
                                deadline=None):
        self._check_deadline(deadline)
        if self._numba_kernel_enabled:
            try:
                result = torque_transition_batch(
                    states, torques, self._kernel_parameters,
                    self._kernel_geometric_springs, self.cfg.physics_dt,
                    round(self.cfg.control_dt / self.cfg.physics_dt),
                    time_s, kick_torque, kick_at, kick_duration)
                self._check_deadline(deadline)
                return result
            except Exception as exc:
                if isinstance(exc, _DeadlineReached):
                    raise
                self._numba_kernel_enabled = False
                self._numba_kernel_error = str(exc)

        y = np.asarray(states, dtype=float).copy()
        for i in range(round(self.cfg.control_dt / self.cfg.physics_dt)):
            self._check_deadline(deadline)
            now = time_s + i * self.cfg.physics_dt
            ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at + kick_duration else 0.
            y = self.plant.rk4_batch(y, torques, self.cfg.physics_dt,
                                     external_torque=ext)
        self._check_deadline(deadline)
        return y

    def _transition(self, y, accel):
        """Legacy-shaped torque transition retained for callers and diagnostics."""
        applied = self.controller.project_accel(y, float(accel))
        torque = self.controller.torque_from_accel(y, applied)
        return held_transition(self.plant, y, torque, self.cfg.physics_dt,
                               self.cfg.control_dt)

    def _transition_active(self, y, accel, step_dir_actuator=None, *, time_s=0.,
                           kick_torque=0., kick_at=-1., kick_duration=0.,
                           deadline=None):
        """Run one exact control interval and return state plus actuator copy.

        ``accel`` is the queued value before the environment's application-time
        rail projection. STEP/DIR state is cloned so prediction never mutates
        the live environment actuator.
        """
        y = np.asarray(y, dtype=float).copy()
        applied = self.controller.project_accel(y, float(accel))
        nsteps = round(self.cfg.control_dt / self.cfg.physics_dt)
        actuator = None
        if self.cfg.actuator_mode == 'torque':
            torque = self.controller.torque_from_accel(y, applied)
            y = self._integrate_torque_batch(
                y.reshape(1, 7), np.asarray([torque]), time_s=time_s,
                kick_torque=kick_torque, kick_at=kick_at,
                kick_duration=kick_duration, deadline=deadline)[0]
            return y, None

        actuator = deepcopy(step_dir_actuator) if step_dir_actuator is not None else StepDirActuator(self.cfg.step_dir)
        if actuator.params != self.cfg.step_dir:
            raise ValueError('STEP/DIR actuator parameters do not match MPC configuration')
        max_carriage_speed = self.cfg.step_dir.max_velocity_rad_s * self.p.pulley_radius
        target_carriage_velocity = clamp(
            float(y[3]) + applied * self.cfg.control_dt,
            -max_carriage_speed, max_carriage_speed)
        requested_motor_velocity = target_carriage_velocity / max(self.p.pulley_radius, 1e-12)
        actuator.command(requested_motor_velocity)
        for i in range(nsteps):
            self._check_deadline(deadline)
            now = time_s + i * self.cfg.physics_dt
            ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at + kick_duration else 0.
            diag = actuator.advance(
                self.cfg.physics_dt, float(y[0]), float(y[1]),
                self.plant.motor_torque_limit(float(y[1])))
            y = self.plant.rk4(y, diag['torque_command_nm'], self.cfg.physics_dt,
                               external_torque=ext)
        self._check_deadline(deadline)
        return y, actuator

    def _augmented_state(self, y, actuator):
        y = np.asarray(y, dtype=float)
        if self.cfg.actuator_mode == 'torque':
            return y.copy()
        actuator = actuator or StepDirActuator(self.cfg.step_dir)
        return np.r_[y, actuator.velocity_rad_s, actuator.position_rad]

    def _transition_augmented(self, z, accel, actuator_template, *, time_s=0.,
                              kick_torque=0., kick_at=-1., kick_duration=0.,
                              deadline=None):
        z = np.asarray(z, dtype=float)
        if self.cfg.actuator_mode == 'torque':
            return self._transition_active(z[:7], accel, time_s=time_s,
                                           kick_torque=kick_torque, kick_at=kick_at,
                                           kick_duration=kick_duration,
                                           deadline=deadline)[0]
        actuator = deepcopy(actuator_template) if actuator_template is not None else StepDirActuator(self.cfg.step_dir)
        actuator.velocity_rad_s = float(z[7])
        actuator.position_rad = float(z[8])
        actuator.step_count = int(round(actuator.position_rad / actuator.params.step_angle_rad))
        actuator.step_phase_rad = (actuator.position_rad
                                   - actuator.step_count * actuator.params.step_angle_rad)
        y_next, actuator_next = self._transition_active(
            z[:7], accel, actuator, time_s=time_s, kick_torque=kick_torque,
            kick_at=kick_at, kick_duration=kick_duration, deadline=deadline)
        return self._augmented_state(y_next, actuator_next)

    def _transition_batch_torque(self, states, accelerations, *, time_s=0.,
                                 kick_torque=0., kick_at=-1., kick_duration=0.,
                                 deadline=None):
        """Advance torque-mode perturbations together for finite differences."""
        self._check_deadline(deadline)
        states = np.asarray(states, dtype=float)
        accelerations = np.asarray(accelerations, dtype=float)
        if (states.ndim != 2 or states.shape[1] != 7
                or accelerations.shape != (len(states),)):
            raise ValueError('Batched torque transition requires (n, 7) states and (n,) accelerations')

        applied = np.asarray([
            self.controller.project_accel(state, float(accel))
            for state, accel in zip(states, accelerations)
        ])
        torques = np.asarray([
            self.controller.torque_from_accel(state, float(accel))
            for state, accel in zip(states, applied)
        ])
        return self._integrate_torque_batch(
            states, torques, time_s=time_s, kick_torque=kick_torque,
            kick_at=kick_at, kick_duration=kick_duration, deadline=deadline)

    def _linearize_torque_transition(self, z, accel, *, time_s=0., kick_torque=0.,
                                     kick_at=-1., kick_duration=0., deadline=None):
        """Compute transition Jacobians in one batched RK4 rollout."""
        self._check_deadline(deadline)
        z = np.asarray(z, dtype=float)
        points, inputs, epsilons = [], [], []
        for i in range(len(z)):
            eps = 1e-5 * max(1., abs(float(z[i])))
            delta = np.zeros_like(z)
            delta[i] = eps
            points.extend((z + delta, z - delta))
            inputs.extend((accel, accel))
            epsilons.append(eps)

        input_eps = 1e-4
        points.extend((z, z))
        inputs.extend((accel + input_eps, accel - input_eps))
        outputs = self._transition_batch_torque(
            np.asarray(points), np.asarray(inputs), time_s=time_s,
            kick_torque=kick_torque, kick_at=kick_at,
            kick_duration=kick_duration, deadline=deadline)
        self._check_deadline(deadline)
        F = np.column_stack([
            (outputs[2*i] - outputs[2*i+1]) / (2. * eps)
            for i, eps in enumerate(epsilons)
        ])
        G = (outputs[-2] - outputs[-1]) / (2. * input_eps)
        return F, G

    @staticmethod
    def _sparse_numerical_jacobian(function, point, columns):
        """Finite-difference only state coordinates used by a scalar model."""
        point = np.asarray(point, dtype=float)
        jacobian = np.zeros(len(point))
        for i in columns:
            eps = 1e-5 * max(1., abs(float(point[i])))
            delta = np.zeros_like(point)
            delta[i] = eps
            plus = float(np.asarray(function(point + delta)).reshape(-1)[0])
            minus = float(np.asarray(function(point - delta)).reshape(-1)[0])
            jacobian[i] = (plus - minus) / (2. * eps)
        return jacobian

    def _command_source_plan(self, command_queue, time_s):
        """Map each predicted control interval to its issuing decision or a held value."""
        n = self.cfg.horizon
        queue = (deepcopy(command_queue) if command_queue is not None else
                 CommandDelayQueue(self.cfg.command_delay, self.cfg.command_jitter,
                                   self.cfg.command_seed))
        markers = [1e4 + 17. * k for k in range(n)]
        marker_to_index = {round(value, 9): k for k, value in enumerate(markers)}
        sources, fixed = [], []
        for k, marker in enumerate(markers):
            now = time_s + k * self.cfg.control_dt
            timing = queue.issue(marker, now)
            active = round(float(timing['command_applied']), 9)
            source = marker_to_index.get(active)
            sources.append(source)
            fixed.append(0. if source is not None else float(timing['command_applied']))
        return sources, fixed

    def action(self, state, references, **kwargs):
        """Return a bounded action and always retain solver/model diagnostics."""
        y = np.asarray(state, dtype=float)
        refs = np.asarray(references, dtype=float)
        n = self.cfg.horizon
        if (y.shape != (7,) or refs.shape != (n + 1, 3)
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(refs))):
            raise ValueError('MPC requires finite seven-state input and horizon+1 references')
        started = time.perf_counter()
        base = 0.
        try:
            base = self.controller.modern_accel(y, tuple(refs[0]), 'lqr')
            return self._action_impl(y, refs, _started=started, **kwargs)
        except _DeadlineReached:
            return self._fallback(started, 'time_limit',
                                  'configured MPC time limit reached during model construction or rollout', base)
        except Exception as exc:
            return self._fallback(started, 'numerical_solver_error',
                                  f'model/solver exception: {exc}', base)

    @staticmethod
    def _solver_outcome(result):
        status = int(getattr(result, 'status', -1))
        if bool(getattr(result, 'success', False)):
            return 'success'
        if status == 9:
            return 'iteration_limit'
        if status == 4:
            return 'infeasible_constraints'
        if status in (5, 6, 7):
            return 'numerical_solver_error'
        return 'solver_failure'

    def _check_deadline(self, deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _DeadlineReached()

    @staticmethod
    def _solve_osqp(H, g, matrix, lower, upper, lb, ub, initial,
                    *, max_iterations, deadline):
        """Solve the condensed convex QP using OSQP's CPU backend."""
        n = len(g)
        P = sparse.triu(sparse.csc_matrix(2. * H), format='csc')
        A = sparse.vstack((sparse.csc_matrix(matrix), sparse.eye(n, format='csc')),
                          format='csc')
        lo = np.r_[lower, lb]
        hi = np.r_[upper, ub]
        settings = {
            'verbose': False,
            'max_iter': max_iterations,
            'eps_abs': 1e-6,
            'eps_rel': 1e-6,
            'check_termination': 10,
            'polishing': False,
        }
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.:
                raise _DeadlineReached()
            settings['time_limit'] = remaining

        solver = osqp.OSQP()
        solver.setup(P=P, q=2. * g, A=A, l=lo, u=hi, **settings)
        solver.warm_start(x=np.asarray(initial, dtype=float))
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.:
                raise _DeadlineReached()
            solver.update_settings(time_limit=remaining)
        solved = solver.solve()
        if deadline is not None and time.perf_counter() >= deadline:
            raise _DeadlineReached()
        info = solved.info
        status_value = int(info.status_val)
        status = {
            1: 0,   # solved
            2: 0,   # solved inaccurate; primal feasibility is checked below
            3: 4,   # primal infeasible
            4: 4,   # primal infeasible inaccurate
            7: 9,   # maximum iterations reached
            8: 10,  # time limit reached
            9: 6,   # nonconvex problem
            10: 10, # interrupted
            11: 8,  # unsolved
        }.get(status_value, 6)
        return SimpleNamespace(
            x=solved.x, success=status_value in (1, 2), status=status,
            nit=int(info.iter), message=str(info.status),
            solver_backend='osqp', solver_status=status_value,
            solver_seconds=float(info.solve_time))

    def _action_impl(self, state, references, *, command_queue=None, step_dir_actuator=None,
                     time_s=0., kick_torque=0., kick_at=-1., kick_duration=0.,
                     _started=None):
        c, p, cp = self.cfg, self.p, self.cp
        n = c.horizon
        y = np.asarray(state, dtype=float)
        refs = np.asarray(references, dtype=float)
        if (y.shape != (7,) or refs.shape != (n + 1, 3)
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(refs))):
            raise ValueError('MPC requires finite seven-state input and horizon+1 references')
        started = time.perf_counter() if _started is None else _started
        deadline = None if c.time_limit_seconds is None else started + c.time_limit_seconds
        self._check_deadline(deadline)
        base = self.controller.modern_accel(y, tuple(refs[0]), 'lqr')
        self._check_deadline(deadline)
        if command_queue is None and (c.command_delay > 0 or c.command_jitter > 0):
            return self._fallback(started, 'model_configuration_error',
                                  'live command queue required when delay or jitter is configured', base)
        if c.solver_backend == 'osqp' and osqp is None:
            return self._fallback(started, 'model_configuration_error',
                                  'OSQP backend requested but the osqp package is unavailable', base)
        use_osqp = osqp is not None and (
            c.solver_backend == 'osqp'
            or (c.solver_backend == 'auto' and n >= 10))

        sources, fixed = self._command_source_plan(command_queue, time_s)
        self._check_deadline(deadline)
        z0 = self._augmented_state(y, step_dir_actuator)
        dim = len(z0)
        actuator_template = deepcopy(step_dir_actuator) if step_dir_actuator is not None else StepDirActuator(c.step_dir)
        # Nominal trajectory follows the LQR command stream through the cloned queue.
        nominal_z = [z0]
        nominal_u = np.zeros(n)
        nominal_applied = np.zeros(n)
        nominal_actuators = [actuator_template]
        for k in range(n):
            self._check_deadline(deadline)
            current = nominal_z[-1]
            nominal_y = current[:7]
            nominal_u[k] = self.controller.modern_accel(nominal_y, tuple(refs[k]), 'lqr')
            source = sources[k]
            nominal_applied[k] = fixed[k] if source is None else nominal_u[source]
            next_z = self._transition_augmented(
                current, nominal_applied[k], nominal_actuators[-1],
                time_s=time_s + k * c.control_dt, kick_torque=kick_torque,
                kick_at=kick_at, kick_duration=kick_duration, deadline=deadline)
            nominal_z.append(next_z)
            if c.actuator_mode == 'step_dir':
                next_actuator = deepcopy(nominal_actuators[-1])
                next_actuator.velocity_rad_s = float(next_z[7])
                next_actuator.position_rad = float(next_z[8])
                next_actuator.step_count = int(round(next_actuator.position_rad / next_actuator.params.step_angle_rad))
                nominal_actuators.append(next_actuator)
            else:
                nominal_actuators.append(None)

        H = np.zeros((n, n)); g = np.zeros(n)
        rows, lower, upper = [], [], []
        S = np.zeros((dim, n)); offset = z0.copy(); eye = np.eye(n)
        q = np.zeros(dim)
        q[2] = c.position_weight / c.tracking_scale ** 2
        q[3] = c.velocity_weight / c.velocity_scale ** 2
        q[4] = c.angle_weight / c.angle_scale ** 2
        q[5] = c.angular_rate_weight / c.angular_rate_scale ** 2
        Q = np.diag(q)

        def constrain(row, lo, hi):
            rows.append(np.asarray(row, dtype=float)); lower.append(float(lo)); upper.append(float(hi))

        max_motor_speed = cp.max_speed / max(p.pulley_radius, 1e-12)
        torque_limit = self.plant.motor_torque_limit(max_motor_speed)
        for k in range(n):
            self._check_deadline(deadline)
            z_nom = nominal_z[k]
            y_nom = z_nom[:7]
            state_sensitivity = S.copy()
            state_offset = offset.copy()
            ref = tuple(refs[k])
            base_nom = nominal_u[k]
            refresh_linearization = (c.actuator_mode != 'torque'
                                     or k % c.linearization_stride == 0)
            if refresh_linearization:
                base_y = self._sparse_numerical_jacobian(
                    lambda s: np.array([self.controller.modern_accel(s[:7], ref, 'lqr')]),
                    z_nom, (2, 3, 4, 5))
            residual_row = eye[k] - base_y @ S
            residual_const = -base_nom - base_y @ (offset - z_nom)
            constrain(residual_row, -c.residual_accel_limit - residual_const,
                      c.residual_accel_limit - residual_const)
            rail_lo, rail_hi = self.controller.rail_accel_bounds(y_nom)
            lb = max(rail_lo, -cp.max_accel)
            ub = min(rail_hi, cp.max_accel)
            if lb > ub or abs(y_nom[2]) > cp.rail_soft_fraction * p.rail_half_travel + 1e-6:
                return self._fallback(started, 'infeasible_initial_bounds',
                                      'nominal state has no admissible command inside the soft rail', base)
            command_row = eye[k]
            constrain(command_row, lb, ub)

            H += (c.action_weight / c.residual_accel_limit ** 2) * np.outer(residual_row, residual_row)
            g += (c.action_weight / c.residual_accel_limit ** 2) * residual_const * residual_row

            applied_nom = nominal_applied[k]
            if c.actuator_mode == 'torque':
                if refresh_linearization:
                    F, G = self._linearize_torque_transition(
                        z_nom, applied_nom, time_s=time_s + k*c.control_dt,
                        kick_torque=kick_torque, kick_at=kick_at,
                        kick_duration=kick_duration, deadline=deadline)
            else:
                f = lambda z: self._transition_augmented(
                    z, applied_nom, nominal_actuators[k], time_s=time_s + k*c.control_dt,
                    kick_torque=kick_torque, kick_at=kick_at, kick_duration=kick_duration,
                    deadline=deadline)
                F = numerical_jacobian(f, z_nom)
                eps = .1
                G = (self._transition_augmented(
                        z_nom, applied_nom + eps, nominal_actuators[k],
                        time_s=time_s + k*c.control_dt, kick_torque=kick_torque,
                        kick_at=kick_at, kick_duration=kick_duration, deadline=deadline)
                     - self._transition_augmented(
                        z_nom, applied_nom - eps, nominal_actuators[k],
                        time_s=time_s + k*c.control_dt, kick_torque=kick_torque,
                        kick_at=kick_at, kick_duration=kick_duration, deadline=deadline)) / (2. * eps)
            self._check_deadline(deadline)
            # The nominal rollout already evaluated this exact transition.
            affine = nominal_z[k + 1] - F @ z_nom - G * applied_nom
            source = sources[k]
            source_row = np.zeros(n) if source is None else eye[source]
            source_constant = fixed[k] if source is None else 0.
            S = F @ S + np.outer(G, source_row)
            offset = F @ offset + G * source_constant + affine

            target = np.zeros(dim)
            target[2] = refs[k + 1, 0]
            target[3] = refs[k + 1, 1]
            weight = c.terminal_multiplier if k == n - 1 else 1.
            H += weight * (S.T @ Q @ S)
            g += weight * (S.T @ Q @ (offset - target))
            constrain(S[2], -cp.rail_soft_fraction*p.rail_half_travel-offset[2],
                      cp.rail_soft_fraction*p.rail_half_travel-offset[2])
            constrain(S[3], -cp.max_speed-offset[3], cp.max_speed-offset[3])
            constrain(S[1], -max_motor_speed-offset[1], max_motor_speed-offset[1])

            # Conservative torque envelope for torque mode; STEP/DIR includes its
            # own torque saturation dynamics and records that saturation instead.
            if c.actuator_mode == 'torque':
                torque_nom = self.controller.torque_from_accel(y_nom, applied_nom)
                if refresh_linearization:
                    torque_y = self._sparse_numerical_jacobian(
                        lambda zz: np.array([self.controller.torque_from_accel(zz[:7], applied_nom)]),
                        z_nom, (1, 3))
                    torque_a = (self.controller.torque_from_accel(y_nom, applied_nom + 1e-4)
                                - self.controller.torque_from_accel(y_nom, applied_nom - 1e-4)) / 2e-4
                torque_source_row = np.zeros(n) if source is None else eye[source]
                tq_row = torque_y @ state_sensitivity + torque_a * torque_source_row
                source_constant = fixed[k] if source is None else 0.
                tq_const = (torque_nom + torque_y @ (state_offset-z_nom)
                            + torque_a * (source_constant-applied_nom))
                constrain(tq_row, -torque_limit-tq_const, torque_limit-tq_const)

        D = eye.copy(); D[1:] -= eye[:-1]
        previous = np.zeros(n); previous[0] = self.previous_accel
        H += c.slew_weight / c.residual_accel_limit ** 2 * (D.T @ D)
        g -= c.slew_weight / c.residual_accel_limit ** 2 * D.T @ previous
        lb = np.full(n, -cp.max_accel); ub = np.full(n, cp.max_accel)
        H = (H + H.T) * .5
        scale = max(float(np.max(np.diag(H))), 1.)
        H /= scale; g /= scale
        matrix = np.asarray(rows); lower_array = np.asarray(lower); upper_array = np.asarray(upper)
        self._check_deadline(deadline)
        equal = np.isclose(lower_array, upper_array, rtol=0., atol=1e-12)
        constraints = []
        if np.any(equal):
            constraints.append(LinearConstraint(matrix[equal], lower_array[equal], upper_array[equal]))
        if np.any(~equal):
            constraints.append(LinearConstraint(matrix[~equal], lower_array[~equal], upper_array[~equal]))

        def callback(_):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _DeadlineReached()

        try:
            if deadline is not None and time.perf_counter() >= deadline:
                raise _DeadlineReached()
            initial = np.clip(nominal_u if not np.any(self.warm) else self.warm, lb, ub)
            if use_osqp:
                result = self._solve_osqp(
                    H, g, matrix, lower_array, upper_array, lb, ub, initial,
                    max_iterations=c.qp_max_iterations, deadline=deadline)
                if deadline is not None and time.perf_counter() >= deadline:
                    return self._fallback(started, 'time_limit',
                                          'configured MPC time limit reached', base,
                                          result=result)
                if int(result.status) == 10:
                    return self._fallback(started, 'time_limit',
                                          str(result.message), base, result=result)
            else:
                result = minimize(
                    lambda u: float(u @ H @ u + 2. * g @ u), initial,
                    jac=lambda u: 2. * (H @ u + g), method='SLSQP',
                    bounds=Bounds(lb, ub), constraints=constraints, callback=callback,
                    options={'maxiter': c.max_iterations, 'ftol': 1e-8})
        except _DeadlineReached:
            return self._fallback(started, 'time_limit', 'configured MPC time limit reached', base)
        except Exception as exc:
            return self._fallback(started, 'numerical_solver_error', str(exc), base)

        candidate = np.asarray(getattr(result, 'x', np.zeros(n)), dtype=float)
        if candidate.shape != (n,) or not np.all(np.isfinite(candidate)):
            return self._fallback(started, 'numerical_solver_error',
                                  'solver returned a non-finite or malformed candidate', base,
                                  result=result)
        violation = max(float(np.max(lower_array - matrix @ candidate)),
                        float(np.max(matrix @ candidate - upper_array)), 0.)
        if not bool(getattr(result, 'success', False)):
            outcome = self._solver_outcome(result)
            return self._fallback(started, outcome, str(result.message), base,
                                  result=result, violation=violation)
        if violation > 1e-5:
            return self._fallback(started, 'infeasible_constraints',
                                  'solver candidate violates linearized MPC constraints', base,
                                  result=result, violation=violation)

        accepted, outcome, reason, nonlinear = self._nonlinear_accept(
            y, refs, candidate, sources, fixed, base, step_dir_actuator, time_s,
            kick_torque, kick_at, kick_duration, torque_limit, max_motor_speed,
            deadline)
        if not accepted:
            return self._fallback(started, outcome, reason, base, result=result,
                                  violation=violation, nonlinear=nonlinear)

        self.warm = np.r_[candidate[1:], candidate[-1]]
        self.previous_accel = float(candidate[0])
        action = float(np.clip((candidate[0] - base) / c.residual_accel_limit, -1., 1.))
        self.diagnostics = {
            'success': True, 'outcome': 'success', 'category': 'success', 'reason': '',
            'fallback_reason': '', 'fallback_action': None,
            'solve_seconds': time.perf_counter() - started,
            'iterations': int(getattr(result, 'nit', 0)),
            'constraint_violation': violation,
            'actuator_mode': c.actuator_mode,
            'solver_backend': getattr(result, 'solver_backend', 'slsqp'),
            'solver_status': getattr(result, 'solver_status', int(getattr(result, 'status', -1))),
            'command_delay_seconds': c.command_delay if command_queue is None else command_queue.delay,
            'command_jitter_seconds': c.command_jitter if command_queue is None else command_queue.jitter,
            **nonlinear,
        }
        return action

    def _state_constraint_violations(self, state, max_motor_speed):
        """Return positive soft-rail and speed excesses for one plant state."""
        p, cp = self.p, self.cp
        state = np.asarray(state, dtype=float)
        return (
            max(abs(float(state[2])) - cp.rail_soft_fraction * p.rail_half_travel, 0.),
            max(abs(float(state[3])) - cp.max_speed, 0.),
            max(abs(float(state[1])) - max_motor_speed, 0.),
        )

    def _nonlinear_accept(self, y, refs, candidate, sources, fixed, base,
                          step_dir_actuator, time_s, kick_torque, kick_at,
                          kick_duration, torque_limit, max_motor_speed, deadline=None):
        c, p, cp = self.cfg, self.p, self.cp
        current = self._augmented_state(y, step_dir_actuator)
        actuator = deepcopy(step_dir_actuator) if step_dir_actuator is not None else StepDirActuator(c.step_dir)
        peak_rail = abs(float(current[2])) / max(p.rail_half_travel, 1e-12)
        peak_carriage_speed = abs(float(current[3]))
        peak_motor_speed = abs(float(current[1]))
        max_torque_ratio = 0.
        step_torque_saturation = 0

        def observe_state(state):
            nonlocal peak_rail, peak_carriage_speed, peak_motor_speed
            state = np.asarray(state, dtype=float)
            peak_rail = max(peak_rail, abs(float(state[2])) / max(p.rail_half_travel, 1e-12))
            peak_carriage_speed = max(peak_carriage_speed, abs(float(state[3])))
            peak_motor_speed = max(peak_motor_speed, abs(float(state[1])))
            return self._state_constraint_violations(state, max_motor_speed)

        def rejection_diagnostics(violations):
            return {
                'nonlinear_peak_rail_fraction': peak_rail,
                'nonlinear_peak_carriage_speed_m_s': peak_carriage_speed,
                'nonlinear_peak_motor_speed_rad_s': peak_motor_speed,
                'nonlinear_max_constraint_violation': max(
                    violations[0], violations[1],
                    violations[2] * p.pulley_radius),
            }

        def violates_tolerance(violations):
            # Match the existing nonlinear endpoint tolerances, now at every
            # physics substep rather than only at control-interval boundaries.
            return (violations[0] > 1e-5 or violations[1] > 1e-4
                    or violations[2] > 1e-3)

        for k, command in enumerate(candidate):
            self._check_deadline(deadline)
            state = current[:7]
            issue_base = self.controller.modern_accel(state, tuple(refs[k]), 'lqr')
            lo, hi = self.controller.rail_accel_bounds(state)
            issued = self.controller.project_accel(state, float(command))
            if abs(command - issue_base) > c.residual_accel_limit + 1e-3:
                return False, 'nonlinear_actuator_rejection', 'nonlinear residual authority check', {
                    'nonlinear_peak_rail_fraction': peak_rail,
                    'nonlinear_max_constraint_violation': abs(command - issue_base) - c.residual_accel_limit}
            if abs(issued-command) > 1e-4 or command < lo-1e-4 or command > hi+1e-4:
                return False, 'nonlinear_actuator_rejection', 'command projection changed the optimized command', {
                    'nonlinear_peak_rail_fraction': peak_rail,
                    'nonlinear_max_constraint_violation': max(abs(issued-command), lo-command, command-hi, 0.)}
            source = sources[k]
            active = fixed[k] if source is None else float(candidate[source])
            applied = self.controller.project_accel(state, active)
            torque = self.controller.torque_from_accel(state, applied)
            if c.actuator_mode == 'torque':
                ratio = abs(torque) / max(torque_limit, 1e-12)
                max_torque_ratio = max(max_torque_ratio, ratio)
                if ratio > 1. + 1e-5:
                    return False, 'nonlinear_actuator_rejection', 'nonlinear motor torque envelope check', {
                        'nonlinear_peak_rail_fraction': peak_rail,
                        'nonlinear_max_constraint_violation': ratio-1.}
                current_y = state.copy()
                nsteps = round(c.control_dt / c.physics_dt)
                for i in range(nsteps):
                    self._check_deadline(deadline)
                    now = time_s + k*c.control_dt + i*c.physics_dt
                    ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at+kick_duration else 0.
                    current_y = self.plant.rk4(
                        current_y, torque, c.physics_dt, external_torque=ext)
                    violations = observe_state(current_y)
                    if violates_tolerance(violations):
                        return False, 'nonlinear_state_rejection', 'nonlinear substep rail or speed check', rejection_diagnostics(violations)
                current = current_y
            if c.actuator_mode == 'step_dir':
                current_y = current[:7]
                max_carriage_speed = c.step_dir.max_velocity_rad_s * p.pulley_radius
                target_v = clamp(float(current_y[3]) + applied*c.control_dt,
                                 -max_carriage_speed, max_carriage_speed)
                actuator.command(target_v/max(p.pulley_radius, 1e-12))
                nsteps = round(c.control_dt/c.physics_dt)
                sat_this_interval = False
                for i in range(nsteps):
                    self._check_deadline(deadline)
                    now = time_s + k*c.control_dt + i*c.physics_dt
                    ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at+kick_duration else 0.
                    diag = actuator.advance(c.physics_dt, float(current_y[0]), float(current_y[1]),
                                            self.plant.motor_torque_limit(float(current_y[1])))
                    sat_this_interval = sat_this_interval or bool(diag['torque_saturated'])
                    current_y = self.plant.rk4(current_y, diag['torque_command_nm'], c.physics_dt,
                                               external_torque=ext)
                    violations = observe_state(current_y)
                    if violates_tolerance(violations):
                        return False, 'nonlinear_state_rejection', 'nonlinear substep rail or speed check', rejection_diagnostics(violations)
                current = np.r_[current_y, actuator.velocity_rad_s, actuator.position_rad]
                step_torque_saturation += int(sat_this_interval)
            violations = observe_state(current[:7])
            if violates_tolerance(violations):
                return False, 'nonlinear_state_rejection', 'nonlinear rail or speed check', rejection_diagnostics(violations)
        self._check_deadline(deadline)
        return True, 'success', '', {
            'nonlinear_peak_rail_fraction': peak_rail,
            'nonlinear_peak_carriage_speed_m_s': peak_carriage_speed,
            'nonlinear_peak_motor_speed_rad_s': peak_motor_speed,
            'nonlinear_max_torque_ratio': max_torque_ratio,
            'step_dir_torque_saturated_intervals': step_torque_saturation,
            'nonlinear_max_constraint_violation': 0.,
        }

    def _fallback(self, started, outcome, reason, base, *, result=None,
                  violation=0., nonlinear=None):
        self.warm[:] = 0.
        self.previous_accel = float(base)
        self.diagnostics = {
            'success': False, 'outcome': outcome, 'category': outcome,
            'reason': str(reason), 'fallback_reason': str(reason), 'fallback_action': 0.,
            'solve_seconds': time.perf_counter()-started,
            'iterations': int(getattr(result, 'nit', 0)) if result is not None else 0,
            'solver_status': (getattr(result, 'solver_status', int(getattr(result, 'status', -1)))
                              if result is not None else None),
            'solver_backend': getattr(result, 'solver_backend', 'slsqp'),
            'constraint_violation': float(violation),
            'actuator_mode': self.cfg.actuator_mode,
        }
        if nonlinear:
            self.diagnostics.update(nonlinear)
        return 0.

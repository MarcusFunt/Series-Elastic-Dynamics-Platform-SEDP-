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

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from active_vibration_rig_2d import (
    Controller, RigPlant, StepDirActuator, StepDirParams, clamp,
)
from state_estimator import held_transition, numerical_jacobian
from timing_models import CommandDelayQueue


@dataclass
class MPCConfig:
    horizon: int = 40
    physics_dt: float = .001
    control_dt: float = .01
    residual_accel_limit: float = 2.5
    max_iterations: int = 60
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
        if self.horizon < 2 or self.residual_accel_limit <= 0 or self.max_iterations < 1:
            raise ValueError('Invalid MPC configuration')
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

    def _transition(self, y, accel):
        """Legacy-shaped torque transition retained for callers and diagnostics."""
        applied = self.controller.project_accel(y, float(accel))
        torque = self.controller.torque_from_accel(y, applied)
        return held_transition(self.plant, y, torque, self.cfg.physics_dt,
                               self.cfg.control_dt)

    def _transition_active(self, y, accel, step_dir_actuator=None, *, time_s=0.,
                           kick_torque=0., kick_at=-1., kick_duration=0.):
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
            for i in range(nsteps):
                now = time_s + i * self.cfg.physics_dt
                ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at + kick_duration else 0.
                y = self.plant.rk4(y, torque, self.cfg.physics_dt, external_torque=ext)
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
            now = time_s + i * self.cfg.physics_dt
            ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at + kick_duration else 0.
            diag = actuator.advance(
                self.cfg.physics_dt, float(y[0]), float(y[1]),
                self.plant.motor_torque_limit(float(y[1])))
            y = self.plant.rk4(y, diag['torque_command_nm'], self.cfg.physics_dt,
                               external_torque=ext)
        return y, actuator

    def _augmented_state(self, y, actuator):
        y = np.asarray(y, dtype=float)
        if self.cfg.actuator_mode == 'torque':
            return y.copy()
        actuator = actuator or StepDirActuator(self.cfg.step_dir)
        return np.r_[y, actuator.velocity_rad_s, actuator.position_rad]

    def _transition_augmented(self, z, accel, actuator_template, *, time_s=0.,
                              kick_torque=0., kick_at=-1., kick_duration=0.):
        z = np.asarray(z, dtype=float)
        if self.cfg.actuator_mode == 'torque':
            return self._transition_active(z[:7], accel, time_s=time_s,
                                           kick_torque=kick_torque, kick_at=kick_at,
                                           kick_duration=kick_duration)[0]
        actuator = deepcopy(actuator_template) if actuator_template is not None else StepDirActuator(self.cfg.step_dir)
        actuator.velocity_rad_s = float(z[7])
        actuator.position_rad = float(z[8])
        actuator.step_count = int(round(actuator.position_rad / actuator.params.step_angle_rad))
        actuator.step_phase_rad = (actuator.position_rad
                                   - actuator.step_count * actuator.params.step_angle_rad)
        y_next, actuator_next = self._transition_active(
            z[:7], accel, actuator, time_s=time_s, kick_torque=kick_torque,
            kick_at=kick_at, kick_duration=kick_duration)
        return self._augmented_state(y_next, actuator_next)

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
            return self._action_impl(y, refs, **kwargs)
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

    def _action_impl(self, state, references, *, command_queue=None, step_dir_actuator=None,
                     time_s=0., kick_torque=0., kick_at=-1., kick_duration=0.):
        c, p, cp = self.cfg, self.p, self.cp
        n = c.horizon
        y = np.asarray(state, dtype=float)
        refs = np.asarray(references, dtype=float)
        if (y.shape != (7,) or refs.shape != (n + 1, 3)
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(refs))):
            raise ValueError('MPC requires finite seven-state input and horizon+1 references')
        started = time.perf_counter()
        deadline = None if c.time_limit_seconds is None else started + c.time_limit_seconds
        base = self.controller.modern_accel(y, tuple(refs[0]), 'lqr')
        if deadline is not None and time.perf_counter() >= deadline:
            return self._fallback(started, 'time_limit', 'MPC time limit elapsed before optimization', base)
        if command_queue is None and (c.command_delay > 0 or c.command_jitter > 0):
            return self._fallback(started, 'model_configuration_error',
                                  'live command queue required when delay or jitter is configured', base)

        sources, fixed = self._command_source_plan(command_queue, time_s)
        z0 = self._augmented_state(y, step_dir_actuator)
        dim = len(z0)
        actuator_template = deepcopy(step_dir_actuator) if step_dir_actuator is not None else StepDirActuator(c.step_dir)
        # Nominal trajectory follows the LQR command stream through the cloned queue.
        nominal_z = [z0]
        nominal_u = np.zeros(n)
        nominal_applied = np.zeros(n)
        nominal_actuators = [actuator_template]
        for k in range(n):
            current = nominal_z[-1]
            nominal_y = current[:7]
            nominal_u[k] = self.controller.modern_accel(nominal_y, tuple(refs[k]), 'lqr')
            source = sources[k]
            nominal_applied[k] = fixed[k] if source is None else nominal_u[source]
            next_z = self._transition_augmented(
                current, nominal_applied[k], nominal_actuators[-1],
                time_s=time_s + k * c.control_dt, kick_torque=kick_torque,
                kick_at=kick_at, kick_duration=kick_duration)
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
            z_nom = nominal_z[k]
            y_nom = z_nom[:7]
            state_sensitivity = S.copy()
            state_offset = offset.copy()
            ref = tuple(refs[k])
            base_nom = nominal_u[k]
            base_y = numerical_jacobian(
                lambda s: np.array([self.controller.modern_accel(s[:7], ref, 'lqr')]), z_nom)[0]
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
            f = lambda z: self._transition_augmented(
                z, applied_nom, nominal_actuators[k], time_s=time_s + k*c.control_dt,
                kick_torque=kick_torque, kick_at=kick_at, kick_duration=kick_duration)
            F = numerical_jacobian(f, z_nom)
            eps = .1 if c.actuator_mode == 'step_dir' else 1e-4
            G = (self._transition_augmented(
                    z_nom, applied_nom + eps, nominal_actuators[k],
                    time_s=time_s + k*c.control_dt, kick_torque=kick_torque,
                    kick_at=kick_at, kick_duration=kick_duration)
                 - self._transition_augmented(
                    z_nom, applied_nom - eps, nominal_actuators[k],
                    time_s=time_s + k*c.control_dt, kick_torque=kick_torque,
                    kick_at=kick_at, kick_duration=kick_duration)) / (2. * eps)
            affine = f(z_nom) - F @ z_nom - G * applied_nom
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
                torque_y = numerical_jacobian(
                    lambda zz: np.array([self.controller.torque_from_accel(zz[:7], applied_nom)]), z_nom)[0]
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
            result = minimize(
                lambda u: float(u @ H @ u + 2. * g @ u),
                np.clip(nominal_u if not np.any(self.warm) else self.warm, lb, ub),
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
            kick_torque, kick_at, kick_duration, torque_limit, max_motor_speed)
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
            'command_delay_seconds': c.command_delay if command_queue is None else command_queue.delay,
            'command_jitter_seconds': c.command_jitter if command_queue is None else command_queue.jitter,
            **nonlinear,
        }
        return action

    def _nonlinear_accept(self, y, refs, candidate, sources, fixed, base,
                          step_dir_actuator, time_s, kick_torque, kick_at,
                          kick_duration, torque_limit, max_motor_speed):
        c, p, cp = self.cfg, self.p, self.cp
        current = self._augmented_state(y, step_dir_actuator)
        actuator = deepcopy(step_dir_actuator) if step_dir_actuator is not None else StepDirActuator(c.step_dir)
        peak_rail = 0.
        peak_speed = 0.
        max_torque_ratio = 0.
        step_torque_saturation = 0
        for k, command in enumerate(candidate):
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
            if c.actuator_mode == 'step_dir':
                current_y = current[:7]
                max_carriage_speed = c.step_dir.max_velocity_rad_s * p.pulley_radius
                target_v = clamp(float(current_y[3]) + applied*c.control_dt,
                                 -max_carriage_speed, max_carriage_speed)
                actuator.command(target_v/max(p.pulley_radius, 1e-12))
                nsteps = round(c.control_dt/c.physics_dt)
                sat_this_interval = False
                for i in range(nsteps):
                    now = time_s + k*c.control_dt + i*c.physics_dt
                    ext = kick_torque if kick_at >= 0 and kick_at <= now < kick_at+kick_duration else 0.
                    diag = actuator.advance(c.physics_dt, float(current_y[0]), float(current_y[1]),
                                            self.plant.motor_torque_limit(float(current_y[1])))
                    sat_this_interval = sat_this_interval or bool(diag['torque_saturated'])
                    current_y = self.plant.rk4(current_y, diag['torque_command_nm'], c.physics_dt,
                                               external_torque=ext)
                    peak_rail = max(peak_rail, abs(float(current_y[2]))/p.rail_half_travel)
                    peak_speed = max(peak_speed, abs(float(current_y[3])))
                current = np.r_[current_y, actuator.velocity_rad_s, actuator.position_rad]
                step_torque_saturation += int(sat_this_interval)
            else:
                current_y, _ = self._transition_active(
                    state, active, time_s=time_s+k*c.control_dt,
                    kick_torque=kick_torque, kick_at=kick_at, kick_duration=kick_duration)
                current = current_y
                peak_rail = max(peak_rail, abs(float(current_y[2]))/p.rail_half_travel)
                peak_speed = max(peak_speed, abs(float(current_y[3])))
            state_next = current[:7]
            peak_speed = max(peak_speed, abs(float(state_next[1]))*p.pulley_radius)
            if (abs(state_next[2]) > cp.rail_soft_fraction*p.rail_half_travel+1e-5
                    or abs(state_next[3]) > cp.max_speed+1e-4
                    or abs(state_next[1]) > max_motor_speed+1e-3):
                return False, 'nonlinear_state_rejection', 'nonlinear rail or speed check', {
                    'nonlinear_peak_rail_fraction': peak_rail,
                    'nonlinear_max_constraint_violation': max(
                        abs(state_next[2])-cp.rail_soft_fraction*p.rail_half_travel,
                        abs(state_next[3])-cp.max_speed,
                        (abs(state_next[1])-max_motor_speed)*p.pulley_radius, 0.)}
        return True, 'success', '', {
            'nonlinear_peak_rail_fraction': peak_rail,
            'nonlinear_peak_carriage_speed_m_s': peak_speed,
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
            'solver_status': int(getattr(result, 'status', -1)) if result is not None else None,
            'constraint_violation': float(violation),
            'actuator_mode': self.cfg.actuator_mode,
        }
        if nonlinear:
            self.diagnostics.update(nonlinear)
        return 0.

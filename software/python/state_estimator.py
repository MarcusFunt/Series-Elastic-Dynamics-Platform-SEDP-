"""Encoder/gyro measurements and an eight-state extended Kalman filter.

The controller uses nominal parameters, never randomized ground-truth parameters.
Carriage position is inferred through the belt model unless a linear encoder is
explicitly enabled. This inference assumes a homed motor and calibrated pulley.
"""
from dataclasses import dataclass
import math
import numpy as np
from active_vibration_rig_2d import PlantParams, RigPlant, quantize_angle_12bit, wrap_pi


@dataclass
class MeasurementModel:
    params: PlantParams
    noise: bool = True
    linear_encoder: bool = False
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(self, true_state, time):
        p = self.params
        gyro_noise = self.rng.normal(0, p.gyro_noise_std) if self.noise else 0.
        z = {'motor_angle': quantize_angle_12bit(float(true_state[0])),
             'lever_angle': quantize_angle_12bit(float(true_state[4])),
             'gyro': float(true_state[5])+p.gyro_bias+gyro_noise, 'time': float(time)}
        if self.linear_encoder:
            z['carriage_position'] = float(true_state[2])+(self.rng.normal(0, 2e-5) if self.noise else 0.)
        return z


def held_transition(plant, state, torque, physics_dt, control_dt):
    y = np.asarray(state, dtype=float).copy()
    for _ in range(round(control_dt/physics_dt)):
        y = plant.rk4(y, torque, physics_dt)
    return y


def held_transition_batch(plant, states, torque, physics_dt, control_dt):
    """Advance many perturbed states through one control interval together."""
    y = np.asarray(states, dtype=float).copy()
    for _ in range(round(control_dt/physics_dt)):
        y = plant.rk4_batch(y, torque, physics_dt)
    return y


def scheduled_transition_batch(plant, states, torque_trace, physics_dt):
    """Advance perturbed states through a per-physics-step torque schedule."""
    y = np.asarray(states, dtype=float).copy()
    for torque in torque_trace:
        y = plant.rk4_batch(y, float(torque), physics_dt)
    return y


def numerical_jacobian(function, point):
    point = np.asarray(point, dtype=float)
    columns = []
    for i in range(len(point)):
        eps = 1e-5*max(1., abs(point[i]))
        delta = np.zeros_like(point); delta[i] = eps
        columns.append((function(point+delta)-function(point-delta))/(2*eps))
    return np.stack(columns, axis=1)


class StateEstimator:
    """EKF with a bounded command history for delayed measurements.

    Delayed packets are inserted at acquisition time and the filter is replayed
    from a saved checkpoint. Sensor sample times are physics-tick aligned by the
    timing model; each controller transition stores its per-physics-step torque
    trace for accurate replay, including STEP/DIR actuator output.
    """
    def __init__(self, params, physics_dt=.001, control_dt=.01, linear_encoder=False,
                 history_horizon=0.0):
        if physics_dt <= 0 or control_dt < physics_dt or not math.isclose(control_dt/physics_dt, round(control_dt/physics_dt), abs_tol=1e-9):
            raise ValueError('Invalid estimator timing')
        self.params = params; self.plant = RigPlant(params)
        self.physics_dt = physics_dt; self.control_dt = control_dt
        self.linear_encoder = linear_encoder
        self.history_horizon = max(0.0, float(history_horizon))
        self.reset()

    def reset(self):
        self.mean = np.zeros(8)
        self.covariance = np.diag(np.square([.002, .5, .002, .08, .04, .2, .03, .03]))
        self.time = 0.; self.innovation_norm = 0.
        self.last_update_was_delayed = False
        self._anchor_time = 0.0
        self._anchor_mean = self.mean.copy()
        self._anchor_covariance = self.covariance.copy()
        self._steps = []
        self._events = []
        self._checkpoints = {self._time_key(0.0): (self.mean.copy(), self.covariance.copy())}
        self._event_sequence = 0

    @staticmethod
    def _time_key(time):
        return round(float(time), 12)

    @property
    def state(self):
        return self.mean[:7].copy()

    def _propagate(self, torque, dt):
        # Preserve the held-torque path when the supplied schedule is constant;
        # replay can supply a changing per-physics-step torque trace.
        old_mean, old_covariance = self.mean.copy(), self.covariance.copy()
        substeps = round(dt / self.physics_dt)
        torques = np.asarray(torque, dtype=float)
        if torques.ndim == 0:
            trace = None
            held_torque = float(torques)
        elif torques.ndim == 1 and torques.shape == (substeps,):
            if not np.all(np.isfinite(torques)):
                raise ValueError('torque trace must contain only finite values')
            trace = torques
            held_torque = float(trace[0]) if len(trace) and np.all(trace == trace[0]) else None
        else:
            raise ValueError('torque input must be scalar or have one value per physics substep')
        perturbations = np.repeat(old_mean[None, :7], 15, axis=0)
        epsilons = []
        for index in range(7):
            eps = 1e-5*max(1., abs(old_mean[index]))
            epsilons.append(eps)
            perturbations[1 + 2*index, index] += eps
            perturbations[2 + 2*index, index] -= eps
        if held_torque is not None:
            propagated = held_transition_batch(
                self.plant, perturbations, held_torque, self.physics_dt, dt
            )
        else:
            propagated = scheduled_transition_batch(
                self.plant, perturbations, trace, self.physics_dt
            )
        F = np.zeros((8, 8), dtype=float)
        F[7, 7] = 1.0
        for index, eps in enumerate(epsilons):
            F[:7, index] = (propagated[1 + 2*index] - propagated[2 + 2*index]) / (2*eps)
        self.mean = np.r_[propagated[0], old_mean[7]]
        Q = np.diag(np.square([2e-5,.06,3e-5,.025,2e-4,.025,.008,1e-4]))
        self.covariance = F @ old_covariance @ F.T + Q * (dt / self.control_dt)
        self.covariance = (self.covariance + self.covariance.T) * .5

    def predict(self, torque):
        start = self.time
        end = start + self.control_dt
        substeps = round(self.control_dt / self.physics_dt)
        values = np.asarray(torque, dtype=float)
        if values.ndim == 0:
            torque_trace = np.full(substeps, float(values), dtype=float)
        elif values.ndim == 1 and values.shape == (substeps,):
            if not np.all(np.isfinite(values)):
                raise ValueError('torque trace must contain only finite values')
            torque_trace = values.copy()
        else:
            raise ValueError('predict expects one torque value or one per physics substep')
        self._propagate(torque_trace, self.control_dt)
        self.time = end
        self._steps.append({'start': start, 'end': end, 'torque_trace': torque_trace})
        self._checkpoints[self._time_key(end)] = (self.mean.copy(), self.covariance.copy())
        self._prune_history()

    def _measurement_matrices(self, measurement):
        rows = {
            'motor_angle': (0, (.001534 / np.sqrt(12)) ** 2),
            'lever_angle': (4, (.001534 / np.sqrt(12)) ** 2),
            'gyro': (5, max(self.params.gyro_noise_std, .001) ** 2),
            'carriage_position': (2, (2e-5) ** 2),
        }
        keys = [key for key in ('motor_angle', 'lever_angle', 'gyro', 'carriage_position')
                if key in measurement and (key != 'carriage_position' or self.linear_encoder)]
        if not keys:
            raise ValueError('measurement contains no enabled sensor channels')
        H = np.zeros((len(keys), 8))
        variance = []
        for i, key in enumerate(keys):
            state_index, sensor_variance = rows[key]
            H[i, state_index] = 1.0
            if key == 'gyro':
                H[i, 7] = 1.0
            variance.append(sensor_variance)
        return keys, H, np.diag(variance)

    def _apply_measurement(self, measurement):
        keys, H, R = self._measurement_matrices(measurement)
        innovation = np.array([measurement[key] for key in keys], dtype=float) - H @ self.mean
        for i, key in enumerate(keys):
            if key in ('motor_angle', 'lever_angle'):
                innovation[i] = wrap_pi(innovation[i])
        P = self.covariance
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        self.mean += K @ innovation
        I = np.eye(8) - K @ H
        self.covariance = I @ P @ I.T + K @ R @ K.T
        self.covariance = (self.covariance + self.covariance.T) * .5
        self.innovation_norm = float(np.sqrt(innovation @ np.linalg.solve(S, innovation)))

    def _rebuild_from_anchor(self):
        self.mean = self._anchor_mean.copy()
        self.covariance = self._anchor_covariance.copy()
        self.time = self._anchor_time
        self._checkpoints = {self._time_key(self.time): (self.mean.copy(), self.covariance.copy())}
        events_by_time = {}
        for event in sorted(self._events, key=lambda item: (item['time'], item['arrival_time'], item['sequence'])):
            events_by_time.setdefault(self._time_key(event['time']), []).append(event['measurement'])
        for measurement in events_by_time.get(self._time_key(self.time), []):
            self._apply_measurement(measurement)
        self._checkpoints[self._time_key(self.time)] = (self.mean.copy(), self.covariance.copy())
        for step in self._steps:
            if step['end'] <= self.time + 1e-12:
                continue
            # Measurements may be acquired between controller ticks. Split the
            # recorded physics-step input trace at each acquisition tick.
            trace = step['torque_trace']
            trace_index = max(0, round((self.time - step['start']) / self.physics_dt))
            event_times = sorted(float(key) for key in events_by_time
                                 if self.time + 1e-12 < float(key) <= step['end'] + 1e-12)
            for event_time in event_times:
                if event_time > self.time + 1e-12:
                    event_index = round((event_time - step['start']) / self.physics_dt)
                    segment = trace[trace_index:event_index]
                    self._propagate(segment, len(segment) * self.physics_dt)
                    trace_index = event_index
                    self.time = event_time
                for measurement in events_by_time[self._time_key(event_time)]:
                    self._apply_measurement(measurement)
                self._checkpoints[self._time_key(self.time)] = (
                    self.mean.copy(), self.covariance.copy())
            if step['end'] > self.time + 1e-12:
                segment = trace[trace_index:]
                self._propagate(segment, len(segment) * self.physics_dt)
                self.time = step['end']
            self._checkpoints[self._time_key(self.time)] = (self.mean.copy(), self.covariance.copy())

    def update(self, measurement):
        measurement = dict(measurement)
        acquired = float(measurement.get('acquisition_time', measurement.get('time', self.time)))
        arrived = float(measurement.get('arrival_time', acquired))
        if not math.isfinite(acquired) or not math.isfinite(arrived):
            raise ValueError('measurement timestamps must be finite')
        measurement['time'] = acquired
        self.last_update_was_delayed = acquired < self.time - 1e-12

        # Keep direct estimator use backward compatible: callers that supply a
        # future timestamp without calling predict mean "observe now".
        if acquired > self.time + 1e-12:
            self.time = acquired
            self._anchor_time = acquired
            self._anchor_mean = self.mean.copy()
            self._anchor_covariance = self.covariance.copy()
            self._steps.clear(); self._events.clear()
            self._checkpoints = {self._time_key(acquired):
                                 (self.mean.copy(), self.covariance.copy())}
            self.last_update_was_delayed = False

        if acquired < self._anchor_time - 1e-12:
            return self.state
        event = {'time': acquired, 'arrival_time': arrived,
                 'sequence': self._event_sequence, 'measurement': measurement}
        self._event_sequence += 1
        self._events.append(event)
        if self.last_update_was_delayed:
            self._rebuild_from_anchor()
        else:
            self._apply_measurement(measurement)
            self._checkpoints[self._time_key(self.time)] = (self.mean.copy(), self.covariance.copy())
        self._prune_history()
        return self.state

    def _prune_history(self):
        if not self._steps or self.history_horizon < 0:
            return
        cutoff = self.time - self.history_horizon
        # Retain one complete transition before the requested horizon so a
        # packet at the horizon boundary can still be replayed.
        candidates = [key for key in self._checkpoints
                      if key < cutoff - 1e-12 and key >= self._anchor_time - 1e-12]
        if not candidates:
            return
        anchor_key = max(candidates)
        anchor_time = float(anchor_key)
        anchor_mean, anchor_covariance = self._checkpoints[anchor_key]
        self._anchor_time = anchor_time
        self._anchor_mean = anchor_mean.copy()
        self._anchor_covariance = anchor_covariance.copy()
        self._steps = [step for step in self._steps if step['end'] > anchor_time + 1e-12]
        self._events = [event for event in self._events if event['time'] > anchor_time + 1e-12]
        self._checkpoints = {key: value for key, value in self._checkpoints.items()
                             if key >= anchor_key}

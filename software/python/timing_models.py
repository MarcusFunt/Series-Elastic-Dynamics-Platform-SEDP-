"""Deterministic sensor and command timing models for simulation environments."""
from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Dict, Mapping, Optional

import numpy as np

from active_vibration_rig_2d import quantize_angle_12bit


SENSOR_CHANNELS = ('motor_angle', 'lever_angle', 'gyro', 'carriage_position')
_CHANNEL_SEEDS = {'motor_angle': 101, 'lever_angle': 211, 'gyro': 307,
                  'carriage_position': 401}


@dataclass(frozen=True)
class SensorChannelTiming:
    """One channel's sample period, transport timing, and packet loss rate."""

    sample_period: Optional[float] = None
    delay: float = 0.0
    jitter: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self):
        values = (self.delay, self.jitter, self.dropout_probability)
        if self.sample_period is not None and (
                not math.isfinite(self.sample_period) or self.sample_period <= 0):
            raise ValueError('sensor sample_period must be finite and positive')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('sensor delay, jitter, and dropout probability must be finite')
        if self.delay < 0 or self.jitter < 0:
            raise ValueError('sensor delay and jitter must be nonnegative')
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError('sensor dropout_probability must be between zero and one')


def normalize_sensor_timing(timing: Optional[Mapping], control_dt: float,
                            linear_encoder: bool = False,
                            physics_dt: Optional[float] = None) -> Dict[str, SensorChannelTiming]:
    """Validate overrides and fill omitted channels with synchronous defaults."""
    timing = timing or {}
    sample_tick = control_dt if physics_dt is None else physics_dt
    if not isinstance(timing, Mapping):
        raise ValueError('sensor_timing must be a mapping of channel names to settings')
    unknown = set(timing) - set(SENSOR_CHANNELS)
    if unknown:
        raise ValueError(f'unknown sensor channels: {sorted(unknown)}')
    channels = SENSOR_CHANNELS if linear_encoder else SENSOR_CHANNELS[:-1]
    normalized = {}
    for channel in channels:
        value = timing.get(channel, {})
        if isinstance(value, SensorChannelTiming):
            config = value
        elif isinstance(value, Mapping):
            config = SensorChannelTiming(**value)
        else:
            raise ValueError(f'{channel} timing must be a SensorChannelTiming or mapping')
        period = control_dt if config.sample_period is None else float(config.sample_period)
        if not math.isclose(period / sample_tick, round(period / sample_tick),
                            rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('sensor sample_period must be an integer multiple of physics_dt')
        normalized[channel] = SensorChannelTiming(
            sample_period=period, delay=float(config.delay), jitter=float(config.jitter),
            dropout_probability=float(config.dropout_probability))
    if 'carriage_position' in timing and not linear_encoder:
        raise ValueError('carriage_position timing requires linear_encoder=True')
    return normalized


def has_synchronous_sensor_timing(timing: Mapping[str, SensorChannelTiming],
                                  control_dt: float) -> bool:
    return all(math.isclose(config.sample_period, control_dt, rel_tol=0.0, abs_tol=1e-12)
               and config.delay == 0 and config.jitter == 0
               and config.dropout_probability == 0
               for config in timing.values())


class AsynchronousSensorSuite:
    """Samples each sensor on its own schedule and queues timestamped packets."""

    def __init__(self, params, timing: Mapping[str, SensorChannelTiming],
                 noise: bool, seed: int = 0):
        self.params = params
        self.timing = dict(timing)
        self.noise = bool(noise)
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.next_sample = {name: 0.0 for name in self.timing}
        self.sample_count = {name: 0 for name in self.timing}
        self.dropout_count = {name: 0 for name in self.timing}
        self.last_arrival = {name: 0.0 for name in self.timing}
        self._sequence = 0
        self._queue = []
        self._streams = {}
        for name, offset in _CHANNEL_SEEDS.items():
            base = self.seed + offset
            self._streams[name] = {
                key: np.random.default_rng(base + i * 100003)
                for i, key in enumerate(('dropout', 'delay', 'noise'), start=1)
            }

    @property
    def max_latency(self):
        return max((cfg.delay + cfg.jitter for cfg in self.timing.values()), default=0.0)

    def _sample(self, name, state):
        rng = self._streams[name]['noise']
        p = self.params
        if name == 'motor_angle':
            return quantize_angle_12bit(float(state[0]))
        if name == 'lever_angle':
            return quantize_angle_12bit(float(state[4]))
        if name == 'gyro':
            noise = rng.normal(0, p.gyro_noise_std) if self.noise else 0.0
            return float(state[5]) + p.gyro_bias + noise
        noise = rng.normal(0, 2e-5) if self.noise else 0.0
        return float(state[2]) + noise

    def tick(self, true_state, time: float):
        """Acquire due samples and return packets whose arrival time has passed."""
        time = float(time)
        eps = 1e-12
        for name, cfg in self.timing.items():
            while self.next_sample[name] <= time + eps:
                acquisition_time = self.next_sample[name]
                self.next_sample[name] += cfg.sample_period
                self.sample_count[name] += 1
                if self._streams[name]['dropout'].random() < cfg.dropout_probability:
                    self.dropout_count[name] += 1
                    continue
                jitter = self._streams[name]['delay'].uniform(-cfg.jitter, cfg.jitter)
                arrival_time = max(acquisition_time, acquisition_time + cfg.delay + jitter,
                                   self.last_arrival[name])
                self.last_arrival[name] = arrival_time
                packet = {
                    'channel': name,
                    'value': self._sample(name, true_state),
                    'acquisition_time': acquisition_time,
                    'arrival_time': arrival_time,
                    'sequence': self._sequence,
                }
                self._sequence += 1
                heapq.heappush(self._queue, (arrival_time, packet['sequence'], packet))
        due = []
        while self._queue and self._queue[0][0] <= time + eps:
            due.append(heapq.heappop(self._queue)[2])
        return due


class CommandDelayQueue:
    """FIFO command transport with bounded jitter and hold-last pending semantics."""

    def __init__(self, delay: float = 0.0, jitter: float = 0.0, seed: int = 0):
        values = (float(delay), float(jitter))
        if not all(math.isfinite(value) for value in values) or min(values) < 0:
            raise ValueError('command delay and jitter must be finite and nonnegative')
        self.delay, self.jitter, self.seed = values[0], values[1], int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.pending = deque()
        self.active_value = 0.0
        self.active_issue_time = None
        self.active_arrival_time = None
        self.active_apply_time = None
        self._last_arrival_time = -math.inf
        self._sequence = 0

    def issue(self, value: float, issue_time: float):
        value, issue_time = float(value), float(issue_time)
        if not math.isfinite(value) or not math.isfinite(issue_time):
            raise ValueError('command and issue time must be finite')
        jitter = self.rng.uniform(-self.jitter, self.jitter)
        arrival = max(issue_time, issue_time + self.delay + jitter,
                      self._last_arrival_time)
        self._last_arrival_time = arrival
        sequence = self._sequence
        self._sequence += 1
        self.pending.append({'value': value, 'issue_time': issue_time,
                             'arrival_time': arrival, 'sequence': sequence})
        applied_now = False
        while self.pending and self.pending[0]['arrival_time'] <= issue_time + 1e-12:
            command = self.pending.popleft()
            self.active_value = command['value']
            self.active_issue_time = command['issue_time']
            self.active_arrival_time = command['arrival_time']
            self.active_apply_time = issue_time
            applied_now = True
        return {
            'command_issue_time': issue_time,
            'command_arrival_time': arrival,
            'command_requested': value,
            'command_applied': self.active_value,
            'command_active_issue_time': self.active_issue_time,
            'command_active_arrival_time': self.active_arrival_time,
            'command_applied_time': self.active_apply_time,
            'command_pending_count': len(self.pending),
            'command_pending': bool(self.pending),
            'command_newly_applied': applied_now,
        }

# Asynchronous Sensor and Command Timing

Phase 3 adds timing impairments to the v4 estimator path and both v3/v4
actuator paths. The defaults retain the existing synchronous sensor updates and
zero-delay commands.

## Sensor configuration

`RLEnvConfigV4.sensor_timing` is a mapping from channel name to timing values.
Supported channels are `motor_angle`, `lever_angle`, `gyro`, and
`carriage_position` (the last requires `linear_encoder=True`). Values are in
seconds except `dropout_probability`:

```python
cfg = RLEnvConfigV4(
    sensor_timing={
        "motor_angle": {"sample_period": 0.01},
        "lever_angle": {"sample_period": 0.02, "delay": 0.008, "jitter": 0.002},
        "gyro": {"sample_period": 0.01, "delay": 0.015,
                 "jitter": 0.003, "dropout_probability": 0.05},
    }
)
```

Omitted channels use the control period with no delay, jitter, or dropouts.
Sample periods must be integer multiples of `physics_dt`, so rates may be
faster or slower than the controller; each channel has
independent seeded dropout, delay-jitter, and measurement-noise streams. Jitter
is uniform over `[-jitter, +jitter]`; arrival is never earlier than acquisition,
and each channel's packets remain ordered. The environment samples at physics
ticks; several packets can arrive between controller observations.

Packets retain `acquisition_time` and `arrival_time`. Packets become available
to the estimator at their arrival time and are consumed at the next controller
observation. A delayed EKF update is inserted at its acquisition time; replay
splits held-command transitions at physics ticks, then replays later
measurements through the current time. Its history horizon covers the maximum
configured sensor latency plus one control interval. `sensor_packets`, message
counts, and cumulative dropout counts are included in v4 step info.

An empty mapping, or explicit per-channel settings with the control period and
zero impairments, uses the existing synchronous measurement bundle. This keeps
the default seeded transition path unchanged.

## Command configuration

`RLEnvConfigV3` and `RLEnvConfigV4` provide `command_delay` and
`command_jitter`, independently of sensor settings. Both are nonnegative
seconds; jitter is uniform over `[-command_jitter, +command_jitter]`. Commands
are acceleration requests shared by torque and STEP/DIR modes. The queue keeps
issue order even when jitter would otherwise reorder commands.

Commands arriving between controller ticks are applied on the next control
tick. While a command is pending, the actuator holds the last active command;
at reset that safe command is zero acceleration. Every active command is
projected through the rail-safety constraint before conversion to torque or
STEP/DIR velocity. Step info reports the latest command's issue and scheduled
arrival times, the activation time and value of the active command, pending
count, and safety-projection saturation.

```python
cfg = RLEnvConfigV4(command_delay=0.012, command_jitter=0.002)
```

## Verification

`tests/test_async_timing.py` checks independent rates/dropouts, delayed EKF
replay, timestamp reporting, FIFO command application, safe pending behavior,
and explicit zero-impairment parity. Existing v4 and STEP/DIR regression tests
also cover the unchanged baseline path.

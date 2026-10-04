# RP2350 firmware architecture

## Partitioning

- **PIO/peripherals**: deterministic I/O.
- **Core 0**: deterministic state estimation and inner control.
- **Core 1**: trajectory generation, high-level control, experiments, USB/logging.

PIO should not contain the spring model. Use it for exact STEP timing.

## Core 0 fast path

```text
timestamp
 -> sensor snapshot
 -> angle unwrap
 -> encoder/gyro/IMU observer
 -> spring / geometry model
 -> force/torque estimate
 -> inner force/motion loop
 -> rail/motor safety
 -> motion command
 -> PIO queue
```

Avoid blocking USB, printf, flash writes, and dynamic allocation here.

## Core 1

- trajectory generation
- LQR / LQG
- MPC
- RL inference
- haptic effects
- system identification
- experiment sequencing
- USB and telemetry

## Suggested initial rates

- STEP: event-driven PIO
- IMU: ~2–4 kHz
- state prediction: ~2–4 kHz
- inner loop: ~1–2 kHz
- AS5600 correction: ~1 kHz starting point
- LQR/haptics: ~500 Hz–1 kHz
- MPC: ~100–500 Hz
- RL inference: hundreds of Hz to ~1 kHz

## State estimation progression

1. encoder only
2. filtered finite-difference velocity
3. encoder + gyro complementary estimator
4. linear Kalman filter
5. EKF only if actually necessary

Do not use the accelerometer as a simple inclinometer during aggressive base motion; the measurement includes base, tangential, centripetal, and gravity terms.

## Inter-core communication

Use a sequence-numbered double-buffer or lock-free snapshot. Core 0 should remain safe if Core 1 temporarily stalls.

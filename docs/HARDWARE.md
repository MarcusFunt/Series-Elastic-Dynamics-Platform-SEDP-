# Hardware design

## Mechanical stack

```text
NEMA17 -> GT2 pulley -> GT2 belt -> MGN7 carriage
                                  |
                                  +-> pivoted 1-DOF resonator
                                      /          \
                                   spring      spring
                                     \          /
                                      carriage anchors
```

Baseline choices:

- 2020 extrusion
- ~150 mm MGN7 rail
- optional second rail if torsional rigidity requires it
- GT2 belt and ~20T pulley as a starting point
- NEMA17 stepper
- interchangeable resonator mass
- intentionally low passive damping

## Electronics

RP2350 is the baseline controller.

Buses:

- I2C0 -> AS5600 motor encoder
- I2C1 -> AS5600 resonator encoder
- SPI -> 6-axis IMU + data-ready
- GPIO -> left/right travel limits
- UART -> TMC2209 configuration/diagnostics
- PIO -> STEP/DIR
- USB -> telemetry / experiment control
- SWD -> debug

## Power

```text
24 V input
  +-- TMC2209 / stepper
  +-- 5 V buck
        +-- 3.3 V regulator
              +-- RP2350 + sensors
```

## Position sensing hierarchy

Keep separate:

- commanded step position
- measured motor position
- actual carriage position
- resonator angle

A direct carriage encoder is optional in V1 but becomes valuable if belt dynamics become important.

## Series-elastic force calibration

Apply known dead weights:

\[
F=mg.
\]

Record both encoder states and fit the force/torque relationship of the **assembled mechanism**, including geometry and friction.

## Safety

Use:

1. mechanical end stops;
2. independent end-limit sensors;
3. software braking envelope / predicted stopping distance;
4. commanded-vs-measured motor tracking fault;
5. watchdog behavior if the supervisory core stalls.

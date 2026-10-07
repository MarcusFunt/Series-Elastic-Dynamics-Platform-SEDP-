# MPC Runtime and Imitation Evaluation

- Source revision: `8529ded7e0fe7e97ba9ed7c0a2dc4ae993fc580c`
- Source fingerprint: `2478e16f54bc1c225661f5d8659372ca17e7cd4e44859d652c94d32ad23c11b8`
- Task: random target holds, 15 mm edge exclusion, at least 2.5 s training holds and 2 s evaluated in-tolerance holds.
- Control period: 10.0 ms; actuator: `torque`.

## MPC optimization

- Decision: kept stride 1.
- Mean per-seed p95 solve time: 7.394 ms at stride 1, 8.679 ms at stride 2.
- p95 improvement: -17.4%.
- Paired behavior gate: failed.

| Seed | stride 1 p95 (ms) | stride 2 p95 (ms) | position Δ (mm) | angle RMS Δ (°) | peak rail Δ | energy ratio | deadline miss Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7001 | 7.338 | 7.747 | +0.0182 | +0.00201 | -0.0001 | 1.0064 | +0.0025 |
| 7002 | 7.713 | 8.342 | +0.0005 | +0.00019 | -0.0001 | 1.0034 | +0.0108 |
| 7003 | 6.845 | 9.383 | +0.0012 | +0.00024 | -0.0002 | 1.0031 | +0.0300 |
| 7004 | 7.119 | 8.974 | +0.0037 | -0.00160 | +0.0001 | 0.9844 | +0.0192 |
| 7005 | 7.955 | 8.950 | +0.0086 | +0.00170 | -0.0001 | 1.0118 | +0.0108 |

- Fresh rerun of selected stride 1 on seeds 9001, 9002, 9003: mean p95 10.935 ms; stride 2 did not pass the initial paired gate.

Reasons for retaining stride 1:

- seed 7002: 10 ms deadline-miss fraction regressed by more than 0.01
- seed 7003: 10 ms deadline-miss fraction regressed by more than 0.01
- seed 7004: 10 ms deadline-miss fraction regressed by more than 0.01
- seed 7005: 10 ms deadline-miss fraction regressed by more than 0.01
- mean per-seed p95 solve latency improved by less than 10%

## MPC imitation

- Best held-out action MAE: 0.005887 normalized action units.
- Successful labels used: 9584; fallback labels excluded: 16.
- Mean MPC p50/p95/p99 solve time: 6.180/10.156/12.709 ms.
- Mean imitation p50/p95/p99 action time: 0.3652/0.7221/1.0424 ms; p95/p99 speedup: 14.1×/12.2×.
- Mean 10 ms deadline misses: MPC 0.0597, imitation 0.0000.
- Closed-loop performance gates: not all passed.
- Speed/performance usefulness gate: not passed.
- Mean integrated resonator energy ratio vs MPC: 1.0452.

| Seed | MPC p50/p95/p99 (ms) | imitation p50/p95/p99 (ms) | position Δ (mm) | angle RMS Δ (°) | rail Δ | energy ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8011 | 6.544/10.492/12.350 | 0.3643/0.8240/1.1424 | +0.0025 | +0.00380 | -0.0005 | 1.0933 |
| 8012 | 6.334/11.364/14.905 | 0.3797/0.7849/1.0040 | +0.0024 | +0.00235 | -0.0007 | 1.0231 |
| 8013 | 6.039/9.800/12.611 | 0.3752/0.7295/1.0450 | +0.0079 | +0.00148 | +0.0003 | 1.0557 |
| 8014 | 6.576/11.082/13.041 | 0.3772/0.7769/1.1814 | +0.0040 | +0.00269 | +0.0007 | 1.0455 |
| 8015 | 6.264/10.179/12.753 | 0.3963/0.7692/1.1094 | +0.0037 | +0.00222 | +0.0007 | 1.0199 |
| 8016 | 6.367/10.703/13.638 | 0.3787/0.7289/1.0736 | +0.0210 | +0.00264 | +0.0006 | 1.0704 |
| 8017 | 6.177/10.682/13.367 | 0.3782/0.7571/1.1653 | -0.0012 | +0.00071 | +0.0004 | 1.0065 |
| 8018 | 6.274/10.428/13.334 | 0.3711/0.7024/1.0068 | +0.0142 | +0.01307 | +0.0002 | 1.1115 |
| 8019 | 6.051/9.430/12.155 | 0.3486/0.6722/1.0578 | -0.0078 | -0.00025 | +0.0005 | 1.0008 |
| 8020 | 5.169/7.401/8.938 | 0.2832/0.4754/0.6384 | -0.0012 | +0.00113 | -0.0000 | 1.0249 |

Gate findings:

- seed 8011: integrated resonator energy exceeded MPC by more than 2%
- seed 8012: integrated resonator energy exceeded MPC by more than 2%
- seed 8013: integrated resonator energy exceeded MPC by more than 2%
- seed 8014: integrated resonator energy exceeded MPC by more than 2%
- seed 8016: integrated resonator energy exceeded MPC by more than 2%
- seed 8018: peak angle exceeded paired MPC bound
- seed 8018: integrated resonator energy exceeded MPC by more than 2%
- seed 8020: integrated resonator energy exceeded MPC by more than 2%

## Artifacts

- Full JSON evidence: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\docs\control\evidence\mpc_runtime_imitation_20261007.json`
- Run directory: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase6\mpc_runtime_imitation_20261007`
- Timing measurements are Python simulation-host CPU measurements; they do not certify deployment-hardware real-time latency.
- Energy is integrated resonator mechanical energy, not motor electrical consumption.

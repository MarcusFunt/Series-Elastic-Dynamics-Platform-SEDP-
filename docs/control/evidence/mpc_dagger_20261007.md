# MPC DAgger Evaluation

- Status: completed
- Source study: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase6\mpc_runtime_imitation_20261007`
- Runtime dependency SHA256: `a7f0d201e2d6174b73410f5c0ac2c1e6c8f0e709ac1f042d5c46e3058e248104` (12 dependency files).
- Task: random goal hold, 15 mm edge exclusion, 2.5 s minimum target hold.
- MPC: horizon 8, linearization stride 1.
- DAgger rollout seeds: `[42420, 43429, 44438, 45447, 46456, 47465, 48474, 49483, 50492, 51501, 52510, 53519, 54528, 55537, 56546, 57555]`
- Fresh evaluation seeds: `[12011, 12012, 12013, 12014, 12015, 12016, 12017, 12018, 12019, 12020]`

## Training

Collected 9600 student-rollout states; 9567 MPC labels were successful and 33 fallback labels were excluded.
Combined dataset: 19200 labels (9600 original, 9600 DAgger); 49 unsuccessful MPC labels were excluded from the combined fit. Best episode-held-out action MAE: 0.010035.
Warm-start checkpoint: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase6\mpc_runtime_imitation_20261007\teacher\policy_candidate.pt`.

## Paired comparison against MPC

| Student | Passed behavior gates | p95 speedup vs MPC | Mean energy ratio vs MPC | Failed paired gates |
|---|---:|---:|---:|---:|
| Original | False | 14.87× | 1.0352 | 11 |
| DAgger | False | 14.59× | 1.0327 | 10 |

## DAgger change on paired fresh seeds

DAgger/original mean resonator-energy ratio: 0.9974; lower energy on 7/10 seeds and within ±2% on 7/10. Policy action p95 means: 0.5816 ms (DAgger), 0.5705 ms (original).

## Per-seed DAgger vs MPC

| Seed | Energy ratio | Position RMSE change (mm) | Angle RMS change (deg) | Holds | Gate failures |
|---:|---:|---:|---:|---:|---|
| 12011 | 1.0447 | +0.0076 | +0.0028 | 3/3 | integrated resonator energy exceeded MPC by more than 2% |
| 12012 | 1.0989 | +0.0081 | +0.0052 | 3/3 | integrated resonator energy exceeded MPC by more than 2% |
| 12013 | 1.0204 | +0.0060 | +0.0011 | 3/3 | integrated resonator energy exceeded MPC by more than 2% |
| 12014 | 1.0155 | +0.0118 | +0.0011 | 3/3 | none |
| 12015 | 1.0211 | +0.0694 | +0.0038 | 3/3 | position tracking exceeded paired MPC bound; integrated resonator energy exceeded MPC by more than 2% |
| 12016 | 1.0460 | +0.0035 | +0.0025 | 3/3 | peak angle exceeded paired MPC bound; integrated resonator energy exceeded MPC by more than 2% |
| 12017 | 1.0070 | -0.0193 | +0.0001 | 3/3 | none |
| 12018 | 0.9721 | -0.1635 | -0.0039 | 3/3 | none |
| 12019 | 1.1193 | +0.0132 | +0.0076 | 3/3 | peak angle exceeded paired MPC bound; integrated resonator energy exceeded MPC by more than 2% |
| 12020 | 0.9817 | -0.0836 | -0.0029 | 3/3 | peak angle exceeded paired MPC bound |

JSON evidence: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\docs\control\evidence\mpc_dagger_20261007.json`

# PPO energy reduction: accepted baseline result

Checkpoint: `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase5\energy_alignment_mpc_20261006\zero_init_ppo\weight32_seed389\policy_candidate.pt`

- SHA-256: `5d7ef2805c9c016c38c98b3757ac876b8118a1087c2d9a07544912d69c23f30f`
- Training: seed 389, 65,536 PPO steps from zero residual initialization; learning rate 3e-4; anchor KL 0.001; energy gate weight 32; original 2.5 m/s² residual acceleration limit.
- The checkpoint itself is directly trained; no parameter interpolation or post-training authority scaling was applied.

## Development evaluation

- Randomized reference seeds: `1001, 1002, 1003`; paired cases: 6.
- Mean policy/LQR energy ratio: **0.931721** (6.83% lower energy).
- Promotion accepted: **True**; rejection reasons: `[]`.

## Independent confirmation

- Randomized reference seeds: `8001, 8002, 8003`; paired cases: 6.
- Mean policy/LQR energy ratio: **0.924384** (7.56% lower energy).
- Promotion accepted: **True**; rejection reasons: `[]`.

## Repository MPC comparison

- Matched-case policy/MPC energy ratio: **0.516535** (policy uses **48.35%** less energy).
- MPC mean p95 solve time: **8.99 ms**; mean fallback fraction: **0.19%**.

## Provenance

The trainer recorded source revision `36ee7c3af0bc09081101c8a437d3e96f8c0ef5a9`; the source files used in this working tree are fingerprinted in `baseline_win_summary.json`.

Evaluation artifacts:

- `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase5\energy_alignment_mpc_20261006\zero_init_ppo\weight32_seed389\evaluation_dev.json`
- `C:\Users\marcu\OneDrive\Dokumenter\GitHub\Series-Elastic-Dynamics-Platform-SEDP-\runs\phase5\energy_alignment_mpc_20261006\zero_init_ppo\weight32_seed389\evaluation_confirm_8001_8003.json`

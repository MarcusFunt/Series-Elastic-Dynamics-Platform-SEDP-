# Phase 5 PPO Ablation Results

Source revision: `4165d631c24f6f5ad5936ab7e9ec500cdc701509`

Training steps per seed: 16,384; training seeds: 173, 271, 389; shared held-out seeds: 1001, 1002, 1003.

Metrics below average each policy over the same three fixed motion scenarios and three randomized held-out references. The uncertainty column is the sample standard deviation across the three independent training seeds.

| Configuration | Position RMSE (mm) | Angle RMS (deg) | Energy integral (mJ·s) | Completion | Δ energy vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline_estimated_history8_preview | 11.411 ± 0.004 | 1.613 ± 0.001 | 1.488 ± 0.002 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| oracle_state_upper_bound | 11.410 ± 0.002 | 1.615 ± 0.000 | 1.512 ± 0.000 | 1.000 ± 0.000 | 0.024 ± 0.002 |
| estimated_history1 | 11.410 ± 0.002 | 1.613 ± 0.000 | 1.489 ± 0.001 | 1.000 ± 0.000 | 0.000 ± 0.003 |
| estimated_history8_no_preview | 11.410 ± 0.002 | 1.613 ± 0.000 | 1.490 ± 0.001 | 1.000 ± 0.000 | 0.002 ± 0.002 |

Lower values are better for position error, angle, energy, rail/speed violations, and saturation. Higher completion is better. Oracle-state results are an information upper bound and are not eligible for deployment. No ablation is promoted unless every existing `evaluate_v4.promotion` safety and paired-performance gate accepts it.

## Promotion Results

`mean_energy_ratio` is the arithmetic mean of six paired ratios per training seed: policy energy integral divided by LQR energy integral for each of the three fixed scenarios and three held-out randomized cases. A value of `0.95` means 5% less integrated resonator energy than LQR. The table reports the mean and sample standard deviation of that ratio across the three independent training seeds.

| Configuration | Mean energy ratio vs LQR | Accepted candidates |
|---|---:|---:|
| baseline_estimated_history8_preview | 0.995996 ± 0.002386 | 0 / 3 |
| oracle_state_upper_bound | 0.999890 ± 0.000241 | 0 / 3 |
| estimated_history1 | 0.996622 ± 0.001938 | 0 / 3 |
| estimated_history8_no_preview | 0.996649 ± 0.000884 | 0 / 3 |

All twelve candidates completed all six evaluation cases with no saturation or stop contact. Each candidate passed the safety and tracking checks but was rejected because it did not reach the required 5% mean energy reduction (`mean_energy_ratio <= 0.95`). These small differences are inconclusive at this training budget; they do not establish that the configurations are equivalent or that PPO cannot improve further.

Raw evaluation rows and per-run training logs are embedded in `docs\control\evidence\ppo_ablation_v5.json`. Checkpoint and full per-run artifacts remain under `runs\phase5\ppo_v5` (ignored by Git).

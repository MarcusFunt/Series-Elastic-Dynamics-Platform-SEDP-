# MPC Initialized PPO Energy Study

## Goal

Restart the energy-alignment study using the modified PPO defaults and the repository's `ConstrainedMPC` implementation.

## Implementation

1. Train one MPC teacher per training seed using the existing teacher collection and distillation path.
2. Initialize all pilot and full PPO runs for that seed from its teacher checkpoint.
3. Run standalone MPC evaluations on the development and final seed sets, and report PPO energy ratios against both LQR and MPC.
4. Record teacher settings, MPC horizon, modified PPO settings, and relevant source hashes in the protocol.
5. Launch the study into a new run directory and verify its process and initial artifacts.

## Verification

Compile the modified Python files and run `git diff --check`. Inspect the launch protocol, status, and process to confirm the new run uses the expected code and MPC teacher path.

## Outcome

The MPC-distilled study was run first but did not produce a policy that cleared the paired LQR promotion gates. The improved PPO settings and energy objective were then evaluated with zero-residual initialization; the repository MPC remained in the paired controller comparisons. The selected seed-389 PPO checkpoint passed both the development set and an independent confirmation set. Its exact training configuration, checkpoint fingerprint, gate results, and MPC comparison are recorded in `runs/phase5/energy_alignment_mpc_20261006/zero_init_ppo/weight32_seed389/baseline_win_summary.md` and the adjacent JSON file.

The winning checkpoint itself is directly trained from zero-residual initialization; it is not an MPC-distilled checkpoint. The MPC-distilled route was retained as a failed pilot, not misreported as the winning method.

Compilation and `git diff --check` passed. The evaluation scripts reported promotion acceptance on development seeds 1001–1003 and fresh confirmation seeds 8001–8003. No training process remains active.

# MPC DAgger Plan

> **For agentic workers:** Execute inline in the existing user-approved worktree. Track steps and evidence in `.superpowers/sdd/2026-10-07-mpc-dagger/progress.md`.

**Goal:** Improve the MPC imitation policy by collecting expert labels on states reached by the current student, fine-tune the student on those labels plus the original MPC data, and compare both students with MPC on fresh, disjoint random-goal-hold seeds.

**Architecture:** Extend `train_residual_v4.py` with a DAgger mode that loads the complete saved environment, plant, and MPC configuration from the existing benchmark; collects MPC actions at student-visited states while applying student actions; excludes unsuccessful MPC labels; and warm-starts distillation from the existing student. Add a finite benchmark driver that verifies seed separation, runs collection/training, evaluates MPC and both students under identical held-out conditions, and writes JSON plus Markdown evidence.

**Tech Stack:** Python, NumPy, PyTorch, existing randomized goal-hold environment and constrained MPC.

**Spec:** User request in the current conversation. Keep the 15 mm edge exclusion, random targets, at least 2.5 s target hold, selected stride-1 MPC configuration, and existing task and plant settings from the committed study.

## Global Constraints

- Use the exact serialized environment, MPC config, and plant parameters from `runs/phase6/mpc_runtime_imitation_20261007/teacher/teacher_config.json`.
- DAgger collection applies deterministic student actions to the environment and records MPC actions for the states observed before each action.
- Keep MPC fallback labels out of supervised fitting and report their count.
- Split validation by whole episodes; warm-start from `policy_candidate.pt`.
- DAgger collection seeds and final evaluation seeds must not overlap each other or any prior optimization, confirmation, teacher, or evaluation seeds.
- Evaluate MPC, the original student, and DAgger student on exactly the same fresh seeds. Report latency and closed-loop metrics separately.

## Review Focus

- Verify recorded states are on student rollouts while labels are MPC actions and transitions apply student actions.
- Verify warm-start weights, config parity, fallback filtering, group-disjoint validation, and seed separation.
- Verify the final benchmark is paired and records reproducible configuration, model paths, seeds, and results.

### Task 1: Add failing unit tests

**Files:** Create `tests/test_mpc_dagger.py`.

- [x] Check DAgger queries MPC labels at visited observations while applying student actions.
- [x] Check warm-started distillation and episode-disjoint split behavior.
- [x] Check seed protocol rejects collisions and accepts the selected disjoint seed sets.

### Task 2: Implement DAgger collection and warm-started training

**Files:** Modify `software/python/train_residual_v4.py`.

- [x] Add `collect_dagger` and DAgger mode using the saved complete config.
- [x] Combine original teacher labels with successful DAgger labels and offset episode groups.
- [x] Fine-tune from the existing student and persist collection, fit, config, and metadata artifacts.

### Task 3: Build and run the paired fresh-seed benchmark

**Files:** Create `software/python/benchmark_mpc_dagger.py`; add tests as needed.

- [x] Verify seed groups are disjoint before execution.
- [x] Collect 9,600 DAgger labels over 16 student-rollout episodes and fine-tune for 40 epochs.
- [x] Evaluate MPC, original student, and DAgger student on 10 fresh seeds with paired configuration.
- [x] Save run artifacts under `runs/phase6/mpc_dagger_20261007/` and a report under `docs/control/evidence/`.

### Task 4: Verify and review

- [x] Run focused tests and the full test suite; record any environment-only failures.
- [x] Inspect final artifacts and code diff; complete the required fresh review pass and correct the two-percent summary and source dependency fingerprint.
- [x] Summarize whether DAgger improved performance and whether the student meets the MPC behavior and latency gates.

## Outcome

DAgger collected 9,600 student-rollout observations, excluded 33 unsuccessful MPC labels, and warm-started on the 9,600 original labels. The combined fit excluded 49 fallback labels in total and reached best episode-held-out action MAE 0.01003. On 10 fresh seeds the DAgger policy was 14.59× faster at p95 than MPC, completed all target holds, and had no exclusion violations, but mean resonator energy remained 3.27% above MPC and the strict paired behavior gate failed on 7/10 seeds. Compared with the original student on those same seeds, mean energy was 0.26% lower. Evidence: `docs/control/evidence/mpc_dagger_20261007.md` and `.json`.

# MPC Runtime and Imitation Plan

> **For agentic workers:** Execute inline in the existing user-approved worktree. Track steps and evidence in `.superpowers/sdd/2026-10-07-mpc-runtime-imitation/progress.md`.

**Goal:** Measure current MPC latency on the random-goal hold task, reduce runtime while preserving closed-loop performance, then train and evaluate an MPC imitation policy on that same task.

**Architecture:** Reuse the existing random-goal-hold evaluator and teacher/distillation pipeline. Establish a paired MPC baseline, benchmark a reduced torque-model linearization stride against it, and keep the optimized setting only if behavior gates hold. Make the teacher, recorded config, and evaluator share the same MPC settings; measure student action latency alongside MPC solve latency.

**Tech Stack:** Python, NumPy, SciPy SLSQP, Numba RK4 kernel, PyTorch actor, existing JSON experiment artifacts.

**Spec:** User request in the current conversation; random targets, at least two seconds settled per target, and 15 mm edge exclusion carry forward from the project specification.

## Global Constraints

- The plant control interval is 10 ms.
- Training uses `random_goal_hold`, 15 mm edge margin, and at least 2.5 s minimum target hold.
- Evaluation requires the existing two-second in-tolerance hold criterion and paired random seeds.
- MPC fallback demonstrations remain excluded from supervised fitting.
- Report simulated resonator energy separately from motor or electrical energy.

## Review Focus

- Teacher and evaluation MPC objectives/configuration must match; verify saved config and runtime construction.
- Fallback labels must be excluded from training; verify recorded counts.
- Optimization must preserve per-seed safety, exclusion, hold, tracking, and angle gates; verify before training on the selected configuration.
- Report policy inference timing and MPC solve timing as separate measures; verify both fields are present and bounded.
- Evaluation seeds must be held out from teacher episodes; verify disjoint seed lists in the protocol.

### Task 1: Establish and instrument the MPC baseline

**Files:** Modify `software/python/evaluate_goal_hold_v4.py`; create baseline evidence under `runs/phase6/mpc_runtime_imitation_20261007/`.

- [ ] Run the existing goal-hold baseline on paired evaluation seeds; record per-action MPC p50/p95/p99, 10 ms deadline misses, fallback rate, tracking, energy, exclusion, and hold outcomes.
- [ ] Preserve the baseline JSON and command/config metadata.

### Task 2: Benchmark a fidelity-preserving MPC optimization

**Files:** Modify `software/python/constrained_mpc.py` only if measurements justify it; extend `tests/test_constrained_mpc_phase4.py` for configuration and safety behavior; record candidate evidence beside baseline.

- [ ] Compare default torque linearization stride 1 with stride 2 on the exact same seeds.
- [ ] Keep the faster stride only if every paired run retains target completion, exclusion, safety, tracking, and angle gates against stride 1.
- [ ] Re-run the selected setting to confirm the result and report latency percentiles and fallback changes.

### Task 3: Train an MPC imitation policy with matching teacher settings

**Files:** Modify `software/python/train_residual_v4.py` and `software/python/evaluate_goal_hold_v4.py` as needed; save artifacts under `runs/phase6/mpc_runtime_imitation_20261007/`.

- [ ] Collect MPC actions under the selected optimized configuration with the random-goal-hold task, the 15 mm exclusion margin, and at least 2.5 s target holds.
- [ ] Distill a deterministic actor using episode-disjoint validation; record teacher action error and the number of excluded fallback labels.
- [ ] Evaluate MPC, LQR, and imitation on held-out seeds with identical episode seeds/configuration; record controller latency distributions and closed-loop metrics.

### Task 4: Report the recommendation

- [ ] Save a concise Markdown report linking source revision, commands, settings, artifacts, per-seed results, and whether the imitation policy meets the performance/speed tradeoff.
- [ ] Re-run required verification and inspect the final diff before committing.

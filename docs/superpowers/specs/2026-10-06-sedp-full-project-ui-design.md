# SEDP Full-Project Visualization Workspace

## Problem

The current browser UI exposes a live plant simulator and a limited v4 training panel. The repo contains substantially more project content—models, controller comparisons, MPC, estimation and timing, RL studies, benchmark evidence, OpenModelica, firmware architecture, haptics, and the planned physical rig—but these areas are not discoverable or explorable through the GUI. The recent random-goal-and-hold experiment is also not represented in the current training view.

## Goals

- Make the project’s complete scope navigable from one browser workspace.
- Let users inspect live simulation, recorded experiment evidence, and source-defined architecture in meaningful diagrams and charts.
- Make data provenance and maturity visible: **Live simulation**, **Recorded evidence**, **Model/source view**, and **Planned / hardware unavailable**.
- Surface the random-goal-and-hold task, its two-second in-tolerance success rule, the 15 mm edge exclusion, and the repo MPC comparison.
- Play the three saved videos from early, middle, and final checkpoints of the selected PPO run.
- Keep existing simulator and training controls and their APIs working.
- Make the interface adapt to narrow viewports and accessible through keyboard and semantic HTML.

## Out of scope

- Claiming physical telemetry before the RP2350 rig is connected and validated.
- Replacing model, training, evaluation, or firmware implementations with GUI-only mock data.
- Adding new training objectives or changing benchmark acceptance gates.
- Changing scientific results or accepting results by changing display thresholds.

## Product structure

Use a persistent project-wide navigation shell with these views:

1. **Overview** — subsystem map, current validation state, live/recorded/planned counts, links into key data, and a recent evidence feed.
2. **Rig simulator** — existing interactive SVG plant, trajectories, controller selection, plant controls, and live traces.
3. **Digital twin** — reduced/nonlinear/Modelica model coverage, physical parameter map, energy/physics validation plots where source evidence exists, and explicit placeholder/calibration labels.
4. **Control lab** — servo, damping, LQR, repository MPC, and accepted/experimental learned policy comparisons on available matched scenarios. Show controller details and MPC fallback/solve timing where recorded.
5. **Sensors & timing** — encoder/gyro/IMU and command/sensor timing architecture, estimator flow, sample rates and recorded delay/timing results when present. Mark hardware channels as simulated or planned until measured.
6. **Learning & experiments** — training controls, run status, PPO versions and checkpoints, teacher/distillation, benchmark gates, random-goal-and-hold results, downloadable artifacts, and training-stage videos. Each result links to its configuration and evidence source.
7. **Hardware & firmware** — mechanical/electrical block diagrams, Core 0/Core 1/PIO partition, planned rates and safety path, build/validation state, and clear hardware-not-connected status.
8. **Haptics & system identification** — supported concepts and model/identification workflows; distinguish implemented simulation capability from planned physical validation.
9. **Project map** — searchable source/document/evidence catalog so no subsystem is hidden behind the first few UI tabs.

## Information design

- Keep a consistent header, active module navigation, status badges, units, and data-source labels.
- Every chart identifies its source, scenario, controller, units, and whether values are live or recorded.
- Comparison plots pair controllers on the same scenario/seeds when the stored evidence permits it; otherwise do not imply a paired comparison.
- Include clear empty states: no hardware telemetry, no measurement data, no matching evidence, or module is planned.
- Show important constraints beside results: random target range respects the 15 mm exclusion zone; a successful hold is at least two continuous seconds in tolerance; report the experiment’s actual configured minimum dwell separately.
- Avoid exposing raw logs as the only view. Keep downloadable source artifacts available alongside readable summaries and plots.

## Data and architecture

- Retain the existing FastAPI app and current WebSocket/API contract.
- Move the page into maintainable static UI assets (HTML, CSS, and browser JavaScript modules) served by the existing app; preserve the existing training workflow and policy-load path.
- Add a read-only project catalog endpoint backed by an explicit manifest of subsystems, status, source paths, and supported visualizations.
- Add read-only evidence inventory/summary endpoints that expose only committed or local project artifacts intended for display. Parse known JSON/CSV evidence with schema checks; do not recursively publish arbitrary filesystem content.
- Continue using the live simulator snapshot/WebSocket for real-time state. Render recorded evidence separately so it cannot be mistaken for the live plant.
- Serve generated videos only through an explicit media allowlist so local run folders are not exposed as browsable paths.
- Use Plotly already present in the browser page and native SVG/CSS diagrams; do not introduce a frontend build tool unless the current repository requires one.

## Key user flows

### Explore project coverage

Open Overview → select a subsystem badge/card → inspect its diagram, data availability, and linked evidence/source → navigate to its live or recorded view.

### Compare controllers

Open Control lab → select a supported scenario/evidence set → view matched traces and benchmark metrics → inspect provenance and download the underlying evidence.

### Review learning result

Open Learning & experiments → choose the random-goal-and-hold run → view the randomized target sequence, 15 mm exclusion zone, two-second-in-tolerance holds, PPO/LQR/MPC metrics, promotion result, and configuration/artifact links.

### Distinguish model from hardware

Open Hardware & firmware → inspect architecture/rates/safety → see a persistent “hardware not connected / no measured telemetry” state until validated live telemetry exists.

## Acceptance criteria

- Every current and planned top-level subsystem documented in README/project overview has an entry in the GUI project map.
- Every visualized datum is tagged as live, recorded, or source/model-defined, and every planned or missing data source has a clear empty/status state.
- The existing interactive simulator and v4 training actions remain available through the redesigned page.
- The latest random-goal-and-hold evidence is discoverable and displays its actual task constraints and experiment metadata without conflating it with classic B1 results.
- Users can navigate and resize the complete interface without content being hidden below an inaccessible fixed-height section.
- The interface contains no synthetic physical telemetry or unsupported performance claims.

## Risks and mitigations

- **Evidence schemas vary:** use explicit adapters per known artifact format and show unsupported formats as source links instead of guessing.
- **Scope grows to a second simulator:** separate live, recorded, and planned content and prioritize navigable coverage over duplicating the underlying physics UI.
- **Single-page performance degrades:** lazy-render non-active charts and load evidence only when its view opens.
- **Old run paths are machine-specific:** treat them as metadata, display repo-relative links where available, and handle missing local outputs gracefully.

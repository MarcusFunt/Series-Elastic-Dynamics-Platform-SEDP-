# SEDP Full-Project Visualization Workspace Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after the user reviews it and selects an execution method.

**Goal:** Expand the browser application from a simulator/training page into a project-wide, evidence-backed visualization workspace for every SEDP subsystem.

**Architecture:** Keep FastAPI, simulator state streaming, and training APIs as the backend. Serve maintainable static HTML/CSS/JavaScript modules and add explicit read-only catalog/evidence APIs. Build each view from a status/source manifest and typed adapters for committed evidence, keeping live, recorded, model-defined, and planned data distinct.

**Tech Stack:** Python, FastAPI, existing WebSocket, HTML/CSS, browser JavaScript modules, existing Plotly.js, native SVG.

## Task 1: Inventory and model the project visualization catalog

**Files:**
- Create: `software/python/ui/project_catalog.json`
- Create: `software/python/project_catalog.py`
- Modify: `software/python/active_vibration_rig_web.py`
- Reference: `README.md`, `docs/PROJECT_OVERVIEW.md`, `docs/FIRMWARE.md`, `docs/OPENMODELICA.md`, `docs/HARDWARE.md`

1. Add explicit entries for simulation, digital twin, controls/MPC, sensing/timing/estimation, learned control, benchmark/evidence, firmware, physical hardware, haptics, and system identification.
2. Record each entry’s maturity, source/document paths, available data types, and supported visualizations.
3. Implement a read-only catalog API and path-safe evidence/video inventory limited to declared roots and file types.
4. Add schema-aware adapters for the current random-goal-and-hold confirmation result and the fixed B1 result.
5. Manually inspect catalog coverage against README/project overview and ensure absent live data is reported as unavailable.

## Task 2: Extract and preserve the existing simulator/training interface

**Files:**
- Create: `software/python/ui/index.html`
- Create: `software/python/ui/styles.css`
- Create: `software/python/ui/js/app.js`
- Create: `software/python/ui/js/simulator.js`
- Create: `software/python/ui/js/training.js`
- Modify: `software/python/active_vibration_rig_web.py`
- Modify: `software/python/training_ui.html` (retire after migration or keep only a compatibility include)

1. Move the current SVG simulator, WebSocket handling, charts, and training actions into static assets without changing API behavior.
2. Serve UI assets through FastAPI with the required MIME types and safe local paths.
3. Confirm visually that simulator controls, live state, CSV export, training polling/cancel, artifact downloads, and checkpoint loading remain accessible.

## Task 3: Build the project navigation shell and overview

**Files:**
- Create: `software/python/ui/js/navigation.js`
- Create: `software/python/ui/js/overview.js`
- Modify: `software/python/ui/index.html`
- Modify: `software/python/ui/styles.css`

1. Implement persistent navigation for Overview, Rig, Digital twin, Control lab, Sensors & timing, Learning & experiments, Hardware & firmware, Haptics & system identification, and Project map.
2. Load catalog data and render coverage/status tiles with live/recorded/model/planned badges.
3. Add accessible responsive navigation, focus states, and clear no-data states.
4. Manually inspect at desktop and narrow viewport sizes for hidden modules or unreachable content.

## Task 4: Add model, controller, estimator, and timing views

**Files:**
- Create: `software/python/ui/js/models.js`
- Create: `software/python/ui/js/controls.js`
- Create: `software/python/ui/js/sensors.js`
- Modify: `software/python/ui/index.html`
- Modify: `software/python/ui/styles.css`
- Modify: `software/python/project_catalog.py`

1. Render source-backed model coverage, parameters, OpenModelica placeholders, and available physics validation evidence.
2. Display controller benchmark comparisons and MPC solve/fallback fields only when present in evidence.
3. Render sensor, estimator, and timing architecture with planned rates separate from measured runtime channels.
4. Link each chart/diagram to its evidence or source and show scenario, controller, seed, and units.

## Task 5: Add complete experiment and learning evidence exploration

**Files:**
- Create: `software/python/ui/js/experiments.js`
- Modify: `software/python/project_catalog.py`
- Modify: `software/python/ui/index.html`
- Modify: `software/python/ui/styles.css`

1. Add experiment selection and readable summaries for existing PPO generations, MPC references, B1/V4/Phase5 evidence, and random-goal-and-hold.
2. Plot supported evaluation metrics and traces from actual stored evidence; expose downloadable source artifacts and embed the saved 8,192-, 32,768-, and 65,536-step PPO videos through the explicit media allowlist.
3. Show target range, 15 mm exclusion zone, dwell requirement, continuous in-tolerance criterion, comparison controllers, seeds, and promotion status from run metadata.
4. Handle missing local checkpoints/logs with graceful states and retain provenance without leaking absolute machine paths.

## Task 6: Visualize hardware, firmware, haptics, and project map

**Files:**
- Create: `software/python/ui/js/architecture.js`
- Create: `software/python/ui/js/project-map.js`
- Modify: `software/python/ui/index.html`
- Modify: `software/python/ui/styles.css`
- Modify: `software/python/project_catalog.py`

1. Render RP2350 Core 0/Core 1/PIO data flow, safety path, sensors, driver, and planned sample rates from firmware/hardware documentation.
2. Show hardware validation and calibration state; keep physical telemetry unavailable until the repo receives a real connection/telemetry source.
3. Show haptic modes and system-identification stages with implemented/planned status.
4. Add searchable project map across subsystem sources, docs, and declared evidence.

## Task 7: Visual review and documentation

**Files:**
- Modify: `docs/BROWSER_UI.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-06-sedp-full-project-ui-design.md`

1. Launch the browser app locally and manually navigate every module.
2. Confirm chart provenance labels, missing-data states, units, responsive layout, and existing simulator/training flows.
3. Update run instructions and document which views are live, recorded, model-defined, and planned.
4. Review the implementation diff for accidental changes to experiments or benchmark data.

## Completion criteria

- All subsystem entries from the design are visible and navigable from the browser.
- Actual project evidence drives all result charts and comparisons.
- Planned and unavailable sources are explicit; no physical measurements are fabricated.
- Existing simulation and training workflows are preserved.
- Documentation explains startup and the provenance labels.

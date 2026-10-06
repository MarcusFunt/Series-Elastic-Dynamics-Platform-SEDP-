# Browser workspace and visualizations

The FastAPI browser workspace is the project-wide entry point. It presents the live Python simulation, model and controller evidence, estimator/timing design, learning studies, firmware and hardware architecture, haptics, system identification, and a searchable project source map.

Start it from `software/python`:

```bash
python active_vibration_rig_web.py
```

Open `http://127.0.0.1:8765`.

If port 8765 is already in use, start the same app on another local port:

```bash
python active_vibration_rig_web.py --port 8875
```

Then open `http://127.0.0.1:8875`.

## Project views

- **Overview** — project maturity, coverage, and a summary of the random-goal-and-hold confirmation.
- **Rig simulator** — embeds the existing interactive nonlinear simulator. It retains controller/trajectory controls, live mechanism rendering, plots, CSV export, and the training tab.
- **Digital twin** — reduced and nonlinear model descriptions, OpenModelica coverage, and the committed spring-physics comparison.
- **Control lab** — controller metrics and plots from B1, MPC, and paired goal-hold evidence. Comparisons stay within the selected study because their scenario sets differ.
- **Sensors & timing** — measurement/estimator flow, planned firmware rates, and sensor timing from saved simulation configurations.
- **Learning & experiments** — PPO training curves, per-seed random-goal confirmation, target sequences, hold checks, promotion gates, three playable training-stage videos, and a link to live v4 training/evaluation controls.
- **Hardware & firmware** — proposed mechanical/electrical arrangement, RP2350 core partition, PIO path, and safety design.
- **Haptics & system ID** — force-feedback concepts and the planned physical calibration workflow.
- **Project map** — filterable catalog of subsystem sources and saved evidence.

## Provenance and maturity labels

- **Live simulation** values stream from the Python simulator; they are not measurements from the physical machine.
- **Recorded evidence** is loaded from explicitly listed JSON/Markdown files in `software/python/ui/project_catalog.json`.
- **Source/model view** describes documented or implemented architecture. Rates marked as firmware targets are proposed rates, not measured rates.
- **Planned / no telemetry** means the repository has design information but no validated physical data for that item.

The catalog/evidence API reads only declared source and evidence files, and the video route serves only explicitly listed files. Machine-local checkpoint paths are filtered from the evidence views. Checkpoints and generated videos remain on the artifact paths described in [ARTIFACTS.md](ARTIFACTS.md).

## Random goal-and-hold evidence

The saved confirmation study evaluates eight paired seeds against LQR and the repository MPC. Goal targets stay within ±48 mm on the modeled ±65 mm half-travel rail, maintaining a 15 mm edge exclusion band and an additional 2 mm target buffer. The environment requests at least a 2.5 second dwell; a successful hold requires two continuous seconds within the 5 mm target tolerance.

The chart label **integrated resonator energy** means the integral of modeled resonator kinetic-plus-spring energy over the episode. It is a vibration metric, not electrical consumption or total motor energy. In this confirmation, PPO passes the 5% energy gate versus LQR. Repository MPC has lower mean integrated resonator energy and position RMSE than PPO, so the promotion is scoped to LQR.

## Local training controls

Open Learning & experiments → **Start or evaluate a run**, or visit `/live?view=training`. The existing local v4 workflow provides PPO training, MPC teacher collection/distillation, checkpoint evaluation, standalone MPC reference evaluation, run cancellation, artifact downloads, and loading of a completed policy into the simulator.

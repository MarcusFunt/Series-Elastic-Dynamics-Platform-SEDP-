# Browser visualization

The polished front end separates **physics** from **presentation**:

- `active_vibration_rig_2d.py` — nonlinear mechanics, RK4 integration, sensors, trajectories and classical control.
- `active_vibration_rig_web.py` — FastAPI/WebSocket server, SVG mechanism renderer, Plotly live traces and motion controls.

The browser uses SVG for the parametric mechanism and Plotly.js for scientific traces. The current mechanism renderer includes the corrected two diagonal spring elements between the moving shaft and carriage.

Run:

```bash
cd software/python
python active_vibration_rig_web.py
```

For trained PPO playback:

```bash
python active_vibration_rig_web.py \
  --ppo-model ../../models/ppo_v2/ppo_v2_reversal2.pt \
  --controller ppo \
  --trajectory aggressive
```

Open `http://127.0.0.1:8765`.

Controls include play/pause, reset, resonator kick, CSV export, live plant-parameter sliders, manual rail targeting, servo/LQR/PPO/motor-position modes, and live angle/position/actuation traces.

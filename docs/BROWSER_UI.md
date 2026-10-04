# Browser visualization

The browser UI is a presentation layer over the same nonlinear plant in `active_vibration_rig_2d.py`.

- SVG renders the parametric mechanism, including the corrected two diagonal springs.
- Plotly.js renders live scientific traces.
- WebSocket streams simulation state.
- FastAPI handles commands, parameter updates and CSV export.

Run:

```bash
cd software/python
python active_vibration_rig_web.py --controller lqr --trajectory aggressive
```

PPO v3:

```bash
python active_vibration_rig_web.py \
  --ppo-model ../../models/ppo_v3/policy_accepted.pt \
  --controller ppo --trajectory aggressive
```

Open `http://127.0.0.1:8765`.

Current controller modes:

- Legacy servo
- Safe servo
- Energy damping
- **Constrained LQR**
- **PPO v3 residual**
- Motor-position mode

Current trajectories include hold, step, sine, chirp, minimum-jerk aggressive reversals, historical square-reversal stress, and manual targeting.

The PPO command path is explicitly accepted by the WebSocket command handler; an earlier UI bug that displayed PPO without accepting its controller command has been fixed.

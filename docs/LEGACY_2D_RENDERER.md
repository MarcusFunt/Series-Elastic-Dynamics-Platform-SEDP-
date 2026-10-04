# Legacy Matplotlib 2D renderer

The original code-first renderer combines the nonlinear plant with a simple Matplotlib mechanism view. It is retained in `active_vibration_rig_2d.py` because it is useful for headless debugging and parameter sweeps even though the browser visualization is now the preferred UI.

Generalized coordinates:

\[
q=[\phi_m,x,\theta]^T
\]

Belt extension:

\[
\delta_b=r_p\phi_m-x
\]

Belt force:

\[
F_b=k_b\delta_b+c_b\dot\delta_b+k_{b3}\delta_b^3
\]

The model also includes coupled carriage/resonator dynamics, motor inertia/friction, a speed-dependent torque envelope, first-order torque lag, rail friction/end stops, synthetic AS5600 measurements, and a lever IMU model.

Headless example:

```bash
python active_vibration_rig_2d.py \
  --headless 5 \
  --controller lqr \
  --trajectory aggressive \
  --csv run.csv \
  --frame final.png
```

The browser renderer supersedes the Matplotlib visualization for interactive use, but both use the same nonlinear plant.

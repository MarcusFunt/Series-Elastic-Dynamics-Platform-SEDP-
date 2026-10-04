from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "python"))

from active_vibration_rig_2d import PlantParams, ControllerParams, MotionParams, Simulator


def run(mode: str):
    sim = Simulator(PlantParams(), ControllerParams(), MotionParams(), dt=0.0005)
    sim.controller.mode = mode
    sim.trajectory.mode = "aggressive"
    sim.step(4000)
    assert all(abs(float(v)) < 1e6 for v in sim.y), sim.y
    assert len(sim.history) > 10
    peak_rail=max(abs(float(r["x"])) for r in sim.history)/sim.p.rail_half_travel
    return sim,peak_rail


if __name__ == "__main__":
    base,_=run("safe_servo")
    active,rail=run("lqr")
    assert active.controller.K_reduced is not None
    assert rail < 0.80, rail
    print("OK: SEDP safe-servo + constrained-LQR smoke test")

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "python"))

from active_vibration_rig_2d import PlantParams, ControllerParams, MotionParams, Simulator


def run(mode: str) -> None:
    sim = Simulator(PlantParams(), ControllerParams(), MotionParams(), dt=0.0005)
    sim.controller.mode = mode
    sim.trajectory.mode = "aggressive"
    sim.step(4000)
    assert all(map(lambda v: abs(float(v)) < 1e6, sim.y)), sim.y
    assert len(sim.history) > 10


if __name__ == "__main__":
    run("servo")
    run("lqr")
    print("OK: SEDP nonlinear plant servo + LQR smoke test")

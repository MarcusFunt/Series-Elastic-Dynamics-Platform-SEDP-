#!/usr/bin/env python3
"""Immutable controller benchmark + acceptance gates for SEDP.

Every controller sees the same plant, reference, motor envelope, rail limits,
integration step, and initial condition. Candidates must improve vibration
without solving the problem by abandoning tracking or consuming the rail.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from active_vibration_rig_2d import ControllerParams, MotionParams, PlantParams, Simulator


@dataclass(frozen=True)
class Scenario:
    name: str
    trajectory: str
    duration: float
    position_rmse_cap_mm: float
    rail_cap: float = 0.78
    saturation_cap: float = 0.08
    angle_rms_ratio_cap: float = 0.78
    angle_peak_ratio_cap: float = 0.90
    initial_theta_deg: float = 0.0


BENCHMARK_VERSION = "SEDP-B1"
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("step", "step", 5.0, position_rmse_cap_mm=8.0,
             angle_rms_ratio_cap=0.78, angle_peak_ratio_cap=0.85),
    Scenario("aggressive", "aggressive", 6.0, position_rmse_cap_mm=6.0,
             angle_rms_ratio_cap=0.78, angle_peak_ratio_cap=0.90),
    Scenario("square_stress", "aggressive_square", 6.0, position_rmse_cap_mm=32.0,
             angle_rms_ratio_cap=0.75, angle_peak_ratio_cap=0.85),
)


@dataclass
class Metrics:
    controller: str
    scenario: str
    peak_angle_deg: float
    angle_rms_deg: float
    residual_angle_rms_deg: float
    position_rmse_mm: float
    peak_rail_fraction: float
    saturation_fraction: float
    stop_contact_fraction: float
    torque_rms_nm: float
    resonator_energy_integral_mJs: float
    passed_hard_constraints: bool = False
    passed_vibration_gate: bool = False
    accepted: bool = False
    failure_reason: str = ""


def run_controller(controller: str, scenario: Scenario, *, ppo_policy=None,
                   plant: Optional[PlantParams] = None,
                   cp: Optional[ControllerParams] = None,
                   mp: Optional[MotionParams] = None,
                   dt: float = 0.0005) -> Simulator:
    sim = Simulator(plant or PlantParams(), cp or ControllerParams(), mp or MotionParams(), dt=dt)
    sim.controller.mode = controller
    if controller == "ppo":
        if ppo_policy is None:
            raise ValueError("controller='ppo' requires ppo_policy")
        sim.controller.set_rl_policy(ppo_policy)
    sim.trajectory.mode = scenario.trajectory
    if scenario.initial_theta_deg:
        sim.y[4] = math.radians(scenario.initial_theta_deg)
    sim.step(int(round(scenario.duration / dt)))
    return sim


def measure(sim: Simulator, scenario: Scenario, controller: str) -> Metrics:
    h = sim.history
    if not h:
        raise RuntimeError("simulation produced no logged history")
    t = np.asarray([r["t"] for r in h], dtype=float)
    theta = np.asarray([r["theta"] for r in h], dtype=float)
    w = np.asarray([r["theta_dot"] for r in h], dtype=float)
    x = np.asarray([r["x"] for r in h], dtype=float)
    wm = np.asarray([r["omega_m"] for r in h], dtype=float)
    tau = np.asarray([r["tau_cmd"] for r in h], dtype=float)
    xr = np.asarray([sim.trajectory.sample(float(tt))[0] for tt in t], dtype=float)
    th_deg = np.degrees(theta)

    residual_mask = t >= (sim.mp.step_time + 0.35) if scenario.trajectory == "step" else np.ones_like(t, dtype=bool)
    p = sim.p
    k_eff = max(abs(p.effective_small_angle_stiffness), 0.02)
    E = 0.5*p.resonator_inertia_pivot*w*w + 0.5*k_eff*theta*theta
    limits = np.asarray([sim.plant.motor_torque_limit(float(v)) for v in wm])
    saturation = np.abs(tau) >= (0.995*limits)
    rail = np.abs(x) / p.rail_half_travel
    stop = rail >= 1.0

    return Metrics(
        controller=controller,
        scenario=scenario.name,
        peak_angle_deg=float(np.max(np.abs(th_deg))),
        angle_rms_deg=float(np.sqrt(np.mean(th_deg*th_deg))),
        residual_angle_rms_deg=float(np.sqrt(np.mean(th_deg[residual_mask]**2))),
        position_rmse_mm=float(1000.0*np.sqrt(np.mean((x-xr)**2))),
        peak_rail_fraction=float(np.max(rail)),
        saturation_fraction=float(np.mean(saturation)),
        stop_contact_fraction=float(np.mean(stop)),
        torque_rms_nm=float(np.sqrt(np.mean(tau*tau))),
        resonator_energy_integral_mJs=float(1000.0*np.trapezoid(E, t)),
    )


def apply_gates(candidate: Metrics, reference: Metrics, scenario: Scenario) -> Metrics:
    reasons: List[str] = []
    hard = True
    if candidate.peak_rail_fraction > scenario.rail_cap:
        hard = False; reasons.append(f"rail {candidate.peak_rail_fraction:.3f}>{scenario.rail_cap:.3f}")
    if candidate.stop_contact_fraction > 0.0:
        hard = False; reasons.append("physical/software stop contact")
    if candidate.saturation_fraction > scenario.saturation_cap:
        hard = False; reasons.append(f"saturation {candidate.saturation_fraction:.3f}>{scenario.saturation_cap:.3f}")
    if candidate.position_rmse_mm > scenario.position_rmse_cap_mm:
        hard = False; reasons.append(f"tracking {candidate.position_rmse_mm:.2f}mm>{scenario.position_rmse_cap_mm:.2f}mm")

    vib = True
    rms_limit = reference.angle_rms_deg * scenario.angle_rms_ratio_cap
    peak_limit = reference.peak_angle_deg * scenario.angle_peak_ratio_cap
    if candidate.angle_rms_deg > rms_limit:
        vib = False; reasons.append(f"angle RMS {candidate.angle_rms_deg:.2f}>{rms_limit:.2f}deg")
    if candidate.peak_angle_deg > peak_limit:
        vib = False; reasons.append(f"peak angle {candidate.peak_angle_deg:.2f}>{peak_limit:.2f}deg")

    candidate.passed_hard_constraints = hard
    candidate.passed_vibration_gate = vib
    candidate.accepted = hard and vib
    candidate.failure_reason = "; ".join(reasons)
    return candidate


def benchmark(controllers: Iterable[str], *, ppo_policy=None, dt: float = 0.0005) -> list[Metrics]:
    controllers = list(controllers)
    rows: list[Metrics] = []
    for sc in SCENARIOS:
        ref_sim = run_controller("safe_servo", sc, dt=dt)
        ref = measure(ref_sim, sc, "safe_servo")
        ref.passed_hard_constraints = True
        ref.passed_vibration_gate = True
        ref.accepted = True
        rows.append(ref)
        for ctrl in controllers:
            if ctrl == "safe_servo":
                continue
            sim = run_controller(ctrl, sc, ppo_policy=ppo_policy, dt=dt)
            m = measure(sim, sc, ctrl)
            rows.append(apply_gates(m, ref, sc))
    return rows


def summarize(rows: list[Metrics]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    ctrls = sorted({r.controller for r in rows if r.controller != "safe_servo"})
    for c in ctrls:
        rr = [r for r in rows if r.controller == c]
        out[c] = bool(rr) and all(r.accepted for r in rr)
    return out


def print_table(rows: list[Metrics]) -> None:
    print(f"Benchmark {BENCHMARK_VERSION}")
    print("scenario        controller     peak°   rms°   track-mm  rail   sat%   ACCEPT")
    print("-"*84)
    for r in rows:
        print(f"{r.scenario:15s} {r.controller:13s} {r.peak_angle_deg:6.2f} "
              f"{r.angle_rms_deg:6.2f} {r.position_rmse_mm:9.2f} "
              f"{r.peak_rail_fraction:5.3f} {100*r.saturation_fraction:6.2f}  "
              f"{'YES' if r.accepted else 'NO ':3s} {r.failure_reason}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["energy", "lqr"])
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--fail-on-reject", action="store_true")
    args = ap.parse_args()
    rows = benchmark(args.controllers)
    print_table(rows)
    summary = summarize(rows)
    print("\nsummary:", summary)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader(); w.writerows(asdict(r) for r in rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"benchmark_version": BENCHMARK_VERSION,
                                         "summary": summary,
                                         "rows": [asdict(r) for r in rows]}, indent=2))
    if args.fail_on_reject and not all(summary.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Active vibration rig — polished browser visualization
=====================================================

A modern browser UI for ``active_vibration_rig_2d.py``.

Visualization stack
-------------------
* SVG: crisp parametric mechanism rendering and animation.
* Plotly.js: live scientific traces.
* FastAPI + WebSocket: low-latency Python simulation -> browser state stream.

The physical model, numerical integrator, sensor model, motion generators and
controllers remain in active_vibration_rig_2d.py so the visualization layer is
cleanly separated from the dynamics source-of-truth.

Run:
    python active_vibration_rig_web.py

Then open:
    http://127.0.0.1:8765

Dependencies:
    numpy, fastapi, uvicorn, plotly

The UI is self-contained/offline: Plotly.js is embedded from the installed
Python plotly package rather than loaded from a CDN.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from plotly.offline import get_plotlyjs
import uvicorn

# The dynamics source of truth stays in the original script.
from active_vibration_rig_2d import (
    PlantParams,
    ControllerParams,
    MotionParams,
    Simulator,
    clamp,
)


class Runtime:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.lock = threading.RLock()
        self.playing = True
        self.speed = 1.0
        self.running = True
        self.last_wall = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, name="rig-sim", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        # Advance in real time, but never allow the physics catch-up loop to explode
        # after the process is paused by a debugger/window drag.
        while self.running:
            now = time.perf_counter()
            elapsed = now - self.last_wall
            self.last_wall = now
            with self.lock:
                if self.playing:
                    target_dt = min(elapsed * self.speed, 0.030)
                    n = max(0, int(target_dt / self.sim.dt))
                    if n:
                        self.sim.step(n)
            time.sleep(0.001)

    def snapshot(self, include_history: bool = False) -> Dict[str, Any]:
        with self.lock:
            s = self.sim
            p = s.p
            y = s.y.copy()
            ref = s.trajectory.sample(s.t)
            u = s.controller.command(y, ref)
            sensor = s.plant.sensor_values(y, u, s.rng)
            Fb, delta, delta_dot = s.plant.belt_force(y)
            tau_lim = s.plant.motor_torque_limit(float(y[1]))
            out: Dict[str, Any] = {
                "t": float(s.t),
                "playing": self.playing,
                "speed": self.speed,
                "controller": s.controller.mode,
                "trajectory": s.trajectory.mode,
                "state": {
                    "phi_m": float(y[0]),
                    "omega_m": float(y[1]),
                    "x": float(y[2]),
                    "x_dot": float(y[3]),
                    "theta": float(y[4]),
                    "theta_dot": float(y[5]),
                    "tau_act": float(y[6]),
                },
                "ref": {"x": float(ref[0]), "v": float(ref[1]), "a": float(ref[2])},
                "derived": {
                    "tau_cmd": float(u),
                    "tau_limit": float(tau_lim),
                    "belt_force": float(Fb),
                    "belt_extension": float(delta),
                    "belt_extension_rate": float(delta_dot),
                    "k_eff": float(p.effective_small_angle_stiffness),
                    "energy_proxy": float(
                        0.5 * p.resonator_inertia_pivot * y[5] ** 2
                        + 0.5 * max(p.effective_small_angle_stiffness, 0.0) * y[4] ** 2
                    ),
                    "gyro": float(sensor["gyro"]),
                    "accel_tan": float(sensor["accel_tan"]),
                    "accel_long": float(sensor["accel_long"]),
                    "x_ddot": float(sensor["x_ddot"]),
                    "theta_ddot": float(sensor["theta_ddot"]),
                },
                "params": {
                    "rail_half_travel": float(p.rail_half_travel),
                    "lever_com_distance": float(p.lever_com_distance),
                    "imu_radius": float(p.imu_radius),
                    "k_theta": float(p.k_theta),
                    "c_theta": float(p.c_theta),
                    "resonator_mass": float(p.resonator_mass),
                    "belt_stiffness": float(p.belt_stiffness),
                    "amplitude": float(s.mp.amplitude),
                },
            }
            if include_history:
                rows = s.history
                if rows:
                    t0 = max(0.0, s.t - 10.0)
                    rows = [r for r in rows if r["t"] >= t0]
                    # Bound browser work to ~600 samples.
                    stride = max(1, math.ceil(len(rows) / 600))
                    rows = rows[::stride]
                    out["history"] = {
                        "t": [float(r["t"]) for r in rows],
                        "theta": [math.degrees(float(r["theta"])) for r in rows],
                        "x": [1000.0 * float(r["x"]) for r in rows],
                        "x_ref": [1000.0 * float(s.trajectory.sample(float(r["t"]))[0]) for r in rows],
                        "tau": [float(r["tau_act"]) for r in rows],
                        "force": [float(r["belt_force"]) for r in rows],
                    }
                else:
                    out["history"] = {"t": [], "theta": [], "x": [], "x_ref": [], "tau": [], "force": []}
            return out

    def apply(self, msg: Dict[str, Any]) -> None:
        typ = msg.get("type")
        with self.lock:
            s = self.sim
            if typ == "play":
                self.playing = bool(msg.get("value", True))
                self.last_wall = time.perf_counter()
            elif typ == "toggle_play":
                self.playing = not self.playing
                self.last_wall = time.perf_counter()
            elif typ == "reset":
                s.reset()
                self.last_wall = time.perf_counter()
            elif typ == "kick":
                s.kick(float(msg.get("torque", 0.025)), float(msg.get("duration", 0.06)))
            elif typ == "controller":
                v = str(msg.get("value", "servo"))
                if v in {"servo", "safe_servo", "energy", "lqr", "lqr_legacy", "motor_position", "ppo"}:
                    s.controller.mode = v
            elif typ == "trajectory":
                v = str(msg.get("value", "step"))
                if v in {"hold", "step", "sine", "chirp", "aggressive", "manual"}:
                    s.trajectory.mode = v
            elif typ == "manual_target":
                L = s.p.rail_half_travel * 0.92
                s.trajectory.manual_target = clamp(float(msg.get("value", 0.0)), -L, L)
                s.trajectory.mode = "manual"
            elif typ == "speed":
                self.speed = clamp(float(msg.get("value", 1.0)), 0.05, 4.0)
            elif typ == "param":
                key = str(msg.get("key", ""))
                val = float(msg.get("value", 0.0))
                if key == "k_theta":
                    s.p.k_theta = clamp(val, 0.05, 2.0)
                elif key == "c_theta":
                    s.p.c_theta = clamp(val, 0.0, 0.08)
                elif key == "resonator_mass":
                    s.p.resonator_mass = clamp(val, 0.02, 0.50)
                elif key == "belt_stiffness":
                    s.p.belt_stiffness = clamp(val, 200.0, 30000.0)
                elif key == "amplitude":
                    s.mp.amplitude = clamp(val, 0.001, s.p.rail_half_travel * 0.90)
                else:
                    return
                s.controller.recompute_lqr()

    def csv_bytes(self) -> bytes:
        with self.lock:
            if not self.sim.history:
                return b""
            out = io.StringIO()
            keys = list(self.sim.history[0].keys())
            w = csv.DictWriter(out, fieldnames=keys)
            w.writeheader()
            w.writerows(self.sim.history)
            return out.getvalue().encode("utf-8")


def build_html() -> str:
    plotly_js = get_plotlyjs()
    # No external URLs: everything required to render is local in this document.
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Active Vibration Rig</title>
<script>{plotly_js}</script>
<style>
:root {{
  color-scheme: dark;
  --bg:#0b0f14; --panel:#121820; --panel2:#171f29; --line:#263241;
  --text:#edf3f8; --muted:#8fa1b3; --cyan:#59d8d1; --blue:#61aefc;
  --violet:#a485ff; --orange:#ffb45f; --red:#ff6f6f; --green:#70d7a2;
}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 30% -10%,#162638 0,#0b0f14 42%);color:var(--text);font:14px/1.4 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
button,input,select{{font:inherit}} .app{{max-width:1600px;margin:auto;padding:18px}}
.topbar{{display:flex;justify-content:space-between;gap:14px;align-items:center;margin-bottom:14px;flex-wrap:wrap}}
.brand h1{{font-size:20px;margin:0 0 2px;font-weight:700;letter-spacing:-.02em}} .brand div{{color:var(--muted);font-size:12px}}
.status{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .pill{{background:#111922;border:1px solid var(--line);border-radius:999px;padding:7px 10px;color:var(--muted)}} .live-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;box-shadow:0 0 12px #70d7a280}}
.grid{{display:grid;grid-template-columns:minmax(660px,1.45fr) minmax(400px,.8fr);gap:14px}} .panel{{background:linear-gradient(180deg,#141b24,#10161e);border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 40px #0005;overflow:hidden}}
.panel-title{{padding:13px 15px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px}} .panel-title b{{font-size:13px}} .panel-title span{{font-size:11px;color:var(--muted)}}
.mechanism{{padding:8px 12px 12px}} #rigSvg{{display:block;width:100%;height:auto;min-height:390px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px 4px 2px}} .metric{{background:#0d131a;border:1px solid var(--line);border-radius:11px;padding:8px 10px}} .metric label{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}} .metric strong{{display:block;margin-top:3px;font:600 16px ui-monospace,SFMono-Regular,Menlo,monospace}}
.controls{{padding:12px;display:grid;gap:12px}} .row{{display:flex;gap:8px;flex-wrap:wrap}} .btn,.seg button{{border:1px solid #2a3949;background:#121b25;color:var(--text);border-radius:9px;padding:8px 10px;cursor:pointer;transition:.12s}} .btn:hover,.seg button:hover{{background:#192635}} .btn.primary{{border-color:#376fa7;background:#173654}} .btn.warn{{border-color:#7f5330;background:#3b2818}} .seg{{display:flex;gap:4px;background:#0d131a;border:1px solid var(--line);padding:4px;border-radius:11px;flex-wrap:wrap}} .seg button{{padding:7px 9px;border-color:transparent;background:transparent;color:var(--muted)}} .seg button.active{{background:#223447;color:var(--text);border-color:#35516d}}
.group-title{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}} .slider-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}} .slider label{{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px}} input[type=range]{{width:100%;accent-color:var(--blue)}}
.chart-stack{{display:grid;grid-template-columns:1fr;gap:10px;padding:10px}} .chart{{height:205px;background:#0d131a;border:1px solid var(--line);border-radius:11px;overflow:hidden}}
.footer{{color:var(--muted);font-size:11px;margin-top:12px;text-align:center}}
svg text{{font-family:Inter,system-ui,sans-serif}} .rail-shadow{{filter:drop-shadow(0 7px 9px #0008)}} .carriage{{filter:drop-shadow(0 9px 10px #0008)}} .mass{{filter:drop-shadow(0 8px 9px #0007)}}
@media(max-width:1050px){{.grid{{grid-template-columns:1fr}}.slider-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:650px){{.app{{padding:8px}}.grid{{display:block}}.panel{{margin-bottom:10px}}.metrics{{grid-template-columns:1fr 1fr}}.slider-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand"><h1>Active vibration / series-elastic rig</h1><div>Nonlinear digital plant • live vector renderer • closed-loop motion playground</div></div>
    <div class="status"><span class="pill"><span class="live-dot" id="dot"></span><span id="conn">connecting</span></span><span class="pill" id="clock">t = 0.000 s</span><span class="pill" id="fps">0 Hz stream</span></div>
  </div>

  <div class="grid">
    <div>
      <div class="panel">
        <div class="panel-title"><b>Mechanical scene</b><span>click the rail to set a manual target</span></div>
        <div class="mechanism">
          <svg id="rigSvg" viewBox="0 0 1100 570" aria-label="Animated active vibration rig">
            <defs>
              <linearGradient id="railG" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#536678"/><stop offset="1" stop-color="#25313d"/></linearGradient>
              <linearGradient id="cartG" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#355d81"/><stop offset="1" stop-color="#1d3348"/></linearGradient>
              <linearGradient id="massG" x1="0" x2="1"><stop offset="0" stop-color="#725cc2"/><stop offset="1" stop-color="#3d306d"/></linearGradient>
              <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            <rect x="0" y="0" width="1100" height="570" rx="14" fill="#0d131a"/>
            <g opacity=".18" stroke="#385069" stroke-width="1">
              <path d="M80 70V500M180 70V500M280 70V500M380 70V500M480 70V500M580 70V500M680 70V500M780 70V500M880 70V500M980 70V500"/>
              <path d="M65 100H1035M65 200H1035M65 300H1035M65 400H1035M65 500H1035"/>
            </g>
            <text x="70" y="42" fill="#8fa1b3" font-size="13">− travel</text><text x="986" y="42" fill="#8fa1b3" font-size="13">+ travel</text>
            <g id="target"><line x1="550" x2="550" y1="102" y2="470" stroke="#61aefc" stroke-width="2" stroke-dasharray="6 7" opacity=".55"/><path d="M542 104 L558 104 L550 118 Z" fill="#61aefc" opacity=".8"/></g>
            <g id="limits" stroke="#ff6f6f" opacity=".5"><line x1="145" x2="145" y1="350" y2="473" stroke-dasharray="5 7"/><line x1="955" x2="955" y1="350" y2="473" stroke-dasharray="5 7"/></g>
            <g class="rail-shadow"><rect id="railHit" x="130" y="430" width="840" height="40" rx="11" fill="url(#railG)"/><rect x="130" y="444" width="840" height="7" rx="3.5" fill="#8192a3" opacity=".3"/></g>
            <g id="motorGroup">
              <rect x="62" y="337" width="92" height="88" rx="13" fill="#202a35" stroke="#526273" stroke-width="2"/><circle cx="108" cy="381" r="29" fill="#10161d" stroke="#ffb45f" stroke-width="4"/><g id="motorSpokes" stroke="#ffb45f" stroke-width="3"><line x1="108" y1="357" x2="108" y2="405"/><line x1="84" y1="381" x2="132" y2="381"/></g><text x="108" y="472" text-anchor="middle" fill="#8fa1b3" font-size="12">NEMA 17 + AS5600</text>
            </g>
            <circle cx="175" cy="415" r="19" fill="#0f151c" stroke="#ffb45f" stroke-width="3"/><circle cx="925" cy="415" r="19" fill="#0f151c" stroke="#758596" stroke-width="3"/>
            <path d="M175 396 H925 M175 434 H925" stroke="#ffb45f" stroke-width="3" opacity=".82"/>
            <g id="carriage" class="carriage" transform="translate(550,430)">
              <rect x="-58" y="-32" width="116" height="64" rx="11" fill="url(#cartG)" stroke="#61aefc" stroke-width="2"/>
              <rect x="-36" y="26" width="72" height="17" rx="5" fill="#182634" stroke="#3e678a"/>
              <circle cx="0" cy="-40" r="10" fill="#111821" stroke="#a485ff" stroke-width="3"/>
              <path id="springLeft" d="M0 0" fill="none" stroke="#ffb45f" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
              <path id="springRight" d="M0 0" fill="none" stroke="#ffb45f" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
              <g id="lever" transform="translate(0,-40)">
                <line x1="0" y1="0" x2="0" y2="-208" stroke="#a485ff" stroke-width="9" stroke-linecap="round"/>
                <line x1="0" y1="-30" x2="0" y2="-178" stroke="#c7b9ff" stroke-width="2" opacity=".65"/>
                <rect id="imu" x="-22" y="-149" width="44" height="27" rx="5" fill="#173c3d" stroke="#59d8d1" stroke-width="2"/><text x="0" y="-131" text-anchor="middle" fill="#baf7f3" font-size="10">IMU</text>
                <circle class="mass" cx="0" cy="-210" r="31" fill="url(#massG)" stroke="#a485ff" stroke-width="3"/><circle cx="0" cy="-210" r="7" fill="#d6ceff" opacity=".8"/>
              </g>
              <text x="0" y="9" text-anchor="middle" fill="#dbe8f3" font-size="12">carriage</text>
            </g>
            <g id="vectors" stroke-linecap="round">
              <line id="velVec" x1="0" y1="0" x2="0" y2="0" stroke="#70d7a2" stroke-width="3"/><line id="accVec" x1="0" y1="0" x2="0" y2="0" stroke="#ff6f6f" stroke-width="3"/>
            </g>
            <text id="thetaLabel" x="760" y="98" fill="#c9bcff" font-size="16">θ = 0.0°</text>
            <text id="forceLabel" x="760" y="124" fill="#ffca8c" font-size="14">belt F = 0.0 N</text>
            <text id="energyLabel" x="760" y="149" fill="#8fa1b3" font-size="13">E* = 0.0000 J</text>
            <g transform="translate(72,520)"><rect width="956" height="1" fill="#314050"/><text x="0" y="26" fill="#8fa1b3" font-size="11">Green = carriage velocity • red = carriage acceleration • violet = compliant resonator</text></g>
          </svg>
          <div class="metrics">
            <div class="metric"><label>Carriage x</label><strong id="mx">0.00 mm</strong></div>
            <div class="metric"><label>Resonator θ</label><strong id="mth">0.00°</strong></div>
            <div class="metric"><label>Motor torque</label><strong id="mtau">0.000 Nm</strong></div>
            <div class="metric"><label>Belt force</label><strong id="mf">0.00 N</strong></div>
          </div>
        </div>
      </div>

      <div class="panel" style="margin-top:14px">
        <div class="panel-title"><b>Live dynamics</b><span>rolling 10 s window</span></div>
        <div class="chart-stack"><div id="thetaChart" class="chart"></div><div id="xChart" class="chart"></div><div id="actChart" class="chart"></div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title"><b>Motion laboratory</b><span>all changes apply live</span></div>
      <div class="controls">
        <div><div class="group-title">Simulation</div><div class="row"><button class="btn primary" id="playBtn">Pause</button><button class="btn" id="resetBtn">Reset</button><button class="btn warn" id="kickBtn">Kick resonator</button><a class="btn" href="/api/export.csv" download="rig_log.csv" style="text-decoration:none">Export CSV</a></div></div>
        <div><div class="group-title">Controller</div><div class="seg" id="controllerSeg"><button data-v="servo">Legacy servo</button><button data-v="safe_servo" class="active">Safe servo</button><button data-v="energy">Energy damping</button><button data-v="lqr">Constrained LQR</button><button data-v="ppo">PPO v3</button><button data-v="motor_position">Motor position</button></div></div>
        <div><div class="group-title">Motion</div><div class="seg" id="trajectorySeg"><button data-v="hold">Hold</button><button data-v="step" class="active">Step</button><button data-v="sine">Sine</button><button data-v="chirp">Chirp</button><button data-v="aggressive">Aggressive</button><button data-v="aggressive_square">Square stress</button><button data-v="manual">Manual</button></div></div>
        <div><div class="group-title">Plant parameters</div><div class="slider-grid">
          <div class="slider"><label><span>Spring kθ</span><span id="vk">0.42 Nm/rad</span></label><input id="sk" type="range" min="0.15" max="1.2" step="0.005" value="0.42"></div>
          <div class="slider"><label><span>Damping cθ</span><span id="vc">0.006 Nms/rad</span></label><input id="sc" type="range" min="0" max="0.035" step="0.00025" value="0.006"></div>
          <div class="slider"><label><span>Resonator mass</span><span id="vm">0.120 kg</span></label><input id="sm" type="range" min="0.04" max="0.30" step="0.002" value="0.12"></div>
          <div class="slider"><label><span>Belt stiffness</span><span id="vb">6500 N/m</span></label><input id="sb" type="range" min="1000" max="16000" step="100" value="6500"></div>
          <div class="slider"><label><span>Motion amplitude</span><span id="va">35.0 mm</span></label><input id="sa" type="range" min="0.005" max="0.055" step="0.001" value="0.035"></div>
          <div class="slider"><label><span>Simulation speed</span><span id="vs">1.00×</span></label><input id="ss" type="range" min="0.1" max="2" step="0.05" value="1"></div>
        </div></div>
        <div><div class="group-title">State details</div><div class="metrics" style="grid-template-columns:1fr 1fr;padding:0">
          <div class="metric"><label>θ̇</label><strong id="mw">0.0 °/s</strong></div><div class="metric"><label>ẋ</label><strong id="mv">0.000 m/s</strong></div>
          <div class="metric"><label>ẍ</label><strong id="ma">0.00 m/s²</strong></div><div class="metric"><label>gyro</label><strong id="mgyro">0.0 °/s</strong></div>
          <div class="metric"><label>k effective</label><strong id="mke">0.000 Nm/rad</strong></div><div class="metric"><label>belt stretch</label><strong id="mbelt">0.000 mm</strong></div>
        </div></div>
        <div class="metric"><label>Keyboard</label><div style="margin-top:5px;color:var(--muted);font-size:12px">Space play/pause • R reset • K kick • ←/→ move manual target</div></div>
      </div>
    </div>
  </div>
  <div class="footer">Physics runs in Python at the configured integration timestep; the browser only renders and commands the plant.</div>
</div>
<script>
const $=id=>document.getElementById(id); let ws=null,lastRx=performance.now(),rxCount=0,rxWindow=performance.now(),latest=null;
const layoutBase={{paper_bgcolor:'#0d131a',plot_bgcolor:'#0d131a',font:{{color:'#b9c7d4',size:11}},margin:{{l:48,r:14,t:14,b:36}},xaxis:{{gridcolor:'#22303e',zerolinecolor:'#314355'}},yaxis:{{gridcolor:'#22303e',zerolinecolor:'#314355'}},showlegend:true,legend:{{orientation:'h',x:0,y:1.12}},hovermode:'x unified'}};
Plotly.newPlot('thetaChart',[{{x:[],y:[],name:'θ [deg]',line:{{color:'#a485ff',width:2.2}}}}],{{...layoutBase,yaxis:{{...layoutBase.yaxis,title:'θ [deg]'}},xaxis:{{...layoutBase.xaxis,title:'time [s]'}}}},{{displayModeBar:false,responsive:true}});
Plotly.newPlot('xChart',[{{x:[],y:[],name:'x',line:{{color:'#61aefc',width:2.2}}}},{{x:[],y:[],name:'x target',line:{{color:'#8fa1b3',width:1.5,dash:'dot'}}}}],{{...layoutBase,yaxis:{{...layoutBase.yaxis,title:'x [mm]'}},xaxis:{{...layoutBase.xaxis,title:'time [s]'}}}},{{displayModeBar:false,responsive:true}});
Plotly.newPlot('actChart',[{{x:[],y:[],name:'τ motor',line:{{color:'#ffb45f',width:2}}}},{{x:[],y:[],name:'belt F',yaxis:'y2',line:{{color:'#70d7a2',width:1.8}}}}],{{...layoutBase,xaxis:{{...layoutBase.xaxis,title:'time [s]'}},yaxis:{{...layoutBase.yaxis,title:'τ [Nm]'}},yaxis2:{{title:'F [N]',overlaying:'y',side:'right',gridcolor:'rgba(0,0,0,0)'}}}},{{displayModeBar:false,responsive:true}});
function send(o){{if(ws&&ws.readyState===1)ws.send(JSON.stringify(o));}}
function connect(){{ws=new WebSocket(`${{location.protocol==='https:'?'wss':'ws'}}://${{location.host}}/ws`);ws.onopen=()=>{{$('conn').textContent='connected';$('dot').style.background='#70d7a2'}};ws.onclose=()=>{{$('conn').textContent='reconnecting';$('dot').style.background='#ff6f6f';setTimeout(connect,800)}};ws.onmessage=e=>{{latest=JSON.parse(e.data);lastRx=performance.now();rxCount++;render(latest);}};}} connect();
function setSeg(id,v){{document.querySelectorAll(`#${{id}} button`).forEach(b=>b.classList.toggle('active',b.dataset.v===v));}}
function springPath(x1,y1,x2,y2,turns=5,amp=7){{
  const dx=x2-x1, dy=y2-y1;
  const L=Math.hypot(dx,dy)||1e-9;
  const ux=dx/L, uy=dy/L;
  const px=-uy, py=ux;
  const lead=Math.min(10,0.18*L);
  const tail=Math.min(10,0.18*L);
  const pts=[[x1,y1],[x1+ux*lead,y1+uy*lead]];
  const usable=Math.max(0,L-lead-tail);
  for(let i=1;i<turns;i++){{
    const t=i/turns;
    const along=lead+t*usable;
    const sign=(i%2===1)?1:-1;
    pts.push([x1+ux*along+px*amp*sign,y1+uy*along+py*amp*sign]);
  }}
  pts.push([x2-ux*tail,y2-uy*tail],[x2,y2]);
  return 'M '+pts.map(p=>`${{p[0].toFixed(1)}} ${{p[1].toFixed(1)}}`).join(' L ');
}}
function render(s){{
  $('clock').textContent=`t = ${{s.t.toFixed(3)}} s`; $('playBtn').textContent=s.playing?'Pause':'Play'; setSeg('controllerSeg',s.controller); setSeg('trajectorySeg',s.trajectory);
  const st=s.state,d=s.derived,p=s.params; const deg=st.theta*180/Math.PI;
  $('mx').textContent=`${{(st.x*1000).toFixed(2)}} mm`; $('mth').textContent=`${{deg.toFixed(2)}}°`; $('mtau').textContent=`${{st.tau_act.toFixed(3)}} Nm`; $('mf').textContent=`${{d.belt_force.toFixed(2)}} N`;
  $('mw').textContent=`${{(st.theta_dot*180/Math.PI).toFixed(1)}} °/s`; $('mv').textContent=`${{st.x_dot.toFixed(3)}} m/s`; $('ma').textContent=`${{d.x_ddot.toFixed(2)}} m/s²`; $('mgyro').textContent=`${{(d.gyro*180/Math.PI).toFixed(1)}} °/s`; $('mke').textContent=`${{d.k_eff.toFixed(3)}} Nm/rad`; $('mbelt').textContent=`${{(d.belt_extension*1000).toFixed(3)}} mm`;
  $('thetaLabel').textContent=`θ = ${{deg.toFixed(2)}}°`; $('forceLabel').textContent=`belt F = ${{d.belt_force.toFixed(2)}} N`; $('energyLabel').textContent=`E* = ${{d.energy_proxy.toFixed(5)}} J`;
  const x0=550, scale=6200; const cx=x0+st.x*scale; const tx=x0+s.ref.x*scale;
  $('carriage').setAttribute('transform',`translate(${{cx}},430)`); $('target').setAttribute('transform',`translate(${{tx-550}},0)`); $('lever').setAttribute('transform',`translate(0,-40) rotate(${{deg}})`);
  const th=st.theta;
  const shaftAttachLen=18;
  const shaftX=Math.sin(th)*shaftAttachLen;
  const shaftY=-40-Math.cos(th)*shaftAttachLen;
  const leftAx=-40, leftAy=-6, rightAx=40, rightAy=-6;
  $('springLeft').setAttribute('d', springPath(shaftX, shaftY, leftAx, leftAy, 5, 6));
  $('springRight').setAttribute('d', springPath(shaftX, shaftY, rightAx, rightAy, 5, 6));
  $('motorSpokes').setAttribute('transform',`rotate(${{(st.phi_m*180/Math.PI)%360}} 108 381)`);
  const vlen=Math.max(-90,Math.min(90,st.x_dot*330)); const alen=Math.max(-100,Math.min(100,d.x_ddot*13));
  $('velVec').setAttribute('x1',cx);$('velVec').setAttribute('y1',488);$('velVec').setAttribute('x2',cx+vlen);$('velVec').setAttribute('y2',488);
  $('accVec').setAttribute('x1',cx);$('accVec').setAttribute('y1',506);$('accVec').setAttribute('x2',cx+alen);$('accVec').setAttribute('y2',506);
  if(s.history){{const h=s.history;Plotly.react('thetaChart',[{{x:h.t,y:h.theta,name:'θ [deg]',line:{{color:'#a485ff',width:2.2}}}}],{{...layoutBase,yaxis:{{...layoutBase.yaxis,title:'θ [deg]'}},xaxis:{{...layoutBase.xaxis,title:'time [s]'}}}},{{displayModeBar:false,responsive:true}});Plotly.react('xChart',[{{x:h.t,y:h.x,name:'x',line:{{color:'#61aefc',width:2.2}}}},{{x:h.t,y:h.x_ref,name:'x target',line:{{color:'#8fa1b3',width:1.5,dash:'dot'}}}}],{{...layoutBase,yaxis:{{...layoutBase.yaxis,title:'x [mm]'}},xaxis:{{...layoutBase.xaxis,title:'time [s]'}}}},{{displayModeBar:false,responsive:true}});Plotly.react('actChart',[{{x:h.t,y:h.tau,name:'τ motor',line:{{color:'#ffb45f',width:2}}}},{{x:h.t,y:h.force,name:'belt F',yaxis:'y2',line:{{color:'#70d7a2',width:1.8}}}}],{{...layoutBase,xaxis:{{...layoutBase.xaxis,title:'time [s]'}},yaxis:{{...layoutBase.yaxis,title:'τ [Nm]'}},yaxis2:{{title:'F [N]',overlaying:'y',side:'right',gridcolor:'rgba(0,0,0,0)'}}}},{{displayModeBar:false,responsive:true}});}}
}}
setInterval(()=>{{const now=performance.now();if(now-rxWindow>1000){{$('fps').textContent=`${{rxCount}} Hz stream`;rxCount=0;rxWindow=now;}}}},1000);
$('playBtn').onclick=()=>send({{type:'toggle_play'}});$('resetBtn').onclick=()=>send({{type:'reset'}});$('kickBtn').onclick=()=>send({{type:'kick'}});
$('controllerSeg').onclick=e=>{{if(e.target.dataset.v)send({{type:'controller',value:e.target.dataset.v}})}};$('trajectorySeg').onclick=e=>{{if(e.target.dataset.v)send({{type:'trajectory',value:e.target.dataset.v}})}};
function wireSlider(id,label,key,fmt){{const el=$(id);const out=$(label);const apply=()=>{{const v=parseFloat(el.value);out.textContent=fmt(v);send({{type:'param',key,value:v}})}};el.addEventListener('input',apply);}}
wireSlider('sk','vk','k_theta',v=>`${{v.toFixed(3)}} Nm/rad`);wireSlider('sc','vc','c_theta',v=>`${{v.toFixed(4)}} Nms/rad`);wireSlider('sm','vm','resonator_mass',v=>`${{v.toFixed(3)}} kg`);wireSlider('sb','vb','belt_stiffness',v=>`${{v.toFixed(0)}} N/m`);wireSlider('sa','va','amplitude',v=>`${{(v*1000).toFixed(1)}} mm`);
$('ss').addEventListener('input',()=>{{const v=parseFloat($('ss').value);$('vs').textContent=`${{v.toFixed(2)}}×`;send({{type:'speed',value:v}})}});
$('railHit').addEventListener('pointerdown',e=>{{const pt=$('rigSvg').createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const q=pt.matrixTransform($('rigSvg').getScreenCTM().inverse());const L=latest?.params?.rail_half_travel||0.065;const x=((q.x-550)/6200);send({{type:'manual_target',value:Math.max(-.92*L,Math.min(.92*L,x))}})}});
window.addEventListener('keydown',e=>{{if(e.target.matches('input,select,button'))return;if(e.code==='Space'){{e.preventDefault();send({{type:'toggle_play'}})}}else if(e.key==='r'||e.key==='R')send({{type:'reset'}});else if(e.key==='k'||e.key==='K')send({{type:'kick'}});else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){{const L=latest?.params?.rail_half_travel||.065;const cur=latest?.trajectory==='manual'?latest.ref.x:latest?.state?.x||0;send({{type:'manual_target',value:Math.max(-.92*L,Math.min(.92*L,cur+(e.key==='ArrowRight'?.005:-.005)))}})}}}});
</script>
</body></html>'''


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Active vibration rig")
    html = build_html()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return html

    @app.get("/api/state")
    async def state() -> Dict[str, Any]:
        return runtime.snapshot(include_history=True)

    @app.get("/api/export.csv")
    async def export() -> Response:
        return Response(
            runtime.csv_bytes(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="rig_log.csv"'},
        )

    @app.get("/api/params")
    async def params() -> Dict[str, Any]:
        with runtime.lock:
            return {
                "plant": asdict(runtime.sim.p),
                "controller": asdict(runtime.sim.cp),
                "motion": asdict(runtime.sim.mp),
            }

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        next_hist = 0.0
        try:
            while True:
                # Poll commands without blocking the 30 Hz state stream.
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0 / 30.0)
                    runtime.apply(json.loads(raw))
                except asyncio.TimeoutError:
                    pass
                now = time.perf_counter()
                include_history = now >= next_hist
                if include_history:
                    next_hist = now + 0.10
                await ws.send_json(runtime.snapshot(include_history=include_history))
        except WebSocketDisconnect:
            pass

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--params", type=Path, help="JSON file overriding PlantParams")
    ap.add_argument("--controller", choices=["servo", "safe_servo", "energy", "lqr", "lqr_legacy", "motor_position", "ppo"], default="servo")
    ap.add_argument("--ppo-model", type=Path, help="PPO v3 residual-acceleration checkpoint (.pt)")
    ap.add_argument("--ppo-v4-model", type=Path, help="Estimated-state v4 checkpoint; uses checkpoint timing")
    ap.add_argument("--trajectory", choices=["hold", "step", "sine", "chirp", "aggressive", "aggressive_square", "manual"], default="step")
    ap.add_argument("--dt", type=float, default=0.0005)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    p = PlantParams()
    if args.params:
        p = PlantParams.from_dict(json.loads(args.params.read_text()))
    if args.ppo_model and args.ppo_v4_model:
        raise SystemExit("Choose either --ppo-model or --ppo-v4-model")
    if args.ppo_v4_model:
        from evaluate_v4 import load_policy
        from rig_rl_policy_v4 import EstimatedResidualPolicyV4
        model, cfg, _ = load_policy(args.ppo_v4_model)
        sim = Simulator(p, ControllerParams(), MotionParams(), dt=cfg.physics_dt, control_dt=cfg.control_dt)
        sim.controller.set_rl_policy(EstimatedResidualPolicyV4(model,cfg,p,sim.cp,sim.trajectory))
    else:
        sim = Simulator(p, ControllerParams(), MotionParams(), dt=args.dt)
    if args.ppo_model:
        from rig_rl_policy_v3 import PPOPolicyAdapterV3
        sim.controller.set_rl_policy(PPOPolicyAdapterV3(args.ppo_model, sim.p, sim.cp, sim.dt))
    if args.controller == "ppo" and sim.controller.rl_policy is None:
        raise SystemExit("--controller ppo requires --ppo-model or --ppo-v4-model")
    sim.controller.mode = args.controller
    sim.trajectory.mode = args.trajectory
    runtime = Runtime(sim)
    app = create_app(runtime)
    print(f"Active vibration rig UI: http://{args.host}:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        runtime.running = False


if __name__ == "__main__":
    main()
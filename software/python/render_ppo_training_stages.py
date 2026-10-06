"""Render deterministic random-goal-and-hold playbacks for PPO checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from evaluate_v4 import load_policy
from rig_rl_env_v4 import RigRLEnvV4


STAGES = ((8192, "Early training"), (32768, "Mid training"), (65536, "Final training"))


def rollout(checkpoint: Path, seed: int) -> tuple[np.ndarray, object]:
    model, cfg, _ = load_policy(checkpoint)
    if cfg.reference_mode != "random_goal_hold":
        raise ValueError(f"{checkpoint} is not a random-goal-and-hold checkpoint")
    cfg.domain_randomization = False
    cfg.initial_theta_std = 0.0
    cfg.initial_theta_dot_std = 0.0
    cfg.kick_probability = 0.0
    env = RigRLEnvV4(cfg=cfg, seed=seed)
    obs, _ = env.reset()
    rows = []
    for _ in range(cfg.max_steps):
        with torch.no_grad():
            action = float(model.deterministic(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            )[0, 0])
        obs, _, terminated, truncated, _ = env.step([action])
        reference = env.reference.sample(env.t)
        rows.append((env.t, env.y[2], reference[0], env.y[4], action,
                     env.reference.change_count + 1))
        if terminated or truncated:
            break
    return np.asarray(rows, dtype=float), cfg


def render_video(checkpoint: Path, output: Path, seed: int, steps: int, stage: str) -> Path:
    data, cfg = rollout(checkpoint, seed)
    stride = max(1, round((1.0 / 30.0) / cfg.control_dt))
    data = data[::stride]
    t, x_m, target_m, theta, action, goal_index = data.T
    x_mm, target_mm = x_m * 1000.0, target_m * 1000.0
    fig = plt.figure(figsize=(12, 6.76), dpi=100, facecolor="#0b1220")
    ax = fig.add_axes([0.075, 0.40, 0.85, 0.42], facecolor="#101a2c")
    chart = fig.add_axes([0.09, 0.10, 0.82, 0.21], facecolor="#101a2c")
    for axis in (ax, chart):
        axis.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#334155")
        axis.grid(True, color="#334155", alpha=.55, linewidth=.6)

    ax.set_xlim(-76, 76)
    ax.set_ylim(-17, 31)
    ax.set_aspect("auto")
    ax.set_yticks([])
    ax.set_xticks([-65, -48, 0, 48, 65])
    ax.set_xlabel("Carriage position [mm] · target corridor respects the 15 mm edge exclusion",
                  color="#cbd5e1", fontsize=9)
    ax.add_patch(Rectangle((-65, -3), 130, 6, color="#26364d", zorder=1))
    ax.add_patch(Rectangle((-65, -6), 15, 12, color="#8a5735", alpha=.65, zorder=2))
    ax.add_patch(Rectangle((50, -6), 15, 12, color="#8a5735", alpha=.65, zorder=2))
    target_line, = ax.plot([], [], marker="|", markersize=28, markeredgewidth=3,
                           color="#fbbf24", label="Current target")
    cart = Rectangle((0, -5), 9, 7, color="#38bdf8", ec="#bae6fd", lw=1.2, zorder=3)
    ax.add_patch(cart)
    rod, = ax.plot([], [], color="#fb7185", lw=6, solid_capstyle="round", zorder=4)
    bob = Circle((0, 0), 2.1, color="#fda4af", ec="white", lw=1.0, zorder=5)
    ax.add_patch(bob)
    frame_text = ax.text(.985, .94, "", transform=ax.transAxes, ha="right", va="top",
                         color="#cbd5e1", fontsize=10)
    ax.text(-57.5, -12.5, "15 mm exclusion", color="#f2b66d", ha="center", fontsize=8)
    ax.text(57.5, -12.5, "15 mm exclusion", color="#f2b66d", ha="center", fontsize=8)

    chart.fill_between(t, target_mm - 5, target_mm + 5, color="#fbbf24", alpha=.10,
                       label="±5 mm hold tolerance")
    chart.plot(t, x_mm, color="#38bdf8", lw=1.6, label="Carriage")
    chart.plot(t, target_mm, color="#fbbf24", lw=1.3, ls="--", label="Random goal")
    cursor = chart.axvline(t[0], color="white", alpha=.75, lw=1)
    chart.set_xlim(t[0], t[-1])
    chart.set_xlabel("Time [s]", color="#cbd5e1", fontsize=9)
    chart.set_ylabel("Position [mm]", color="#cbd5e1", fontsize=9)
    chart.legend(loc="upper right", ncol=3, frameon=False, fontsize=8, labelcolor="#e2e8f0")
    fig.text(.075, .91, f"RANDOM GOAL + HOLD · {stage.upper()} · {steps:,} TRAINING STEPS",
             color="white", fontsize=14, fontweight="bold")
    fig.text(.075, .865,
             f"Deterministic PPO playback · evaluation seed {seed} · ≥{cfg.goal_hold_min_seconds:g} s configured dwell",
             color="#94a3b8", fontsize=10)
    fig.text(.925, .045, "SEDP · saved checkpoint playback; not a promotion result",
             ha="right", color="#64748b", fontsize=8)

    def frame(index: int):
        carriage = x_mm[index]
        cart.set_x(carriage - 4.5)
        pivot = (carriage, 1.0)
        tip = (carriage + 15 * np.sin(theta[index]), 1.0 + 15 * np.cos(theta[index]))
        rod.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
        bob.center = tip
        target_line.set_data([target_mm[index]], [0])
        cursor.set_xdata([t[index], t[index]])
        frame_text.set_text(
            f"t = {t[index]:.2f} s   ·   goal {int(goal_index[index])}"
            f"   ·   residual action = {action[index]:+.3f}"
        )
        return cart, rod, bob, target_line, cursor, frame_text

    animation = FuncAnimation(fig, frame, frames=len(t), interval=1000 / 30, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=FFMpegWriter(
        fps=30, codec="libx264", bitrate=1400,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ))
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing policy_00008192.pt, policy_00032768.pt, and policy_00065536.pt")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for the three MP4 files")
    parser.add_argument("--seed", type=int, default=101,
                        help="Fixed playback seed used for all checkpoints (default: 101)")
    args = parser.parse_args()
    for steps, stage in STAGES:
        checkpoint = args.run_dir / f"policy_{steps:08d}.pt"
        output = args.output_dir / f"ppo_{steps:06d}.mp4"
        path = render_video(checkpoint, output, args.seed, steps, stage)
        print(f"{stage}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot cube pose CSV logs. Compare multiple runs by editing RUNS below."""

import csv
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Edit this list: comment/uncomment runs to compare stability between experiments.
# ---------------------------------------------------------------------------
RUNS = [
    {"label": "run1", "path": "/home/robert/run3.csv"},
    {"label": "run2", "path": "/home/robert/run4.csv"},
    # {"label": "run2", "path": "/tmp/cube_pose_run2.csv"},
    # {"label": "run3", "path": "/tmp/cube_pose_run3.csv"},
]

# Set True to block with an interactive matplotlib window after saving PNGs.
SHOW_PLOT = False

# Output directory for PNGs (one combined figure + per-run figures).
OUTPUT_DIR = "/home/robert/cube_pose_plots"


def load_run(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "stamp": float(r["stamp"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "roll_deg": float(r["roll_deg"]),
                "pitch_deg": float(r["pitch_deg"]),
                "yaw_deg": float(r["yaw_deg"]),
                "wx": float(r["wx"]),
                "wy": float(r["wy"]),
                "wz": float(r["wz"]),
            })
    if not rows:
        return None

    t = np.array([r["stamp"] for r in rows])
    t = t - t[0]
    data = {
        "t": t,
        "x": np.array([r["x"] for r in rows]),
        "y": np.array([r["y"] for r in rows]),
        "z": np.array([r["z"] for r in rows]),
        "roll": np.array([r["roll_deg"] for r in rows]),
        "pitch": np.array([r["pitch_deg"] for r in rows]),
        "yaw": np.array([r["yaw_deg"] for r in rows]),
        "wx": np.array([r["wx"] for r in rows]),
        "wy": np.array([r["wy"] for r in rows]),
        "wz": np.array([r["wz"] for r in rows]),
        "n": len(rows),
        "duration_s": float(t[-1]) if len(t) > 1 else 0.0,
    }
    return data


def _unwrap_deg(series):
    return np.degrees(np.unwrap(np.radians(series)))


def compute_smoothness_metrics(data):
    t = data["t"]
    n = data["n"]
    if n < 3:
        return {}

    dt = np.diff(t)
    dt = np.where(dt > 1e-6, dt, np.nan)

    pos = np.vstack([data["x"], data["y"], data["z"]])
    dpos = np.diff(pos, axis=1) / dt
    pos_jerk_proxy = np.nanstd(np.linalg.norm(dpos, axis=0))

    roll_u = _unwrap_deg(data["roll"])
    pitch_u = _unwrap_deg(data["pitch"])
    yaw_u = _unwrap_deg(data["yaw"])
    rot = np.vstack([roll_u, pitch_u, yaw_u])
    drot = np.diff(rot, axis=1) / dt
    rot_rate_std = np.nanstd(np.linalg.norm(drot, axis=0))

    omega = np.vstack([data["wx"], data["wy"], data["wz"]])
    omega_mag = np.linalg.norm(omega, axis=0)

    return {
        "pos_std_x": float(np.std(data["x"])),
        "pos_std_y": float(np.std(data["y"])),
        "pos_std_z": float(np.std(data["z"])),
        "pos_jitter_proxy": float(pos_jerk_proxy),
        "rot_rate_std_deg_s": float(rot_rate_std),
        "omega_mag_mean": float(np.mean(omega_mag)),
        "omega_mag_max": float(np.max(omega_mag)),
        "sample_rate_hz": float((n - 1) / data["duration_s"]) if data["duration_s"] > 0 else 0.0,
    }


def print_summary(label, data, metrics):
    print("--- %s ---" % label)
    print("  samples: %d  duration: %.2f s  rate: %.1f Hz" % (
        data["n"], data["duration_s"], metrics.get("sample_rate_hz", 0.0)))
    print("  position std (m): x=%.5f y=%.5f z=%.5f" % (
        metrics["pos_std_x"], metrics["pos_std_y"], metrics["pos_std_z"]))
    print("  position delta std (m/s proxy): %.5f" % metrics["pos_jitter_proxy"])
    print("  rotation rate std (deg/s): %.3f" % metrics["rot_rate_std_deg_s"])
    print("  reported omega |w| mean/max (rad/s): %.3f / %.3f" % (
        metrics["omega_mag_mean"], metrics["omega_mag_max"]))


def plot_compare(runs_data, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs_data), 1)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)

    for i, (label, data) in enumerate(runs_data):
        c = colors[i]
        t = data["t"]
        axes[0].plot(t, data["x"], color=c, label="%s x" % label)
        axes[0].plot(t, data["y"], color=c, linestyle="--", alpha=0.8)
        axes[0].plot(t, data["z"], color=c, linestyle=":", alpha=0.8)
        axes[1].plot(t, _unwrap_deg(data["roll"]), color=c, label="%s roll" % label)
        axes[1].plot(t, _unwrap_deg(data["pitch"]), color=c, linestyle="--", alpha=0.8)
        axes[1].plot(t, _unwrap_deg(data["yaw"]), color=c, linestyle=":", alpha=0.8)
        omega_mag = np.linalg.norm(np.vstack([data["wx"], data["wy"], data["wz"]]), axis=0)
        axes[2].plot(t, omega_mag, color=c, label=label)
        if len(t) > 2:
            dt = np.diff(t)
            dt = np.where(dt > 1e-6, dt, np.nan)
            pos = np.vstack([data["x"], data["y"], data["z"]])
            speed = np.linalg.norm(np.diff(pos, axis=1) / dt, axis=0)
            axes[3].plot(t[1:], speed, color=c, label=label)

    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Cube position over time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_ylabel("Angle (deg, unwrapped)")
    axes[1].set_title("Cube orientation (roll / pitch / yaw)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_ylabel("|omega| (rad/s)")
    axes[2].set_title("Reported angular speed magnitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    axes[3].set_ylabel("|dpos/dt| (m/s)")
    axes[3].set_xlabel("Time since start (s)")
    axes[3].set_title("Finite-difference linear speed (smoothness proxy)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(output_dir, "compare_all_runs.png")
    plt.show(block=True)
    print("Saved:", out)
    return fig


def plot_single_run(label, data, output_dir):
    import matplotlib.pyplot as plt

    t = data["t"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(t, data["x"], label="x")
    axes[0, 0].plot(t, data["y"], label="y")
    axes[0, 0].plot(t, data["z"], label="z")
    axes[0, 0].set_title("%s: position" % label)
    axes[0, 0].set_ylabel("m")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, _unwrap_deg(data["roll"]), label="roll")
    axes[0, 1].plot(t, _unwrap_deg(data["pitch"]), label="pitch")
    axes[0, 1].plot(t, _unwrap_deg(data["yaw"]), label="yaw")
    axes[0, 1].set_title("%s: orientation" % label)
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    omega_mag = np.linalg.norm(np.vstack([data["wx"], data["wy"], data["wz"]]), axis=0)
    axes[1, 0].plot(t, omega_mag)
    axes[1, 0].set_title("%s: |omega| from odom" % label)
    axes[1, 0].set_ylabel("rad/s")
    axes[1, 0].grid(True, alpha=0.3)

    if len(t) > 2:
        dt = np.diff(t)
        dt = np.where(dt > 1e-6, dt, np.nan)
        rot = np.vstack([
            _unwrap_deg(data["roll"]),
            _unwrap_deg(data["pitch"]),
            _unwrap_deg(data["yaw"]),
        ])
        rot_rate = np.linalg.norm(np.diff(rot, axis=1) / dt, axis=0)
        axes[1, 1].plot(t[1:], rot_rate)
    axes[1, 1].set_title("%s: euler rate (finite diff)" % label)
    axes[1, 1].set_ylabel("deg/s")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(data["x"], data["y"])
    axes[2, 0].set_title("%s: XY path" % label)
    axes[2, 0].set_xlabel("x (m)")
    axes[2, 0].set_ylabel("y (m)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_aspect("equal", adjustable="box")

    axes[2, 1].hist(omega_mag, bins=30, color="tab:gray", edgecolor="black", alpha=0.8)
    axes[2, 1].set_title("%s: |omega| histogram" % label)
    axes[2, 1].set_xlabel("rad/s")
    axes[2, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax != axes[2, 0] and ax != axes[2, 1]:
            ax.set_xlabel("time (s)")

    fig.tight_layout()
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    out = os.path.join(output_dir, "single_%s.png" % safe_label)
    fig.savefig(out, dpi=160)
    print("Saved:", out)
    return fig


def main():
    active_runs = [r for r in RUNS if os.path.isfile(r["path"])]
    if not active_runs:
        print("No CSV files found. Edit RUNS in plot_cube_pose.py and log data first:")
        print("  rosrun apriltag_ros cube_pose_logger.py _csv_path:=/tmp/cube_pose_run1.csv")
        return

    runs_data = []
    print("=== Cube pose run summary ===\n")
    for run in active_runs:
        data = load_run(run["path"])
        if data is None:
            print("Skipping empty file:", run["path"])
            continue
        metrics = compute_smoothness_metrics(data)
        print_summary(run["label"], data, metrics)
        runs_data.append((run["label"], data))
        print()

    if not runs_data:
        print("No valid samples in any CSV.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot generate plots.")
        return

    if len(runs_data) >= 1:
        fig_compare = plot_compare(runs_data, OUTPUT_DIR)
        if SHOW_PLOT:
            plt.figure(fig_compare.number)
            plt.show()

    for label, data in runs_data:
        fig = plot_single_run(label, data, OUTPUT_DIR)
        if SHOW_PLOT:
            plt.figure(fig.number)
            plt.show()

    print("\nDone. Plots in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

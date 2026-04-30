#!/usr/bin/env python

import argparse
import csv
import math
import os


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(s)) - 1)
    idx = max(0, min(idx, len(s) - 1))
    return s[idx]


def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "stamp": float(r["stamp"]),
                "measured_dist_m": float(r["measured_dist_m"]),
                "known_dist_m": float(r["known_dist_m"]),
                "signed_error_m": float(r["signed_error_m"]),
                "abs_error_m": float(r["abs_error_m"]),
            })
    return rows


def print_stats(rows):
    n = len(rows)
    if n == 0:
        print("No samples found.")
        return
    signed = [r["signed_error_m"] for r in rows]
    abs_e = [r["abs_error_m"] for r in rows]
    meas = [r["measured_dist_m"] for r in rows]
    known = [r["known_dist_m"] for r in rows]
    known_ref = known[0]
    mean_signed = sum(signed) / n
    mae = sum(abs_e) / n
    rmse = math.sqrt(sum(v * v for v in signed) / n)
    std = math.sqrt(sum((v - mean_signed) ** 2 for v in signed) / n)

    print("=== Distance Error Summary ===")
    print("samples:", n)
    print("known_distance_m:", "%.6f" % known_ref)
    print("measured_range_m:", "%.6f .. %.6f" % (min(meas), max(meas)))
    print("mean_signed_error_m:", "%.6f" % mean_signed)
    print("mae_m:", "%.6f" % mae)
    print("rmse_m:", "%.6f" % rmse)
    print("std_signed_error_m:", "%.6f" % std)
    print("abs_error_p50_m:", "%.6f" % percentile(abs_e, 50))
    print("abs_error_p90_m:", "%.6f" % percentile(abs_e, 90))
    print("abs_error_p95_m:", "%.6f" % percentile(abs_e, 95))
    print("abs_error_p99_m:", "%.6f" % percentile(abs_e, 99))

    # Suggested simulation noise bounds.
    bound95 = max(abs(mean_signed) + 2.0 * std, percentile(abs_e, 95))
    bound99 = max(abs(mean_signed) + 3.0 * std, percentile(abs_e, 99))
    print("\n=== Suggested Training Bounds (meters) ===")
    print("bias_mean_m:", "%.6f" % mean_signed)
    print("noise_std_m:", "%.6f" % std)
    print("bound_95pct_m:", "%.6f" % bound95)
    print("bound_99pct_m:", "%.6f" % bound99)
    print("example_uniform_range_95: [-%.6f, +%.6f]" % (bound95, bound95))
    print("example_uniform_range_99: [-%.6f, +%.6f]" % (bound99, bound99))


def maybe_plot(rows, output_png, show_plot=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot generation.")
        return

    t0 = rows[0]["stamp"] if rows else 0.0
    t = [r["stamp"] - t0 for r in rows]
    measured = [r["measured_dist_m"] for r in rows]
    known = [r["known_dist_m"] for r in rows]
    signed = [r["signed_error_m"] for r in rows]
    abs_e = [r["abs_error_m"] for r in rows]
    mean_signed = sum(signed) / len(signed)
    std_signed = math.sqrt(sum((v - mean_signed) ** 2 for v in signed) / len(signed))
    p95 = percentile(abs_e, 95)
    p99 = percentile(abs_e, 99)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=False)

    axes[0].plot(t, measured, label="Measured distance")
    axes[0].plot(t, known, "--", label="Known distance")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("Measured vs Known Distance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, signed, color="tab:orange")
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].axhline(mean_signed, color="tab:blue", linestyle="--", linewidth=1, label="Mean bias")
    axes[1].axhline(mean_signed + 2.0 * std_signed, color="tab:purple", linestyle=":", linewidth=1, label="+2 sigma")
    axes[1].axhline(mean_signed - 2.0 * std_signed, color="tab:purple", linestyle=":", linewidth=1, label="-2 sigma")
    axes[1].set_ylabel("Signed error (m)")
    axes[1].set_title("Signed Error Over Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, abs_e, color="tab:red")
    axes[2].axhline(p95, color="tab:green", linestyle="--", linewidth=1, label="P95 abs error")
    axes[2].axhline(p99, color="tab:brown", linestyle="--", linewidth=1, label="P99 abs error")
    axes[2].set_ylabel("Absolute error (m)")
    axes[2].set_title("Absolute Error Over Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].hist(signed, bins=40, color="tab:gray", edgecolor="black", alpha=0.8)
    axes[3].axvline(mean_signed, color="tab:blue", linestyle="--", linewidth=1, label="Mean bias")
    axes[3].axvline(mean_signed + std_signed, color="tab:purple", linestyle=":", linewidth=1, label="+1 sigma")
    axes[3].axvline(mean_signed - std_signed, color="tab:purple", linestyle=":", linewidth=1, label="-1 sigma")
    axes[3].set_xlabel("Signed error (m)")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Signed Error Distribution")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[2].set_xlabel("Time since start (s)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    print("\nSaved plot:", output_png)
    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze AprilTag distance-error CSV.")
    parser.add_argument("csv_path", help="Path to distance_error_logger CSV file")
    parser.add_argument(
        "--plot-output",
        default="",
        help="Optional output image path. Default: <csv_basename>_plot.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib window in addition to saving the PNG.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    print_stats(rows)

    if not rows:
        return

    plot_out = args.plot_output
    if not plot_out:
        root, _ = os.path.splitext(args.csv_path)
        plot_out = root + "_plot.png"
    maybe_plot(rows, plot_out, show_plot=args.show)


if __name__ == "__main__":
    main()

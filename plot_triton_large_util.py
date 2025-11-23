"""Plot Triton SD-KDE utilization for the large-n scaling sweep."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HEADER_RE = re.compile(
    r"====\s*Triton only:\s*n_train=(?P<n_train>\d+),\s*n_test=(?P<n_test>\d+)\s*===="
)
SUMMARY_RE = re.compile(
    r"\[Empirical kernel only\]\s+avg=(?P<avg>[0-9.]+)\s+ms,\s+std=(?P<std>[0-9.]+)\s+ms"
)

PEAK_TFLOPS = 40.0  # Approximate FP32 peak for RTX A6000
PEAK_FLOPS = PEAK_TFLOPS * 1e12


def estimate_flops_emp_sd_kde(k: int) -> float:
    """Estimate FLOPs using the 8-flop-per-exp hardware model."""
    c1 = 16.0  # score+shift per (train, train) pair
    c2 = 14.0  # KDE per (train, test) pair
    return c1 * (k ** 2) + c2 * (k ** 2 / 8.0)


def parse_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    current: Dict[str, float] | None = None
    with path.open("r") as fh:
        for line in fh:
            header = HEADER_RE.search(line)
            if header:
                current = {
                    "n_train": int(header.group("n_train")),
                    "n_test": int(header.group("n_test")),
                }
                continue
            if current is None:
                continue
            summary = SUMMARY_RE.search(line)
            if summary:
                current["triton_ms"] = float(summary.group("avg"))
                current["triton_std_ms"] = float(summary.group("std"))
                rows.append(current)
                current = None
    return rows


def plot_util(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No Triton summary lines found in log.")

    ks = np.array([r["n_train"] for r in rows], dtype=float)
    flops = np.array([estimate_flops_emp_sd_kde(int(k)) for k in ks], dtype=float)
    ms = np.array([r["triton_ms"] for r in rows], dtype=float)
    sec = ms / 1e3
    util = flops / (sec * PEAK_FLOPS) * 100.0

    print("n_train  n_test  FLOPs_est   Triton_ms  Triton_%peak")
    for r, f, tm, tu in zip(rows, flops, ms, util):
        print(
            f"{int(r['n_train']):9d}  {int(r['n_test']):7d}  {f:10.3e}  "
            f"{tm:10.3f}  {tu:12.3f}"
        )

    idx = np.arange(len(ks))
    width = 0.6

    plt.figure(figsize=(8, 4))
    bars = plt.bar(idx, util, width, color="#f28e2b", label="Triton SD-KDE")

    for rect, u, t in zip(bars, util, ms):
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        plt.text(
            x,
            y * 1.03,
            f"{t:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(idx, [f"{int(k):d}" for k in ks], rotation=45, ha="right")
    plt.ylabel("GPU utilization (\\% of A6000 FP32 peak)")
    plt.xlabel("$n_{\\text{train}}$ (power of two)")
    plt.title("Large-n Triton SD-KDE Utilization")
    ymax = np.nanmax(util) * 1.3
    plt.ylim(0, ymax)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Triton SD-KDE utilization from large-n scaling log."
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to scaling log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triton-large-util.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    plot_util(rows, args.output)
    print(f"Wrote Triton large-n utilization plot to {args.output}")


if __name__ == "__main__":
    main()

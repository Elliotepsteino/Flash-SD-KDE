"""Plot empirical SD-KDE GPU utilization (Triton vs Torch) vs n_train."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_RE = re.compile(
    r"\(n_train=(?P<n_train>\d+), n_test=(?P<n_test>\d+)\)\s+"
    r"(?P<label>[^:]+):\s+(?P<value>[0-9.]+)\s+ms"
)

PEAK_TFLOPS = 40.0  # Approximate FP32 peak for RTX A6000
PEAK_FLOPS = PEAK_TFLOPS * 1e12


def estimate_flops_emp_sd_kde(k: int) -> float:
    """Estimate total FLOPs for empirical SD-KDE for n_train=k, n_test=k/8.

    Model:
      - Score+shift: c1 * k^2 with c1 ≈ 40 flops per (train, train) pair
      - KDE on debiased data: c2 * k * (k/8) with c2 ≈ 20 flops per (train, test) pair
    """
    c1 = 40.0
    c2 = 20.0
    return c1 * (k ** 2) + c2 * (k ** 2 / 8.0)


def parse_log(path: Path) -> List[Dict[str, float]]:
    entries: Dict[int, Dict[str, float]] = {}
    with path.open("r") as fh:
        for line in fh:
            m = SUMMARY_RE.search(line)
            if not m:
                continue
            n_train = int(m.group("n_train"))
            n_test = int(m.group("n_test"))
            label = m.group("label").strip()
            value = float(m.group("value"))
            entry = entries.setdefault(
                n_train, {"n_train": n_train, "n_test": n_test}
            )

            if "SD-KDE GPU (Triton" in label:
                entry["triton_ms"] = value
            elif "SD-KDE GPU (Torch" in label:
                entry["torch_ms"] = value

    rows = [entries[k] for k in sorted(entries)]
    return rows


def plot_util(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No empirical SD-KDE summary lines found in log.")

    ks = np.array([r["n_train"] for r in rows], dtype=float)
    flops = np.array([estimate_flops_emp_sd_kde(int(k)) for k in ks], dtype=float)

    triton_ms = np.array([r.get("triton_ms", np.nan) for r in rows], dtype=float)
    torch_ms = np.array([r.get("torch_ms", np.nan) for r in rows], dtype=float)

    triton_s = triton_ms / 1e3
    torch_s = torch_ms / 1e3

    util_triton = flops / (triton_s * PEAK_FLOPS) * 100.0
    util_torch = flops / (torch_s * PEAK_FLOPS) * 100.0

    print("n_train  n_test  FLOPs_est   Triton_ms  Triton_%peak  Torch_ms  Torch_%peak")
    for r, f, tm, tu, om, ou in zip(rows, flops, triton_ms, util_triton, torch_ms, util_torch):
        print(
            f"{r['n_train']:7d}  {r['n_test']:7d}  {f:10.3e}  "
            f"{tm:9.3f}  {tu:11.3f}  {om:11.3f}  {ou:13.3f}"
        )

    idx = np.arange(len(ks))
    width = 0.35

    plt.figure(figsize=(8, 4))
    # Colors chosen to match the main runtime plot:
    # Torch (green), Triton (orange).
    bar_torch = plt.bar(
        idx - width / 2, util_torch, width, label="SD-KDE Torch", color="#59a14f"
    )
    bar_triton = plt.bar(
        idx + width / 2, util_triton, width, label="SD-KDE Triton", color="#f28e2b"
    )

    # Annotate bars with utilization values
    for bars, values in [(bar_torch, util_torch), (bar_triton, util_triton)]:
        for rect, val in zip(bars, values):
            if np.isnan(val):
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            y = rect.get_height()
            plt.text(
                x,
                y * 1.03,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(idx, [f"{int(k):d}" for k in ks], rotation=45, ha="right")
    plt.xlabel("$n_{\\text{train}}$")
    plt.ylabel("GPU utilization (\\% of A6000 FP32 peak)")
    plt.title("SD-KDE GPU Utilization (Torch vs Triton)")
    # Give a bit of headroom above the tallest bar
    ymax = np.nanmax([util_triton.max(), util_torch.max()]) * 1.2
    plt.ylim(0, ymax)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot SD-KDE GPU utilization (Triton vs Torch)."
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to sweep log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("emp-sd-kde-util.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    plot_util(rows, args.output)
    print(f"Wrote utilization plot to {args.output}")


if __name__ == "__main__":
    main()

"""Plot utilization for the 16-D SD-KDE benchmark (Torch vs Triton)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HEADER_RE = re.compile(
    r"====\s*16D SD-KDE:\s*n_train=(?P<n_train>\d+),\s*n_test=(?P<n_test>\d+)\s*===="
)
SUMMARY_RE_FULL = re.compile(
    r"\[16D SD-KDE\]\s+avg Torch=(?P<torch>[0-9.]+)\s+ms,\s+avg Triton="
    r"(?P<triton>[0-9.]+)\s+ms"
)
SUMMARY_RE_TRITON_ONLY = re.compile(
    r"\[16D SD-KDE\]\s+Triton-only avg runtime=(?P<triton>[0-9.]+)\s+ms"
)

PEAK_TFLOPS = 40.0  # RTX A6000 FP32 peak
PEAK_FLOPS = PEAK_TFLOPS * 1e12
DIM = 16


def estimate_flops_sd_kde_nd(n_train: int, n_test: int, dim: int = DIM) -> float:
    """Estimate FLOPs for 16-D SD-KDE (score+shift+KDE)."""
    n = float(n_train)
    m = float(n_test)

    # Score stage (train vs train)
    pairs_tt = n * n
    dot_score = 2.0 * dim * pairs_tt  # X X^T
    phi_weight = 2.0 * dim * pairs_tt  # Φ X
    extras_score = (4.0 + 8.0) * pairs_tt  # norms/dist + exp (SFU→8 flops)
    score_flops = dot_score + phi_weight + extras_score

    # KDE stage (queries vs debiased train)
    pairs_qt = m * n
    dot_kde = 2.0 * dim * pairs_qt
    extras_kde = (4.0 + 8.0) * pairs_qt
    kde_flops = dot_kde + extras_kde

    return score_flops + kde_flops


def parse_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    current: Dict[str, float] | None = None

    with path.open("r") as fh:
        for line in fh:
            header = HEADER_RE.search(line)
            if header:
                current = {
                    "n_train": float(header.group("n_train")),
                    "n_test": float(header.group("n_test")),
                }
                continue
            if current is None:
                continue
            summary_full = SUMMARY_RE_FULL.search(line)
            if summary_full:
                current["torch_ms"] = float(summary_full.group("torch"))
                current["triton_ms"] = float(summary_full.group("triton"))
                rows.append(current)
                current = None
                continue
            summary_triton_only = SUMMARY_RE_TRITON_ONLY.search(line)
            if summary_triton_only:
                current["torch_ms"] = float("nan")
                current["triton_ms"] = float(summary_triton_only.group("triton"))
                rows.append(current)
                current = None
    return rows


def plot_util(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No 16-D SD-KDE summary lines found in log.")

    ks = np.array([r["n_train"] for r in rows], dtype=float)
    flops = np.array(
        [estimate_flops_sd_kde_nd(int(r["n_train"]), int(r["n_test"])) for r in rows],
        dtype=float,
    )

    torch_ms = np.array([r["torch_ms"] for r in rows], dtype=float)
    triton_ms = np.array([r["triton_ms"] for r in rows], dtype=float)
    torch_s = torch_ms / 1e3
    triton_s = triton_ms / 1e3

    util_torch = flops / (torch_s * PEAK_FLOPS) * 100.0
    util_triton = flops / (triton_s * PEAK_FLOPS) * 100.0

    print(
        "n_train  n_test  FLOPs_est   Torch_ms  Torch_%peak  "
        "Triton_ms  Triton_%peak"
    )
    for r, f, tm, tu, sm, su in zip(
        rows, flops, torch_ms, util_torch, triton_ms, util_triton
    ):
        print(
            f"{int(r['n_train']):7d}  {int(r['n_test']):7d}  {f:10.3e}  "
            f"{tm:9.3f}  {tu:11.3f}  {sm:11.3f}  {su:13.3f}"
        )

    idx = np.arange(len(ks))
    width = 0.35
    plt.figure(figsize=(8, 4))
    has_torch = not np.all(np.isnan(util_torch))
    if has_torch:
        bars_torch = plt.bar(
            idx - width / 2,
            util_torch,
            width,
            label="Torch 16D SD-KDE",
            color="#59a14f",
        )
    bars_triton = plt.bar(
        idx + (0 if not has_torch else width / 2),
        util_triton,
        width,
        label="Triton 16D SD-KDE",
        color="#f28e2b",
    )

    if has_torch:
        for rect, runtime in zip(bars_torch, torch_ms):
            x = rect.get_x() + rect.get_width() / 2.0
            y = rect.get_height()
            plt.text(
                x,
                y * 1.03,
                f"{runtime:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for rect, runtime in zip(bars_triton, triton_ms):
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        plt.text(
            x,
            y * 1.03,
            f"{runtime:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(idx, [f"{int(k):d}" for k in ks], rotation=45, ha="right")
    plt.ylabel("GPU utilization (\\% of A6000 FP32 peak)")
    plt.xlabel("$n_{\\text{train}}$")
    plt.title("16-D SD-KDE Utilization")
    util_values = [util_triton]
    if has_torch:
        util_values.append(util_torch)
    ymax = np.nanmax([np.nanmax(u) for u in util_values]) * 1.25
    plt.ylim(0, ymax)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot utilization for the 16-D SD-KDE Triton benchmark."
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to sweep log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triton-sd-kde-nd-util.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    plot_util(rows, args.output)
    print(f"Wrote 16-D SD-KDE utilization plot to {args.output}")


if __name__ == "__main__":
    main()

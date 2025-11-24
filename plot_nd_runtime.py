"""Plot runtime comparison for 16-D KDE/SD-KDE (sklearn, Torch, Triton)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HEADER_KDE = re.compile(
    r"====\s*16D KDE Runtime:\s*n_train=(?P<n_train>\d+),\s*n_test=(?P<n_test>\d+)\s*===="
)
HEADER_SD = re.compile(
    r"====\s*16D SD-KDE Runtime:\s*n_train=(?P<n_train>\d+),\s*n_test=(?P<n_test>\d+)\s*===="
)
SUMMARY_KDE = re.compile(
    r"\[16D KDE\]\s+avg sklearn=(?P<sk>[0-9.]+)\s+ms,\s+avg Triton="
    r"(?P<triton>[0-9.]+)\s+ms"
)
SUMMARY_SD = re.compile(
    r"\[16D SD-KDE\]\s+avg Torch=(?P<torch>[0-9.]+)\s+ms,\s+avg Triton="
    r"(?P<triton>[0-9.]+)\s+ms"
)


def parse_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    current: Dict[str, float] | None = None
    phase: str | None = None

    with path.open("r") as fh:
        for line in fh:
            if match := HEADER_KDE.search(line):
                phase = "kde"
                current = {
                    "n_train": float(match.group("n_train")),
                    "n_test": float(match.group("n_test")),
                }
                continue
            if match := HEADER_SD.search(line):
                phase = "sd"
                continue

            if phase == "kde" and current is not None:
                summary = SUMMARY_KDE.search(line)
                if summary:
                    current["sklearn_ms"] = float(summary.group("sk"))
                    current["kde_triton_ms"] = float(summary.group("triton"))
                    rows.append(current)
                    current = None
                    phase = None
                continue

            if phase == "sd":
                summary = SUMMARY_SD.search(line)
                if summary:
                    n_train = rows[-1]["n_train"] if rows else None
                    if n_train is not None:
                        rows[-1]["sd_torch_ms"] = float(summary.group("torch"))
                        rows[-1]["sd_triton_ms"] = float(summary.group("triton"))
                    phase = None
                continue

    # Filter rows that captured both sections
    rows = [
        r
        for r in rows
        if {"sklearn_ms", "kde_triton_ms", "sd_torch_ms", "sd_triton_ms"} <= r.keys()
    ]
    rows.sort(key=lambda r: r["n_train"])
    return rows


def plot_runtime(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No complete 16-D runtime entries found.")

    ks = np.array([r["n_train"] for r in rows], dtype=float)
    sklearn = np.array([r["sklearn_ms"] for r in rows])
    kde_triton = np.array([r["kde_triton_ms"] for r in rows])
    sd_torch = np.array([r["sd_torch_ms"] for r in rows])
    sd_triton = np.array([r["sd_triton_ms"] for r in rows])

    idx = np.arange(len(ks))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    bars_sk = plt.bar(
        idx - width,
        sklearn,
        width,
        label="sklearn KDE",
        color="#4e79a7",
    )
    bars_sd_torch = plt.bar(
        idx,
        sd_torch,
        width,
        label="SD-KDE Torch",
        color="#59a14f",
    )
    bars_sd_triton = plt.bar(
        idx + width,
        sd_triton,
        width,
        label="SD-KDE Triton",
        color="#f28e2b",
    )

    for bars, values in [
        (bars_sk, sklearn),
        (bars_sd_torch, sd_torch),
        (bars_sd_triton, sd_triton),
    ]:
        for rect, val in zip(bars, values):
            x = rect.get_x() + rect.get_width() / 2.0
            y = rect.get_height()
            plt.text(
                x,
                y * 1.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(idx, [f"{int(k):d}" for k in ks], rotation=45, ha="right")
    plt.xlabel("$n_{\\text{train}}$")
    plt.ylabel("Runtime (ms)")
    plt.title("16-D KDE / SD-KDE Runtime Comparison")
    plt.yscale("log")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot runtime comparison for 16-D KDE/SD-KDE."
    )
    parser.add_argument("--log", type=Path, required=True, help="Runtime log path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nd-runtime.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    plot_runtime(rows, args.output)
    print(f"Wrote runtime comparison plot to {args.output}")


if __name__ == "__main__":
    main()

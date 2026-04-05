"""Plot fixed-n_train 16-D query batching runtimes and speedups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["rows"].sort(key=lambda row: row["n_test"])
    return payload


def plot_query_sweep(payload: dict[str, object], output: Path) -> None:
    rows = list(payload["rows"])
    if not rows:
        raise ValueError("No runtime rows found in benchmark JSON.")

    n_test = np.array([row["n_test"] for row in rows], dtype=float)
    n_train = int(payload["n_train"])

    runtime_series = [
        ("sd_torch_ms", "SD-KDE (Torch)", "#59a14f", "s"),
        ("sd_torch_compile_ms", "SD-KDE (Torch compile)", "#76b7b2", "P"),
        ("sd_pykeops_ms", "SD-KDE (PyKeOps)", "#e15759", "^"),
        ("flash_sd_kde_ms", "Flash-SD-KDE", "#f28e2b", "D"),
    ]
    speedup_series = [
        ("speedup_flash_vs_torch", "vs Torch", "#59a14f", "s"),
        ("speedup_flash_vs_torch_compile", "vs Torch compile", "#76b7b2", "P"),
        ("speedup_flash_vs_pykeops", "vs PyKeOps", "#e15759", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))

    for key, label, color, marker in runtime_series:
        values = np.array([row[key] for row in rows], dtype=float)
        if not np.isfinite(values).any():
            continue
        axes[0].plot(
            n_test,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    for key, label, color, marker in speedup_series:
        values = np.array([row[key] for row in rows], dtype=float)
        if not np.isfinite(values).any():
            continue
        axes[1].plot(
            n_test,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_test, [f"{int(n):d}" for n in n_test], rotation=45, ha="right")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
        ax.set_xlabel(r"$n_{\mathrm{test}}$")

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].set_title("Absolute Runtime")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Flash Speedup (baseline / flash)")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(f"16-D Query-Level Batching Sweep at Fixed n_train={n_train}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fixed-n_train 16-D query-batching runtimes and speedups."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Plot output path.")
    args = parser.parse_args()

    payload = _load_payload(args.input)
    plot_query_sweep(payload, args.output)
    print(f"Wrote rebuttal query-sweep plot to {args.output}")


if __name__ == "__main__":
    main()

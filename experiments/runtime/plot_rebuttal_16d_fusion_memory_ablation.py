"""Plot 16-D Flash-SD-KDE fusion/memory ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_METHODS = (
    ("flash_fused_tc", "Fused + Tensor Cores", "#f28e2b", "D"),
    ("flash_fused_no_tc", "Fused + No Tensor Cores", "#76b7b2", "P"),
    ("flash_defused_tc", "De-fused + Tensor Cores", "#e15759", "^"),
    ("flash_defused_no_tc", "De-fused + No Tensor Cores", "#4e79a7", "s"),
)


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["rows"].sort(key=lambda row: row["n_train"])
    return payload


def plot_fusion_memory_ablation(payload: dict[str, object], output: Path) -> None:
    rows = list(payload["rows"])
    if not rows:
        raise ValueError("No runtime rows found in benchmark JSON.")

    n_train = np.array([row["n_train"] for row in rows], dtype=float)
    n_test = np.array([row["n_test"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))

    for key, label, color, marker in _METHODS:
        runtime = np.array([row[f"{key}_ms"] for row in rows], dtype=float)
        memory = np.array([row[f"{key}_memory_peak_extra_allocated_mb"] for row in rows], dtype=float)
        axes[0, 0].plot(
            n_train,
            runtime,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )
        axes[0, 1].plot(
            n_train,
            memory,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    fused_speedup_tc = np.array([row["speedup_fused_vs_defused_tc"] for row in rows], dtype=float)
    fused_speedup_no_tc = np.array([row["speedup_fused_vs_defused_no_tc"] for row in rows], dtype=float)
    mem_reduction_tc = np.array([row["memory_reduction_fused_vs_defused_tc"] for row in rows], dtype=float)
    mem_reduction_no_tc = np.array([row["memory_reduction_fused_vs_defused_no_tc"] for row in rows], dtype=float)

    axes[1, 0].plot(n_train, fused_speedup_tc, color="#f28e2b", marker="D", linewidth=2.2, label="Tensor Cores")
    axes[1, 0].plot(n_train, fused_speedup_no_tc, color="#4e79a7", marker="s", linewidth=2.2, label="No Tensor Cores")
    axes[1, 1].plot(n_train, mem_reduction_tc, color="#f28e2b", marker="D", linewidth=2.2, label="Tensor Cores")
    axes[1, 1].plot(n_train, mem_reduction_no_tc, color="#4e79a7", marker="s", linewidth=2.2, label="No Tensor Cores")

    for ax in axes.ravel():
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_train, [f"{int(n):d}" for n in n_train], rotation=45, ha="right")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
        ax.set_xlabel(r"$n_{\mathrm{train}}$")

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Runtime (ms)")
    axes[0, 0].set_title("Absolute Runtime")
    axes[0, 0].legend(frameon=False, fontsize=9)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Peak Extra Alloc (MB)")
    axes[0, 1].set_title("Peak Memory Above Inputs")

    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    axes[1, 0].set_ylabel("De-fused / Fused Runtime")
    axes[1, 0].set_title("Fusion Speedup")
    axes[1, 0].legend(frameon=False, fontsize=9)

    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    axes[1, 1].set_ylabel("De-fused / Fused Peak Memory")
    axes[1, 1].set_title("Fusion Memory Reduction")
    axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle("16-D Flash-SD-KDE Fusion / Memory Ablation")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.text(
        0.5,
        0.95,
        rf"$n_{{\mathrm{{test}}}} = n_{{\mathrm{{train}}}} / 8$"
        rf" ({int(n_test[0])} to {int(n_test[-1])} queries)",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 16-D Flash-SD-KDE fusion/memory ablation results."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Plot output path.")
    args = parser.parse_args()

    payload = _load_payload(args.input)
    plot_fusion_memory_ablation(payload, args.output)
    print(f"Wrote fusion/memory ablation plot to {args.output}")


if __name__ == "__main__":
    main()

"""Plot Flash-SD-KDE tensorcore ablation results for 16-D runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["rows"].sort(key=lambda row: row["n_train"])
    return payload


def plot_tensorcore_ablation(payload: dict[str, object], output: Path) -> None:
    rows = list(payload["rows"])
    if not rows:
        raise ValueError("No runtime rows found in benchmark JSON.")

    n_train = np.array([int(row["n_train"]) for row in rows], dtype=int)
    tc = np.array([row["flash_tensorcore_ms"] for row in rows], dtype=float)
    no_tc = np.array([row["flash_no_tensorcore_ms"] for row in rows], dtype=float)
    speedup = np.array([row["speedup_tensorcore_vs_no_tensorcore"] for row in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    axes[0].bar(x - width / 2, tc, width=width, color="#f28e2b", label="Tensor Core")
    axes[0].bar(x + width / 2, no_tc, width=width, color="#4e79a7", label="No Tensor Core")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [f"{n:d}" for n in n_train], rotation=45, ha="right")
    axes[0].set_xlabel(r"$n_{\mathrm{train}}$")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].set_title("Absolute Runtime")
    axes[0].grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].bar(x, speedup, width=0.58, color="#59a14f")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    axes[1].set_xticks(x, [f"{n:d}" for n in n_train], rotation=45, ha="right")
    axes[1].set_xlabel(r"$n_{\mathrm{train}}$")
    axes[1].set_ylabel("No-TC / TC Speedup")
    axes[1].set_title("Tensor Core Benefit")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    n_test = np.array([int(row["n_test"]) for row in rows], dtype=int)
    fig.suptitle("16-D Flash-SD-KDE Tensor Core Ablation")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.text(
        0.5,
        0.92,
        rf"$n_{{\mathrm{{test}}}} = n_{{\mathrm{{train}}}} / 8$"
        rf" ({n_test[0]} to {n_test[-1]} queries)",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Flash-SD-KDE tensorcore ablation results for 16-D runtime."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Plot output path.")
    args = parser.parse_args()

    payload = _load_payload(args.input)
    plot_tensorcore_ablation(payload, args.output)
    print(f"Wrote tensorcore ablation plot to {args.output}")


if __name__ == "__main__":
    main()

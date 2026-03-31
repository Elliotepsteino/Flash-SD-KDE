"""Plot rebuttal Figure 1 runtime comparison for 16-D KDE / SD-KDE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]
    rows.sort(key=lambda row: row["n_train"])
    return rows


def plot_runtime(rows: list[dict[str, float]], output: Path) -> None:
    if not rows:
        raise ValueError("No runtime rows found in benchmark JSON.")

    n_train = np.array([row["n_train"] for row in rows], dtype=float)
    n_test = np.array([row["n_test"] for row in rows], dtype=float)

    series = [
        ("sklearn_kde_ms", "sklearn KDE", "#4e79a7", "o"),
        ("sd_torch_ms", "SD-KDE (Torch)", "#59a14f", "s"),
        ("sd_torch_compile_ms", "SD-KDE (Torch compile)", "#76b7b2", "P"),
        ("sd_pykeops_ms", "SD-KDE (PyKeOps)", "#e15759", "^"),
        ("flash_sd_kde_ms", "Flash-SD-KDE", "#f28e2b", "D"),
    ]

    plt.figure(figsize=(8.8, 5.0))
    for key, label, color, marker in series:
        values = np.array([row[key] for row in rows], dtype=float)
        plt.plot(
            n_train,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(n_train, [f"{int(n):d}" for n in n_train], rotation=45, ha="right")
    plt.xlabel(r"$n_{\mathrm{train}}$")
    plt.ylabel("Runtime (ms)")
    plt.title("16-D KDE / SD-KDE Runtime Comparison")
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    plt.figtext(
        0.5,
        0.94,
        rf"$n_{{\mathrm{{test}}}} = n_{{\mathrm{{train}}}} / 8$"
        rf" ({int(n_test[0])} to {int(n_test[-1])} queries)",
        ha="center",
        fontsize=10,
    )
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rebuttal Figure 1 runtime comparison for 16-D KDE / SD-KDE."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Plot output path.")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    plot_runtime(rows, args.output)
    print(f"Wrote rebuttal Figure 1 plot to {args.output}")


if __name__ == "__main__":
    main()

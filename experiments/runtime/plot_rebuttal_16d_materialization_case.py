"""Plot single-case 16-D streaming vs materialization benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_METHODS = (
    ("streamed_tc", "Streamed\n+ TC", "#f28e2b"),
    ("streamed_no_tc", "Streamed\n+ no TC", "#76b7b2"),
    ("materialized_tc", "Materialized\n+ TC", "#e15759"),
    ("materialized_no_tc", "Materialized\n+ no TC", "#4e79a7"),
)


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_case(payload: dict[str, object], output: Path) -> None:
    row = payload["row"]
    labels = [label for _, label, _ in _METHODS]
    runtime = np.array([row[f"{key}_ms"] for key, _, _ in _METHODS], dtype=float)
    memory = np.array([row[f"{key}_peak_extra_allocated_mb"] for key, _, _ in _METHODS], dtype=float)
    explicit = np.array([row[f"{key}_explicit_materialized_kernel_mb"] for key, _, _ in _METHODS], dtype=float)
    colors = [color for _, _, color in _METHODS]
    x = np.arange(len(_METHODS), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))

    axes[0].bar(x, runtime, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].set_title("Runtime")
    axes[0].grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x, memory, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Peak Extra Alloc (MB)")
    axes[1].set_title("Measured Peak Memory")
    axes[1].grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

    axes[2].bar(x, explicit, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Explicit Materialized Kernel Peak (MB)")
    axes[2].set_title("Explicit Matrix Materialization")
    axes[2].grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        f"16-D Streaming vs Full Materialization at n_train={int(row['n_train'])}, n_test={int(row['n_test'])}"
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot single-case 16-D streaming vs materialization benchmark results."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = _load_payload(args.input)
    plot_case(payload, args.output)
    print(f"Wrote materialization case plot to {args.output}")


if __name__ == "__main__":
    main()

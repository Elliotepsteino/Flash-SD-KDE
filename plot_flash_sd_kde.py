"""Create a PDF plot comparing sklearn vs empirical SD-KDE runtimes."""

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


def parse_log(path: Path) -> List[Dict[str, float]]:
    entries: Dict[int, Dict[str, float]] = {}
    with path.open("r") as fh:
        for line in fh:
            match = SUMMARY_RE.search(line)
            if not match:
                continue
            n_train = int(match.group("n_train"))
            n_test = int(match.group("n_test"))
            label = match.group("label").strip()
            value = float(match.group("value"))
            entry = entries.setdefault(
                n_train, {"n_train": n_train, "n_test": n_test}
            )

            if "sklearn KDE" in label:
                entry["sklearn_ms"] = value
            elif "Empirical SD-KDE GPU (Triton" in label:
                entry["emp_triton_ms"] = value
            elif "Empirical SD-KDE GPU (Torch" in label:
                entry["emp_torch_ms"] = value

    rows = [entries[n] for n in sorted(entries)]
    return rows


def plot(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No entries parsed from log.")

    ns = [r["n_train"] for r in rows]
    ticks = [str(n) for n in ns]
    sklearn = [r.get("sklearn_ms", float("nan")) for r in rows]
    emp_triton = [r.get("emp_triton_ms", float("nan")) for r in rows]
    emp_torch = [r.get("emp_torch_ms", float("nan")) for r in rows]

    idx = np.arange(len(ns))
    width = 0.25

    plt.figure(figsize=(12, 5))
    bar_sk = plt.bar(idx - width, sklearn, width, label="sklearn KDE", color="#4e79a7")
    bar_to = plt.bar(idx, emp_torch, width, label="Empirical SD-KDE (Torch)", color="#59a14f")
    bar_tr = plt.bar(idx + width, emp_triton, width, label="Empirical SD-KDE (Triton)", color="#f28e2b")
    plt.yscale("log")

    # Annotate each bar with its runtime in ms
    for bars, values in [(bar_sk, sklearn), (bar_tr, emp_triton), (bar_to, emp_torch)]:
        for rect, val in zip(bars, values):
            if np.isnan(val) or val <= 0:
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            y = rect.get_height()
            plt.text(
                x,
                y * 1.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(idx, ticks, rotation=45, ha="right")
    plt.ylabel("Average runtime (ms, log scale)")
    plt.xlabel("$n_{\\text{train}}$")
    plt.title("sklearn vs Empirical SD-KDE GPU Runtime (Triton vs Torch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot sklearn vs Empirical SD-KDE GPU timings from sweep log."
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to sweep log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flash-sd-kde.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    plot(rows, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()

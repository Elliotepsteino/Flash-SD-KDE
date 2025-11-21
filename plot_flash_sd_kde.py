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
SK_SPEEDUP_RE = re.compile(r"speedup vs sklearn:\s+([0-9.]+|N/A)")


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

            if label.startswith("(n_train"):
                # Already handled
                continue
            if "sklearn KDE" in label:
                entry["sklearn_ms"] = value
            elif "Empirical SD-KDE GPU" in label:
                entry["emp_gpu_ms"] = value
                sk_speed = SK_SPEEDUP_RE.search(line)
                if sk_speed:
                    val = sk_speed.group(1)
                    entry["speedup"] = float(val) if val != "N/A" else None

    rows = [entries[n] for n in sorted(entries)]
    return rows


def plot(rows: List[Dict[str, float]], output: Path):
    if not rows:
        raise ValueError("No entries parsed from log.")

    ns = [r["n_train"] for r in rows]
    ticks = [str(n) for n in ns]
    sklearn = [r.get("sklearn_ms", float("nan")) for r in rows]
    emp = [r.get("emp_gpu_ms", float("nan")) for r in rows]
    speedups = [r.get("speedup") for r in rows]

    idx = np.arange(len(ns))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(idx - width / 2, sklearn, width, label="sklearn KDE", color="#4e79a7")
    plt.bar(idx + width / 2, emp, width, label="Empirical SD-KDE (GPU)", color="#f28e2b")
    plt.yscale("log")

    for i, (x, em, sp) in enumerate(zip(idx, emp, speedups)):
        if np.isnan(em):
            continue
        text = f"x{sp:.1f}" if sp is not None else "N/A"
        plt.text(
            x + width / 2,
            em * 1.15,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    plt.xticks(idx, ticks, rotation=45, ha="right")
    plt.ylabel("Average runtime (ms, log scale)")
    plt.xlabel("$n_{\\text{train}}$")
    plt.title("sklearn vs Empirical SD-KDE GPU Runtime")
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

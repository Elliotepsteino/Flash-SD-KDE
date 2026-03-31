from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.error_suite_16d.plotting import load_rows


FILE_STORAGE_ROOT = "file_storage"


def _latest_error_suite_results(repo_root: Path) -> Path | None:
    base = repo_root / FILE_STORAGE_ROOT / "error_suite_16d"
    if not base.exists():
        print(f"Missing error suite outputs under {base}; cannot build table.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No error_suite_16d runs found under {base}; cannot build table.")
        return None
    return runs[0]


def _fmt_pct(mean: float, std: float) -> str:
    return f"{100.0 * mean:.3f}% ± {100.0 * std:.3f}%"


def _fmt_mass(mean: float, std: float) -> str:
    return f"{mean:.4g} ± {std:.2g}"


def _aggregate(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows
        if row.get("status") == "ok"
        and row.get("method") == method
        and row.get("n_train") is not None
        and row.get("clamped_fraction_laplace") is not None
        and row.get("integrated_negative_mass_laplace") is not None
        and row.get("negative_mass_fraction_laplace") is not None
    ]
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in filtered:
        buckets.setdefault(int(row["n_train"]), []).append(row)

    out = []
    for n_train in sorted(buckets):
        bucket = buckets[n_train]
        n_test = int(bucket[0]["n_test"])
        neg_est = np.asarray([float(row["clamped_fraction_laplace"]) for row in bucket], dtype=float)
        neg_mass = np.asarray([float(row["integrated_negative_mass_laplace"]) for row in bucket], dtype=float)
        neg_mass_frac = np.asarray([float(row["negative_mass_fraction_laplace"]) for row in bucket], dtype=float)
        out.append(
            {
                "n_train": n_train,
                "n_test": n_test,
                "repeats": len(bucket),
                "negative_estimate_fraction_mean": float(neg_est.mean()),
                "negative_estimate_fraction_std": float(neg_est.std(ddof=0)),
                "integrated_negative_mass_mean": float(neg_mass.mean()),
                "integrated_negative_mass_std": float(neg_mass.std(ddof=0)),
                "negative_mass_fraction_mean": float(neg_mass_frac.mean()),
                "negative_mass_fraction_std": float(neg_mass_frac.std(ddof=0)),
            }
        )
    return out


def _render_markdown(rows: list[dict[str, Any]], method: str) -> str:
    method_label = {
        "flash_laplace": "Flash-Laplace-KDE",
        "nonfused_laplace": "Laplace-corrected KDE (non-fused)",
    }.get(method, method)

    lines = [
        f"# {method_label} Negative-Mass Diagnostics",
        "",
        "| n_train | n_test | repeats | negative estimates | integrated negative mass | negative mass / |mass| |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_train']} | {row['n_test']} | {row['repeats']} | "
            f"{_fmt_pct(row['negative_estimate_fraction_mean'], row['negative_estimate_fraction_std'])} | "
            f"{_fmt_mass(row['integrated_negative_mass_mean'], row['integrated_negative_mass_std'])} | "
            f"{_fmt_pct(row['negative_mass_fraction_mean'], row['negative_mass_fraction_std'])} |"
        )
    lines.extend(
        [
            "",
            "`negative estimates` is the mean fraction of query evaluations that were nonpositive before clamping.",
            "`integrated negative mass` is the Monte Carlo estimate of "
            r"$\int \max(-\hat p(x), 0)\,dx$.",
            "`negative mass / |mass|` is the estimated negative mass divided by "
            r"$\int |\hat p(x)|\,dx$.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Markdown table of Laplace negative-mass diagnostics.")
    parser.add_argument("--results_dir", default=None, help="Optional error_suite_16d run directory.")
    parser.add_argument("--method", default="flash_laplace", help="Method to summarize.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else _latest_error_suite_results(repo_root)
    if results_dir is None:
        return 1

    results_path = results_dir / "results.csv"
    if not results_path.exists():
        print(f"Missing results.csv in {results_dir}; cannot build table.")
        return 1

    rows = load_rows(results_path)
    aggregated = _aggregate(rows, args.method)
    if not aggregated:
        print(f"No rows found for method={args.method} in {results_path}.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(aggregated, args.method), encoding="utf-8")
    print(f"Wrote negative-mass table to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

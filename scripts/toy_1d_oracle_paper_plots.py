from __future__ import annotations

import argparse
import json
from pathlib import Path

from plots.plot_toy_1d_mog_oracle import _setup_style, plot_error_vs_n, plot_fused_vs_nonfused_runtime


FILE_STORAGE_ROOT = "file_storage"


def _latest_toy_results(repo_root: Path) -> Path | None:
    base = repo_root / FILE_STORAGE_ROOT / "benchmarks" / "toy_1d_mog_oracle"
    if not base.exists():
        print(f"Missing benchmark outputs under {base}; cannot plot toy_1d.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No toy_1d_mog_oracle runs found under {base}; cannot plot toy_1d.")
        return None
    return runs[0]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate toy 1D oracle paper plots.")
    parser.add_argument("--output", required=True, help="Directory to write plots.")
    parser.add_argument("--results_dir", default=None, help="Optional toy_1d_mog_oracle run directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir) if args.results_dir else _latest_toy_results(repo_root)
    if results_dir is None:
        return 0

    results_path = results_dir / "results.json"
    if not results_path.exists():
        print(f"Missing results.json in {results_dir}; cannot plot toy_1d.")
        return 0

    results = _read_json(results_path)

    _setup_style()
    plot_error_vs_n(output_dir, results, dpi=300)
    plot_fused_vs_nonfused_runtime(output_dir, results, dpi=300)

    print(f"Toy 1D oracle plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

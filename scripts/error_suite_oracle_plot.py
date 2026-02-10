from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
from pathlib import Path

from experiments.error_suite_16d.plotting import load_rows, plot_oracle_mise_miae_vs_n


FILE_STORAGE_ROOT = "file_storage"


def _latest_error_suite_results(repo_root: Path) -> Path | None:
    base = repo_root / FILE_STORAGE_ROOT / "error_suite_16d"
    if not base.exists():
        print(f"Missing error suite outputs under {base}; cannot plot oracle.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No error_suite_16d runs found under {base}; cannot plot oracle.")
        return None
    return runs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 16D oracle error plots from error suite results.")
    parser.add_argument("--output", required=True, help="Directory to write plots.")
    parser.add_argument("--results_dir", default=None, help="Optional error_suite_16d run directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir) if args.results_dir else _latest_error_suite_results(repo_root)
    if results_dir is None:
        return 0

    results_path = results_dir / "results.csv"
    if not results_path.exists():
        print(f"Missing results.csv in {results_dir}; cannot plot oracle.")
        return 0

    rows = load_rows(results_path)
    plot_oracle_mise_miae_vs_n(rows, output_dir)
    print(f"Oracle plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FILE_STORAGE_ROOT = "file_storage"


def _latest_toy_results(repo_root: Path) -> Path | None:
    base = repo_root / FILE_STORAGE_ROOT / "benchmarks" / "toy_1d_mog_oracle"
    if not base.exists():
        print(f"Missing benchmark outputs under {base}; skipping toy_1d plots.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No toy_1d_mog_oracle runs found under {base}; skipping toy_1d plots.")
        return None
    return runs[0]


def _latest_error_suite_results(repo_root: Path) -> Path | None:
    base = repo_root / FILE_STORAGE_ROOT / "error_suite_16d"
    if not base.exists():
        print(f"Missing error suite outputs under {base}; skipping 16D oracle plots.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No error_suite_16d runs found under {base}; skipping 16D oracle plots.")
        return None
    return runs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect plots needed for paper validation.")
    parser.add_argument("--output", required=True, help="Directory to store generated plots.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output)

    latest = _latest_toy_results(repo_root)
    if latest is None:
        return 0

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "toy_1d_oracle_paper_plots.py"),
            "--output",
            str(out_dir),
            "--results_dir",
            str(latest),
        ],
        check=True,
    )

    error_suite = _latest_error_suite_results(repo_root)
    if error_suite is None:
        return 0

    env = os.environ.copy()
    env["ERROR_SUITE_RESULTS_DIR"] = str(error_suite)
    env["ERROR_SUITE_OUTPUT_DIR"] = str(out_dir)
    subprocess.run(
        [sys.executable, str(repo_root / "plots" / "plot_error_suite_16d.py")],
        check=True,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "error_suite_oracle_plot.py"),
            "--output",
            str(out_dir),
            "--results_dir",
            str(error_suite),
        ],
        check=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

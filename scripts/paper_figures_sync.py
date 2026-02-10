from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FIGURE_FILES = [
    "runtime_16d_kde_sdkde.pdf",
    "fig_oracle_error_vs_n_16d.pdf",
    "fig_oracle_error_vs_n_1d.pdf",
    "fig_fused_vs_nonfused_runtime.pdf",
    "util_16d_sdkde_tensorcore.pdf",
    "runtime_1d_kde_sdkde.pdf",
    "util_1d_empirical_sdkde.pdf",
]


def _latest_generated(root: Path) -> Path | None:
    base = root / "file_storage" / "paper_plots"
    if not base.exists():
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        gen = run / "generated"
        if gen.exists():
            return gen
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy generated paper plots into paper/figures.")
    parser.add_argument("--source", default=None, help="Directory containing generated plots.")
    parser.add_argument("--dest", default=None, help="Destination directory (default: paper/figures).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src = Path(args.source) if args.source else _latest_generated(repo_root)
    if src is None:
        print("No generated plot directory found. Run full_paper_experiments_plots first.")
        return 1
    if not src.exists():
        print(f"Source directory {src} does not exist.")
        return 1

    dest = Path(args.dest) if args.dest else repo_root / "paper" / "figures"
    dest.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in FIGURE_FILES:
        src_path = src / name
        if not src_path.exists():
            missing.append(name)
            continue
        shutil.copy2(src_path, dest / name)

    if missing:
        print("Missing files in source:")
        for name in missing:
            print(f"  - {name}")
        return 1

    print(f"Copied {len(FIGURE_FILES)} figures to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

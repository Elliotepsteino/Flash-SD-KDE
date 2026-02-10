from __future__ import annotations

import argparse
import shutil
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
    base = repo_root / FILE_STORAGE_ROOT / "error_suite_a100_16d"
    if not base.exists():
        print(f"Missing error suite outputs under {base}; skipping 16D oracle plots.")
        return None
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"No error_suite_a100_16d runs found under {base}; skipping 16D oracle plots.")
        return None
    return runs[0]


def _copy_if_present(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"Missing {src}; skipping.")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect plots needed for paper validation.")
    parser.add_argument("--output", required=True, help="Directory to store generated plots.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output)

    latest = _latest_toy_results(repo_root)
    if latest is None:
        return 0

    fig_dir = latest / "figures"
    for name in ("fig_oracle_error_vs_n.pdf", "fig_fused_vs_nonfused_runtime.pdf"):
        _copy_if_present(fig_dir / name, out_dir / name)

    error_suite = _latest_error_suite_results(repo_root)
    if error_suite is None:
        return 0

    error_fig_dir = error_suite / "plots"
    _copy_if_present(error_fig_dir / "fig_oracle_error_vs_n.png", out_dir / "fig_oracle_error_vs_n.png")
    _copy_if_present(
        error_fig_dir / "oracle_mise_miae_vs_n.pdf",
        out_dir / "oracle_16d_mise_miae_vs_n.pdf",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

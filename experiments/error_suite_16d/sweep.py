from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import csv
import shutil
import copy
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from experiments.error_suite_16d.config import expand_grid, load_config, resolve_out_dir, set_nested
from experiments.error_suite_16d.pareto import best_under_accuracy, best_under_speed, pareto_frontier
from experiments.error_suite_16d.plotting import make_plots, load_rows
from experiments.error_suite_16d.report import make_latex_tables, make_report
from experiments.error_suite_16d.run import run_one
from flash_sd_kde.utils import get_repo_state, write_json


def _result_columns() -> list[str]:
    return [
        "status",
        "error_type",
        "error_msg",
        "seed",
        "experiment_kind",
        "method",
        "dataset",
        "n_train",
        "n_test",
        "dim",
        "kernel",
        "output",
        "bandwidth",
        "bandwidth_mode",
        "bandwidth_rule",
        "bandwidth_scale",
        "bandwidth_value",
        "compute_dtype",
        "accumulate_dtype",
        "block_q",
        "block_n",
        "num_warps",
        "num_stages",
        "precision_mode",
        "kde_backend",
        "enable_tf32",
        "time_flash_ms",
        "time_flash_p90_ms",
        "time_ref_ms",
        "time_ref_p90_ms",
        "throughput_qps",
        "speedup",
        "max_memory_allocated_mb",
        "max_memory_reserved_mb",
        "max_abs_log_err",
        "mean_abs_log_err",
        "rmse_log_err",
        "p95_abs_log_err",
        "bias_log_err",
        "finite_fraction_flash",
        "finite_fraction_ref",
        "clamped_fraction_flash",
        "clamped_fraction_laplace",
        "negative_fraction_laplace",
        "integrated_negative_mass_laplace",
        "integrated_abs_mass_laplace",
        "negative_mass_fraction_laplace",
        "nll_hat",
        "nll_true",
        "kl_p_to_phat",
        "rmse_logp_err_true",
        "mean_abs_logp_err_true",
        "log_ise_mc",
        "ise_mc",
        "ise_mc_se",
        "log_mise_mc",
        "mise_mc",
        "log_miae_mc",
        "miae_mc",
        "tier1_max_abs_log_err",
        "tier1_mean_abs_log_err",
        "tier1_rmse_log_err",
        "tier1_p95_abs_log_err",
        "tier1_bias_log_err",
        "tier1_finite_fraction_flash",
        "tier1_finite_fraction_ref",
    ]


def _append_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _empty_row(columns: list[str]) -> dict[str, Any]:
    return {col: "" for col in columns}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"total": len(rows), "ok": 0, "failed": 0, "skipped": 0}
    failures: dict[str, int] = {}

    metrics_keys = [
        "kl_p_to_phat",
        "nll_hat",
        "max_abs_log_err",
        "rmse_log_err",
        "throughput_qps",
        "mise_mc",
        "miae_mc",
    ]
    values: dict[str, list[float]] = {k: [] for k in metrics_keys}

    for row in rows:
        status = row.get("status", "")
        if status == "ok":
            counts["ok"] += 1
        elif status == "failed":
            counts["failed"] += 1
        else:
            counts["skipped"] += 1

        if status == "failed":
            err = row.get("error_type", "unknown")
            failures[err] = failures.get(err, 0) + 1

        if status != "ok":
            continue

        for key in metrics_keys:
            val = row.get(key)
            if val is None or val == "":
                continue
            try:
                values[key].append(float(val))
            except ValueError:
                continue

    summary = {"counts": counts, "failures": failures, "metrics": {}}
    for key, vals in values.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary["metrics"][key] = {
            "mean": mean,
            "std": var ** 0.5,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return summary


def _collect_summary_plots(plots: dict[str, str], out_dir: Path, summary_names: list[str]) -> None:
    if not plots:
        return
    summary_dir = out_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"plots": {}, "source": plots}
    for name in summary_names:
        base = plots.get(name)
        if not base:
            continue
        base_path = Path(base)
        copied = []
        for suffix in (".pdf", ".png"):
            src = base_path.with_suffix(suffix)
            if not src.exists():
                continue
            dst = summary_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
        if copied:
            manifest["plots"][name] = copied
    if manifest["plots"]:
        write_json(summary_dir / "manifest.json", manifest)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _get_nested(config: dict[str, Any], path: str) -> Any:
    cursor: Any = config
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _apply_override(config: dict[str, Any], path: str, value: Any) -> None:
    if isinstance(value, dict):
        existing = _get_nested(config, path)
        if isinstance(existing, dict):
            merged = _deep_update(existing, value)
            set_nested(config, path, merged)
            return
    set_nested(config, path, value)


def run_sweep(config_path: Path, *, out_dir_override: Path | None = None, fail_fast: bool | None = None) -> Path:
    cfg = load_config(config_path)
    out_dir = out_dir_override or resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    grid_params = cfg.get("sweep", {}).get("parameters", {}) or {}
    grid = expand_grid(grid_params)
    results_path = out_dir / "results.csv"
    columns = _result_columns()
    rows: list[dict[str, Any]] = []

    failures_log = out_dir / "failures.log"
    failures_log.write_text("", encoding="utf-8")

    base_seed = int(cfg.get("suite", {}).get("seed", 0))
    repeats = int(cfg.get("experiment", {}).get("repeats", 1))
    do_fail_fast = bool(cfg.get("sweep", {}).get("fail_fast", False)) if fail_fast is None else fail_fast

    first_meta: dict[str, Any] | None = None

    for grid_idx, overrides in enumerate(grid):
        for repeat in range(repeats):
            seed = base_seed + repeat
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg["suite"]["seed"] = seed
            for path, value in overrides.items():
                _apply_override(trial_cfg, path, value)

            try:
                result, meta = run_one(trial_cfg)
                rows.append(result)
                _append_row(results_path, result, columns)
                if first_meta is None:
                    first_meta = meta
                trial_dir = out_dir / f"trial_{grid_idx:04d}_rep_{repeat:02d}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                write_json(trial_dir / "metrics.json", {"results": result, "meta": meta})
            except Exception as exc:
                row = _empty_row(columns)
                row.update(
                    {
                        "status": "failed",
                        "error_type": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "exception",
                        "error_msg": str(exc),
                        "seed": seed,
                        "experiment_kind": trial_cfg.get("experiment", {}).get("kind"),
                        "dataset": trial_cfg.get("data", {}).get("dataset"),
                        "n_train": trial_cfg.get("data", {}).get("n_train"),
                        "n_test": trial_cfg.get("data", {}).get("n_test"),
                        "dim": trial_cfg.get("data", {}).get("dim"),
                        "kernel": trial_cfg.get("kde", {}).get("kernel"),
                        "output": trial_cfg.get("kde", {}).get("output"),
                        "bandwidth_mode": trial_cfg.get("kde", {}).get("bandwidth", {}).get("mode"),
                        "bandwidth_rule": trial_cfg.get("kde", {}).get("bandwidth", {}).get("rule"),
                        "bandwidth_scale": trial_cfg.get("kde", {}).get("bandwidth", {}).get("scale"),
                        "bandwidth_value": trial_cfg.get("kde", {}).get("bandwidth", {}).get("fixed_h"),
                        "compute_dtype": trial_cfg.get("flash_impl", {}).get("params", {}).get("compute_dtype"),
                        "accumulate_dtype": trial_cfg.get("flash_impl", {}).get("params", {}).get("accumulate_dtype"),
                        "block_q": trial_cfg.get("flash_impl", {}).get("params", {}).get("block_q"),
                        "block_n": trial_cfg.get("flash_impl", {}).get("params", {}).get("block_n"),
                        "num_warps": trial_cfg.get("flash_impl", {}).get("params", {}).get("num_warps"),
                        "num_stages": trial_cfg.get("flash_impl", {}).get("params", {}).get("num_stages"),
                        "precision_mode": trial_cfg.get("flash_impl", {}).get("precision_mode"),
                        "kde_backend": trial_cfg.get("flash_impl", {}).get("kde_backend"),
                        "enable_tf32": trial_cfg.get("suite", {}).get("enable_tf32"),
                    }
                )
                rows.append(row)
                _append_row(results_path, row, columns)
                with failures_log.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + "=" * 80 + "\n")
                    handle.write(f"trial {grid_idx} repeat {repeat}\n")
                    handle.write(traceback.format_exc())
                if do_fail_fast:
                    raise

    summary = _summarize(rows)
    write_json(out_dir / "summary.json", summary)

    plots, frontier = make_plots(results_path, out_dir / "plots")
    if frontier:
        write_json(out_dir / "pareto_frontier.json", frontier)
    if plots:
        plot_manifest = {
            "script": "experiments/error_suite_16d/plotting.py",
            "results_csv": str(results_path),
            "config": str(config_path),
            "repo": get_repo_state(),
            "plots": plots,
        }
        write_json(out_dir / "plots" / "manifest.json", plot_manifest)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        acc_thresholds = cfg.get("pareto", {}).get("accuracy_thresholds", [0.01, 0.02, 0.05, 0.1])
        speed_thresholds = cfg.get("pareto", {}).get("speed_thresholds", [])
        summary["best_under_accuracy"] = best_under_accuracy(
            ok_rows,
            accuracy_thresholds=acc_thresholds,
            error_key="kl_p_to_phat",
            speed_key="throughput_qps",
        )
        if speed_thresholds:
            summary["best_under_speed"] = best_under_speed(
                ok_rows,
                speed_thresholds=speed_thresholds,
                error_key="kl_p_to_phat",
                speed_key="throughput_qps",
            )
        write_json(out_dir / "summary.json", summary)

    failure_snippet = None
    if failures_log.exists():
        failure_text = failures_log.read_text(encoding="utf-8").strip()
        if failure_text:
            failure_snippet = failure_text[:2000]
    report = make_report(
        meta=first_meta,
        summary=summary,
        ok_rows=ok_rows,
        frontier=frontier,
        failure_snippet=failure_snippet,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    summary_names = cfg.get(
        "summary_plots",
        ["oracle_mise_miae_vs_n_16d", "pareto_kl_vs_throughput", "pareto_nll_vs_throughput"],
    )
    if isinstance(summary_names, list):
        _collect_summary_plots(plots, out_dir, summary_names)

    latex_tables = make_latex_tables(ok_rows=ok_rows, frontier=frontier)
    if latex_tables:
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        combined: list[str] = []
        for name, content in latex_tables.items():
            (tables_dir / f"{name}.tex").write_text(content, encoding="utf-8")
            combined.append(f"% {name}\n{content}")
        (tables_dir / "latex_tables.txt").write_text("\n\n".join(combined), encoding="utf-8")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error suite A100 16D sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir_override = Path(args.out_dir) if args.out_dir else None

    try:
        out_dir = run_sweep(config_path, out_dir_override=out_dir_override, fail_fast=args.fail_fast)
        print(f"Sweep outputs written to {out_dir}")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

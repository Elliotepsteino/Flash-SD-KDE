from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import replace
from pathlib import Path
from typing import Any
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from flash_sd_kde.utils import ensure_repo
from globals import FILE_STORAGE_ROOT
from plots.plot_error_suite_16d_config import ErrorSuitePlotConfig


NUMERIC_INT_KEYS = {"seed", "n_samples", "n_queries", "dim"}


def _resolve_results_dir(config: ErrorSuitePlotConfig) -> Path:
    repo_root = Path(ensure_repo())
    if config.results_dir is not None:
        return repo_root / config.results_dir
    base = repo_root / FILE_STORAGE_ROOT / "error_suite_16d"
    if not base.exists():
        raise FileNotFoundError(f"no error suite outputs found under {base}")
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no error suite runs found under {base}")
    return runs[0]


def _coerce_value(key: str, value: str) -> Any:
    if value == "" or value is None:
        return None
    if key in NUMERIC_INT_KEYS:
        try:
            return int(float(value))
        except ValueError:
            return value
    try:
        return float(value)
    except ValueError:
        return value


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    results_path = results_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found in {results_dir}")
    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({k: _coerce_value(k, v) for k, v in row.items()})
    return rows


def _filter_rows(rows: list[dict[str, Any]], config: ErrorSuitePlotConfig) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if config.status_filter and row.get("status") != config.status_filter:
            continue
        if config.output_filter and row.get("output") != config.output_filter:
            continue
        if config.experiment_kind and row.get("experiment_kind") != config.experiment_kind:
            continue
        filtered.append(row)
    return filtered


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key), []).append(row)
    return grouped


def _aggregate_by_x(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[Any], list[float], list[float]]:
    buckets: dict[Any, list[float]] = {}
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        buckets.setdefault(x, []).append(float(y))
    xs = sorted(buckets.keys())
    means = [float(np.mean(buckets[x])) for x in xs]
    stds = [float(np.std(buckets[x])) for x in xs]
    return xs, means, stds


def _setup_style(config: ErrorSuitePlotConfig) -> None:
    plt.rcParams.update(
        {
            "font.size": config.font_size,
            "axes.titlesize": config.font_size + 1,
            "axes.labelsize": config.font_size,
            "legend.fontsize": config.font_size - 1,
            "xtick.labelsize": config.font_size - 1,
            "ytick.labelsize": config.font_size - 1,
            "lines.linewidth": 2,
            "lines.markersize": 5,
        }
    )


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _label_for_group(row: dict[str, Any]) -> str:
    compute_dtype = row.get("compute_dtype")
    tf32 = row.get("enable_tf32")
    label = f"{compute_dtype}"
    if tf32 is not None:
        label += " +TF32" if tf32 else " -TF32"
    return label


def plot_error_vs_bandwidth(rows: list[dict[str, Any]], output_dir: Path, config: ErrorSuitePlotConfig) -> Path | None:
    if not rows:
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _label_for_group(row)
        grouped.setdefault(label, []).append(row)

    fig, ax = plt.subplots(figsize=(config.column_width_in, 2.6))
    for label, group in grouped.items():
        xs, means, stds = _aggregate_by_x(group, "bandwidth_scale", config.error_metric)
        if not xs:
            xs, means, stds = _aggregate_by_x(group, "bandwidth", config.error_metric)
        if not xs:
            continue
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label)

    if not ax.lines:
        plt.close(fig)
        return None

    ax.set_xlabel("bandwidth scale")
    ax.set_ylabel(config.error_metric)
    ax.set_title("Log error vs bandwidth scale")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)

    path = output_dir / "fig_error_vs_bandwidth_scale"
    _save_fig(fig, path.with_suffix(".pdf"), config.dpi)
    _save_fig(fig, path.with_suffix(".png"), config.dpi)
    plt.close(fig)
    return path


def plot_error_vs_dtype(rows: list[dict[str, Any]], output_dir: Path, config: ErrorSuitePlotConfig) -> Path | None:
    if not rows:
        return None

    grouped = _group_by(rows, "compute_dtype")
    labels = []
    means = []
    stds = []
    for dtype, group in grouped.items():
        vals = [row.get(config.error_metric) for row in group if row.get(config.error_metric) is not None]
        if not vals:
            continue
        labels.append(str(dtype))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(config.column_width_in, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel(config.error_metric)
    ax.set_title("Log error vs compute dtype")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    path = output_dir / "fig_error_vs_dtype"
    _save_fig(fig, path.with_suffix(".pdf"), config.dpi)
    _save_fig(fig, path.with_suffix(".png"), config.dpi)
    plt.close(fig)
    return path


def plot_speedup_vs_error(rows: list[dict[str, Any]], output_dir: Path, config: ErrorSuitePlotConfig) -> Path | None:
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(config.column_width_in, 2.6))
    grouped = _group_by(rows, "compute_dtype")
    for dtype, group in grouped.items():
        xs = []
        ys = []
        for row in group:
            err = row.get(config.error_metric)
            speed = row.get(config.speed_metric)
            if err is None or speed is None:
                continue
            xs.append(float(err))
            ys.append(float(speed))
        if xs:
            ax.scatter(xs, ys, label=str(dtype), s=30, alpha=0.8)

    if not ax.collections:
        plt.close(fig)
        return None

    ax.set_xlabel(config.error_metric)
    ax.set_ylabel("speedup (ref/flash)")
    ax.set_title("Speedup vs log error")
    ax.set_xscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)

    path = output_dir / "fig_speedup_vs_error"
    _save_fig(fig, path.with_suffix(".pdf"), config.dpi)
    _save_fig(fig, path.with_suffix(".png"), config.dpi)
    plt.close(fig)
    return path


def plot_runtime_vs_dtype(rows: list[dict[str, Any]], output_dir: Path, config: ErrorSuitePlotConfig) -> Path | None:
    if not rows:
        return None

    grouped = _group_by(rows, "compute_dtype")
    labels = []
    means = []
    stds = []
    for dtype, group in grouped.items():
        vals = [row.get(config.time_metric) for row in group if row.get(config.time_metric) is not None]
        if not vals:
            continue
        labels.append(str(dtype))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(config.column_width_in, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("flash time (ms)")
    ax.set_title("Flash runtime vs compute dtype")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    path = output_dir / "fig_runtime_vs_dtype"
    _save_fig(fig, path.with_suffix(".pdf"), config.dpi)
    _save_fig(fig, path.with_suffix(".png"), config.dpi)
    plt.close(fig)
    return path


def _format_pm(mean: float, std: float) -> str:
    return f"{mean:.3g} $\\pm$ {std:.2g}"


def make_table(rows: list[dict[str, Any]], output_dir: Path, config: ErrorSuitePlotConfig) -> Path | None:
    if not rows:
        return None

    grouped = _group_by(rows, "compute_dtype")
    lines = []
    lines.append("% Auto-generated error suite table")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Compute dtype & max|log err| & rmse log err & speedup & flash ms \\")
    lines.append("\\midrule")
    for dtype, group in grouped.items():
        err_vals = [row.get("max_abs_log_err") for row in group if row.get("max_abs_log_err") is not None]
        rmse_vals = [row.get("rmse_log_err") for row in group if row.get("rmse_log_err") is not None]
        speed_vals = [row.get("speedup") for row in group if row.get("speedup") is not None]
        time_vals = [row.get("time_flash_ms") for row in group if row.get("time_flash_ms") is not None]
        if not err_vals:
            continue
        err_mean, err_std = float(np.mean(err_vals)), float(np.std(err_vals))
        rmse_mean, rmse_std = float(np.mean(rmse_vals)) if rmse_vals else float("nan"), float(np.std(rmse_vals)) if rmse_vals else float("nan")
        speed_mean = float(np.mean(speed_vals)) if speed_vals else float("nan")
        time_mean = float(np.mean(time_vals)) if time_vals else float("nan")
        lines.append(
            f"{dtype} & {_format_pm(err_mean, err_std)} & {_format_pm(rmse_mean, rmse_std)} & {speed_mean:.2f} & {time_mean:.1f} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    path = output_dir / config.table_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    config = ErrorSuitePlotConfig()
    results_override = os.getenv("ERROR_SUITE_RESULTS_DIR")
    if results_override:
        config = replace(config, results_dir=results_override)
    _setup_style(config)

    results_dir = _resolve_results_dir(config)
    rows = _filter_rows(_load_rows(results_dir), config)

    output_override = os.getenv("ERROR_SUITE_OUTPUT_DIR")
    if output_override:
        output_dir = Path(output_override)
    else:
        output_dir = results_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_error_vs_bandwidth(rows, output_dir, config)
    plot_error_vs_dtype(rows, output_dir, config)
    plot_speedup_vs_error(rows, output_dir, config)
    plot_runtime_vs_dtype(rows, output_dir, config)
    if config.make_table:
        make_table(rows, output_dir, config)

    print(f"Error suite plots saved to {output_dir}")


if __name__ == "__main__":
    main()

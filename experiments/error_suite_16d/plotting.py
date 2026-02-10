from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from pathlib import Path
from typing import Any
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter

from experiments.error_suite_16d.pareto import pareto_frontier

try:
    from plots.plot_error_suite_16d_config import ErrorSuitePlotConfig
except Exception:  # pragma: no cover - optional formatting dependency
    ErrorSuitePlotConfig = None  # type: ignore[misc,assignment]

_DEFAULT_PLOT_CONFIG = ErrorSuitePlotConfig() if ErrorSuitePlotConfig is not None else None


NUMERIC_INT_KEYS = {"seed", "n_train", "n_test", "dim"}
METHOD_LABELS = {
    "kde": "KDE",
    "flash_laplace": "Flash-Laplace-KDE",
    "nonfused_laplace": "Laplace-corrected KDE (non-fused)",
    "emp_sd_kde": "Flash-SD-KDE",
}
METHOD_ORDER = ["kde", "flash_laplace", "nonfused_laplace", "emp_sd_kde"]
METHOD_COLORS = {
    "kde": "#1f77b4",
    "flash_laplace": "#ff7f0e",
    "nonfused_laplace": "#d62728",
    "emp_sd_kde": "#2ca02c",
}
METHOD_STYLES = {
    "kde": dict(linestyle="-", marker="o"),
    "flash_laplace": dict(linestyle="--", marker="o"),
    "nonfused_laplace": dict(linestyle=":", marker="o"),
    "emp_sd_kde": dict(linestyle="--", marker="o"),
}


def _coerce_value(key: str, value: Any) -> Any:
    if value == "" or value is None:
        return None
    if isinstance(value, list):
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


def load_rows(results_path: Path) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, restkey="__extra__", restval=None)
        rows = []
        for row in reader:
            if "__extra__" in row:
                row.pop("__extra__", None)
            rows.append({k: _coerce_value(k, v) for k, v in row.items()})
        return rows


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key), []).append(row)
    return grouped


def _aggregate_by_x(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    filter_nonpositive: bool = False,
) -> tuple[list[Any], list[float], list[float], list[int]]:
    buckets: dict[Any, list[float]] = {}
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        y_val = float(y)
        if filter_nonpositive and y_val <= 0:
            continue
        buckets.setdefault(x, []).append(y_val)
    xs = sorted(buckets.keys())
    means = [float(np.mean(buckets[x])) for x in xs]
    stds = [float(np.std(buckets[x])) for x in xs]
    counts = [len(buckets[x]) for x in xs]
    return xs, means, stds, counts


def _n_label(counts: list[int]) -> str:
    if not counts:
        return "n=0"
    if all(c == counts[0] for c in counts):
        return f"n={counts[0]}"
    return "n varies"


def _metric_label(metric: str) -> str:
    if metric == "kl_p_to_phat":
        return "KL(p || p̂)"
    if metric == "nll_hat":
        return "NLL"
    if metric == "ise_mc":
        return "ISE (MC)"
    if metric == "max_abs_log_err":
        return "max |Δ log p|"
    if metric == "rmse_log_err":
        return "RMSE log p"
    return metric


def _aggregate_by_method(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    filter_nonpositive: bool = False,
) -> dict[str, tuple[list[Any], list[float], list[float], list[int]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method")
        if method is None:
            continue
        grouped.setdefault(str(method), []).append(row)

    outputs: dict[str, tuple[list[Any], list[float], list[float], list[int]]] = {}
    for method, method_rows in grouped.items():
        xs, means, stds, counts = _aggregate_by_x(
            method_rows,
            x_key,
            y_key,
            filter_nonpositive=filter_nonpositive,
        )
        if xs:
            outputs[method] = (xs, means, stds, counts)
    return outputs


def _setup_style() -> None:
    if _DEFAULT_PLOT_CONFIG is not None:
        font_size = _DEFAULT_PLOT_CONFIG.font_size
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size + 1,
                "axes.labelsize": font_size,
                "legend.fontsize": font_size - 1,
                "xtick.labelsize": font_size - 1,
                "ytick.labelsize": font_size - 1,
                "lines.linewidth": 2,
                "lines.markersize": 5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        return
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(fig, "_skip_tight_layout", False):
        fig.tight_layout(pad=0.2)
    dpi = _DEFAULT_PLOT_CONFIG.dpi if _DEFAULT_PLOT_CONFIG is not None else 300
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_kl_vs_n(rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    xs, means, stds, counts = _aggregate_by_x(rows, "n_train", "kl_p_to_phat")
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linestyle="-")
    ax.set_xlabel("n_train")
    ax.set_ylabel(_metric_label("kl_p_to_phat"))
    ax.set_title(f"KL vs n_train (mean ± 1 std; {_n_label(counts)})")
    ax.set_xscale("log")
    if all(val > 0 for val in means):
        ax.set_yscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    path = output_dir / "kl_vs_n"
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path


def plot_nll_vs_n(rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    xs, means, stds, counts = _aggregate_by_x(rows, "n_train", "nll_hat")
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linestyle="-")
    ax.set_xlabel("n_train")
    ax.set_ylabel(_metric_label("nll_hat"))
    ax.set_title(f"NLL vs n_train (mean ± 1 std; {_n_label(counts)})")
    ax.set_xscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    path = output_dir / "nll_vs_n"
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path


def plot_bandwidth_curve(rows: list[dict[str, Any]], output_dir: Path, metric: str, name: str) -> Path | None:
    xs, means, stds, counts = _aggregate_by_x(rows, "bandwidth_scale", metric)
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linestyle="-")
    ax.set_xlabel("bandwidth scale")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"{_metric_label(metric)} vs bandwidth scale (mean ± 1 std; {_n_label(counts)})")
    ax.set_xscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    path = output_dir / name
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path


def _fmt_val(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def _frontier_label(point: dict[str, Any]) -> str:
    dtype = point.get("compute_dtype", "?")
    tf32 = point.get("enable_tf32")
    tf32_str = "tf32" if tf32 else "no_tf32"
    bw = _fmt_val(point.get("bandwidth_scale"))
    bq = _fmt_val(point.get("block_q"))
    bn = _fmt_val(point.get("block_n"))
    nw = _fmt_val(point.get("num_warps"))
    ns = _fmt_val(point.get("num_stages"))
    return f"{dtype} {tf32_str} h{bw}\nq{bq} n{bn} w{nw} s{ns}"


def plot_pareto_kl_vs_throughput(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    points = [r for r in rows if r.get("throughput_qps") is not None and r.get("kl_p_to_phat") is not None]
    if not points:
        return None, []

    frontier = pareto_frontier(points, x_key="throughput_qps", y_key="kl_p_to_phat", minimize_y=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.scatter(
        [p["throughput_qps"] for p in points],
        [p["kl_p_to_phat"] for p in points],
        s=24,
        alpha=0.6,
        label="all",
    )
    ax.scatter(
        [p["throughput_qps"] for p in frontier],
        [p["kl_p_to_phat"] for p in frontier],
        s=40,
        alpha=0.9,
        label="pareto",
    )
    ax.set_xlabel("throughput (q/s)")
    ax.set_ylabel("KL(p || p̂)")
    ax.set_title("Pareto: KL vs throughput")
    if all(float(p["kl_p_to_phat"]) > 0 for p in points):
        ax.set_yscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)

    for point in frontier[:5]:
        ax.annotate(_frontier_label(point), (point["throughput_qps"], point["kl_p_to_phat"]), fontsize=7)

    path = output_dir / "pareto_kl_vs_throughput"
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path, frontier


def plot_pareto_nll_vs_throughput(rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    points = [r for r in rows if r.get("throughput_qps") is not None and r.get("nll_hat") is not None]
    if not points:
        return None

    frontier = pareto_frontier(points, x_key="throughput_qps", y_key="nll_hat", minimize_y=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.scatter(
        [p["throughput_qps"] for p in points],
        [p["nll_hat"] for p in points],
        s=24,
        alpha=0.6,
        label="all",
    )
    ax.scatter(
        [p["throughput_qps"] for p in frontier],
        [p["nll_hat"] for p in frontier],
        s=40,
        alpha=0.9,
        label="pareto",
    )
    ax.set_xlabel("throughput (q/s)")
    ax.set_ylabel("NLL")
    ax.set_title("Pareto: NLL vs throughput")
    if all(float(p["nll_hat"]) > 0 for p in points):
        ax.set_yscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)

    path = output_dir / "pareto_nll_vs_throughput"
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path


def plot_ise_vs_n(rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    xs, means, stds, counts = _aggregate_by_x(rows, "n_train", "ise_mc", filter_nonpositive=True)
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linestyle="-")
    ax.set_xlabel("n_train")
    ax.set_ylabel(_metric_label("ise_mc"))
    ax.set_title(f"ISE vs n_train (mean ± 1 std; {_n_label(counts)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, linestyle="--")
    path = output_dir / "ise_vs_n"
    _save(fig, path.with_suffix(".pdf"))
    _save(fig, path.with_suffix(".png"))
    plt.close(fig)
    return path


def make_plots(results_path: Path, output_dir: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    _setup_style()
    rows = [r for r in load_rows(results_path) if r.get("status") == "ok"]
    plots: dict[str, str] = {}

    plot = plot_kl_vs_n(rows, output_dir)
    if plot:
        plots["kl_vs_n"] = str(plot)

    plot = plot_nll_vs_n(rows, output_dir)
    if plot:
        plots["nll_vs_n"] = str(plot)

    plot = plot_bandwidth_curve(rows, output_dir, "kl_p_to_phat", "bandwidth_scale_vs_kl")
    if plot:
        plots["bandwidth_scale_vs_kl"] = str(plot)

    plot = plot_bandwidth_curve(rows, output_dir, "nll_hat", "bandwidth_scale_vs_nll")
    if plot:
        plots["bandwidth_scale_vs_nll"] = str(plot)

    plot = plot_ise_vs_n(rows, output_dir)
    if plot:
        plots["ise_vs_n"] = str(plot)

    pareto_path, frontier = plot_pareto_kl_vs_throughput(rows, output_dir)
    if pareto_path:
        plots["pareto_kl_vs_throughput"] = str(pareto_path)

    pareto_nll = plot_pareto_nll_vs_throughput(rows, output_dir)
    if pareto_nll:
        plots["pareto_nll_vs_throughput"] = str(pareto_nll)

    oracle_plot = plot_oracle_mise_miae_vs_n(rows, output_dir)
    if oracle_plot:
        plots["fig_oracle_error_vs_n_16d"] = str(oracle_plot)

    return plots, frontier


def plot_oracle_mise_miae_vs_n(rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    oracle_rows = [r for r in rows if r.get("experiment_kind") == "oracle_error_16d"]
    if not oracle_rows:
        return None

    mise_by_method = _aggregate_by_method(oracle_rows, "n_train", "mise_mc", filter_nonpositive=True)
    miae_by_method = _aggregate_by_method(oracle_rows, "n_train", "miae_mc", filter_nonpositive=True)
    if not mise_by_method and not miae_by_method:
        return None

    wide_width = _DEFAULT_PLOT_CONFIG.wide_width_in if _DEFAULT_PLOT_CONFIG is not None else 6.8
    fig, axes = plt.subplots(1, 2, figsize=(wide_width, 2.6), sharex=True)
    legend_handles = {}
    band_scale = 1.0
    band_alpha = 0.06
    for method in METHOD_ORDER:
        if method not in mise_by_method or method not in miae_by_method:
            continue
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method)
        style = METHOD_STYLES.get(method, {})

        xs, means, stds, counts = mise_by_method[method]
        counts_arr = np.maximum(1.0, np.asarray(counts, dtype=float))
        errs = np.asarray(stds) / np.sqrt(counts_arr)
        line = axes[0].plot(xs, means, color=color, label=label, **style)[0]
        axes[0].fill_between(
            xs,
            np.maximum(1e-30, np.array(means) - band_scale * errs),
            np.array(means) + band_scale * errs,
            color=color,
            alpha=band_alpha,
        )
        legend_handles.setdefault(label, line)

        xs, means, stds, counts = miae_by_method[method]
        counts_arr = np.maximum(1.0, np.asarray(counts, dtype=float))
        errs = np.asarray(stds) / np.sqrt(counts_arr)
        line = axes[1].plot(xs, means, color=color, label=label, **style)[0]
        axes[1].fill_between(
            xs,
            np.maximum(1e-30, np.array(means) - band_scale * errs),
            np.array(means) + band_scale * errs,
            color=color,
            alpha=band_alpha,
        )
        legend_handles.setdefault(label, line)

    for ax, metric, title in zip(
        axes,
        ["MISE", "MIAE"],
        [
            r"Oracle error vs $n_{\mathrm{train}}$:" "\n" "Mean Integrated Squared Error",
            r"Oracle error vs $n_{\mathrm{train}}$:" "\n" "Mean Integrated Absolute Error",
        ],
    ):
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel(metric)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_title(title)
        tick_size = 7 if _DEFAULT_PLOT_CONFIG is None else max(6, _DEFAULT_PLOT_CONFIG.font_size - 2)
        ax.tick_params(axis="y", labelsize=tick_size)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        frameon=False,
        ncol=2,
        fontsize=7 if _DEFAULT_PLOT_CONFIG is None else max(6, _DEFAULT_PLOT_CONFIG.font_size - 2),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.subplots_adjust(bottom=0.38, top=0.84, wspace=0.3)

    path = output_dir / "fig_oracle_error_vs_n_16d"
    _save(fig, path.with_suffix(".pdf"))
    plt.close(fig)
    return path

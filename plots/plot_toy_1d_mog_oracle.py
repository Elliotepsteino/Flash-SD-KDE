from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import replace
from pathlib import Path
import os
import shutil
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from flash_sd_kde.utils import ensure_repo, read_json
from globals import FILE_STORAGE_ROOT
from plots.plot_toy_1d_mog_oracle_config import Toy1dMoGOraclePlotConfig

_METHODS = (
    ("kde", "KDE", "C0", "-"),
    ("linearized", "Flash-Laplace-KDE", "C1", "--"),
    ("linearized_nonfused", "Laplace-corrected KDE (non-fused)", "C3", ":"),
    ("emp_sd_kde", "Flash-SD-KDE", "C2", "-."),
)

_POS_EPS = 1e-12


def _resolve_results_dir(config: Toy1dMoGOraclePlotConfig) -> Path:
    repo_root = Path(ensure_repo())
    if config.results_dir is not None:
        return repo_root / config.results_dir
    base = repo_root / FILE_STORAGE_ROOT / "benchmarks" / "toy_1d_mog_oracle"
    if not base.exists():
        raise FileNotFoundError(f"no benchmark outputs found under {base}")
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no benchmark runs found under {base}")
    return runs[0]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _extract_metric(results: Dict, method: str, metric: str, n_train: list[int]) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    errs = []
    for n in n_train:
        entry = results["metrics"][method][str(n)]
        vals.append(entry[f"{metric}_mean"])
        errs.append(entry[f"{metric}_std"])
    return np.asarray(vals, dtype=float), np.asarray(errs, dtype=float)


def _extract_runtime(results: Dict, method: str, n_train: list[int]) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    errs = []
    for n in n_train:
        entry = results["runtime_sec"][method][str(n)]
        vals.append(entry["mean"])
        errs.append(entry["std"])
    return np.asarray(vals, dtype=float), np.asarray(errs, dtype=float)


def plot_error_vs_n(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = sorted(int(n) for n in results["n_train_list"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for method, label, color, linestyle in _METHODS:
        ise, ise_err = _extract_metric(results, method, "ise", n_train)
        iae, iae_err = _extract_metric(results, method, "iae", n_train)
        axes[0].plot(
            n_train, ise, marker="o", label=label, color=color, lw=2, linestyle=linestyle
        )
        axes[1].plot(
            n_train, iae, marker="o", label=label, color=color, lw=2, linestyle=linestyle
        )

        ise_lo = np.maximum(ise - ise_err, _POS_EPS)
        ise_hi = np.maximum(ise + ise_err, _POS_EPS)
        iae_lo = np.maximum(iae - iae_err, _POS_EPS)
        iae_hi = np.maximum(iae + iae_err, _POS_EPS)
        axes[0].fill_between(n_train, ise_lo, ise_hi, color=color, alpha=0.15)
        axes[1].fill_between(n_train, iae_lo, iae_hi, color=color, alpha=0.15)

    for ax, title, ylabel in [
        (axes[0], "MISE", "MISE"),
        (axes[1], "MIAE", "MIAE"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{train}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save_fig(fig, output_dir / "fig_oracle_error_vs_n_1d.pdf", dpi)
    if os.getenv("TOY_1D_DISABLE_PAPER_COPY") != "1":
        paper_fig = Path(ensure_repo()) / "paper" / "figures" / "fig_oracle_error_vs_n_1d.pdf"
        paper_fig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / "fig_oracle_error_vs_n_1d.pdf", paper_fig)
    plt.close(fig)


def plot_runtime_vs_n(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = sorted(int(n) for n in results["n_train_list"])

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for method, label, color, linestyle in _METHODS:
        runtime, runtime_err = _extract_runtime(results, method, n_train)
        ax.plot(
            n_train, runtime, marker="o", label=label, color=color, lw=2, linestyle=linestyle
        )
        rt_lo = np.maximum(runtime - runtime_err, _POS_EPS)
        rt_hi = np.maximum(runtime + runtime_err, _POS_EPS)
        ax.fill_between(n_train, rt_lo, rt_hi, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{train}$")
    ax.set_ylabel("runtime (s)")
    ax.set_title(r"Runtime vs $n_{train}$ (total)")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)

    fig.tight_layout()
    _save_fig(fig, output_dir / "fig_runtime_vs_n.pdf", dpi)
    _save_fig(fig, output_dir / "fig_runtime_vs_n.png", dpi)
    plt.close(fig)


def plot_density_curves(output_dir: Path, densities: Dict, *, dpi: int) -> None:
    x_grid = densities["x_grid"]
    true_density = densities["true_density"]
    kde_density = densities["kde_density"]
    lin_density = densities["linearized_density"]
    lin_nf_density = densities["linearized_nonfused_density"]
    emp_density = densities["emp_sd_kde_density"]
    n_train = int(densities["n_train"][0])

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(x_grid, true_density, color="black", lw=2, label="True density")
    ax.plot(x_grid, kde_density, color="C0", lw=1.8, label="KDE")
    ax.plot(x_grid, lin_density, color="C1", lw=1.8, label="Flash-Laplace-KDE")
    ax.plot(x_grid, lin_nf_density, color="C3", lw=1.8, label="Laplace-corrected KDE (non-fused)")
    ax.plot(x_grid, emp_density, color="C2", lw=1.8, label="Flash-SD-KDE")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(rf"Density estimates at $n_{{train}}$={n_train}")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.25))

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save_fig(fig, output_dir / "fig_density_curves.pdf", dpi)
    _save_fig(fig, output_dir / "fig_density_curves.png", dpi)
    plt.close(fig)


def plot_error_runtime_tradeoff(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = max(int(n) for n in results["n_train_list"])

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for method, label, color, _linestyle in _METHODS:
        metric = results["metrics"][method][str(n_train)]["ise_mean"]
        runtime = results["runtime_sec"][method][str(n_train)]["mean"]
        ax.scatter(runtime, metric, s=60, color=color, label=label)
        ax.text(runtime * 1.05, metric * 1.05, label, fontsize=8)

    ax.set_xlabel("runtime (s)")
    ax.set_ylabel("ISE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(rf"Accuracy-speed tradeoff ($n_{{train}}$={n_train})")
    ax.grid(alpha=0.2, linestyle="--")

    fig.tight_layout()
    _save_fig(fig, output_dir / "fig_error_runtime_tradeoff.pdf", dpi)
    _save_fig(fig, output_dir / "fig_error_runtime_tradeoff.png", dpi)
    plt.close(fig)


def plot_fused_vs_nonfused_runtime(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = sorted(int(n) for n in results["n_train_list"])
    fused_rt, fused_err = _extract_runtime(results, "linearized", n_train)
    nf_rt, nf_err = _extract_runtime(results, "linearized_nonfused", n_train)
    emp_rt, emp_err = _extract_runtime(results, "emp_sd_kde", n_train)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    axes[0].plot(n_train, fused_rt, marker="o", color="C1", label="Flash-Laplace-KDE")
    axes[0].fill_between(
        n_train,
        np.maximum(fused_rt - fused_err, _POS_EPS),
        np.maximum(fused_rt + fused_err, _POS_EPS),
        color="C1",
        alpha=0.15,
    )
    axes[0].plot(n_train, nf_rt, marker="o", color="C3", label="Non-fused Laplace")
    axes[0].fill_between(
        n_train,
        np.maximum(nf_rt - nf_err, _POS_EPS),
        np.maximum(nf_rt + nf_err, _POS_EPS),
        color="C3",
        alpha=0.15,
    )
    speedup = nf_rt / np.maximum(fused_rt, _POS_EPS)
    speedup_emp = emp_rt / np.maximum(fused_rt, _POS_EPS)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$n_{train}$")
    axes[0].set_ylabel("runtime (s)")
    axes[0].set_title("Fused vs non-fused runtime")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].legend().remove()

    axes[1].plot(n_train, speedup, marker="o", color="C2")
    axes[1].plot(n_train, speedup_emp, marker="o", color="C4")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$n_{train}$")
    axes[1].set_ylabel("runtime ratio")
    axes[1].set_title("Speedup from fusion")
    axes[1].grid(alpha=0.2, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += axes[1].get_lines()
    labels += ["Non-fused / Flash-Laplace-KDE", "Flash-SD-KDE / Flash-Laplace-KDE"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save_fig(fig, output_dir / "fig_fused_vs_nonfused_runtime.pdf", dpi)
    plt.close(fig)


def main() -> None:
    config = Toy1dMoGOraclePlotConfig()
    results_override = os.getenv("TOY_1D_RESULTS_DIR")
    if results_override:
        config = replace(config, results_dir=results_override)
    results_dir = _resolve_results_dir(config)
    output_override = os.getenv("TOY_1D_OUTPUT_DIR")
    if output_override:
        output_dir = Path(output_override)
    else:
        output_dir = results_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = read_json(results_dir / "results.json")
    densities_path = results_dir / "densities_curves.npz"
    densities = None
    if densities_path.exists():
        densities = np.load(densities_path)

    _setup_style()
    plot_error_vs_n(output_dir, results, dpi=config.dpi)
    plot_runtime_vs_n(output_dir, results, dpi=config.dpi)
    if densities is not None:
        plot_density_curves(output_dir, densities, dpi=config.dpi)
    plot_error_runtime_tradeoff(output_dir, results, dpi=config.dpi)
    plot_fused_vs_nonfused_runtime(output_dir, results, dpi=config.dpi)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

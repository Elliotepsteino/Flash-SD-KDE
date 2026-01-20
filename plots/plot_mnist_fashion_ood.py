from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from flash_sd_kde.utils import ensure_repo, read_json
from globals import DEFAULT_EPS, FILE_STORAGE_ROOT
from plots.plot_mnist_fashion_ood_config import MnistFashionOodPlotConfig


def _resolve_results_dir(config: MnistFashionOodPlotConfig) -> Path:
    repo_root = Path(ensure_repo())
    if config.results_dir is not None:
        return repo_root / config.results_dir
    base = repo_root / FILE_STORAGE_ROOT / "benchmarks" / "mnist_fashion_pca16_ood"
    if not base.exists():
        raise FileNotFoundError(f"no benchmark outputs found under {base}")
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no benchmark runs found under {base}")
    return runs[0]


def _load_results(results_dir: Path) -> Tuple[Dict, Dict]:
    results = read_json(results_dir / "results.json")
    densities = np.load(results_dir / "densities_curves.npz")
    return results, densities


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


def plot_roc_curves(output_dir: Path, densities: Dict, *, dpi: int) -> None:
    id_kde = densities["kde_id"]
    ood_kde = densities["kde_ood"]
    id_emp = densities["emp_id"]
    ood_emp = densities["emp_ood"]

    labels = np.concatenate([np.ones_like(id_kde), np.zeros_like(ood_kde)])
    scores_kde = np.concatenate([id_kde, ood_kde])
    scores_emp = np.concatenate([id_emp, ood_emp])

    fpr_kde, tpr_kde, _ = roc_curve(labels, scores_kde)
    fpr_emp, tpr_emp, _ = roc_curve(labels, scores_emp)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr_kde, tpr_kde, label="KDE", lw=2)
    ax.plot(fpr_emp, tpr_emp, label="Emp-SD-KDE", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD ROC (MNIST vs Fashion-MNIST)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle="--")
    _save_fig(fig, output_dir / "fig_roc_curves.pdf", dpi)
    _save_fig(fig, output_dir / "fig_roc_curves.png", dpi)
    plt.close(fig)


def plot_log_density_hist(output_dir: Path, densities: Dict, *, dpi: int, bins: int) -> None:
    kde_id = np.log(densities["kde_id"] + DEFAULT_EPS)
    kde_ood = np.log(densities["kde_ood"] + DEFAULT_EPS)
    emp_id = np.log(densities["emp_id"] + DEFAULT_EPS)
    emp_ood = np.log(densities["emp_ood"] + DEFAULT_EPS)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, title, id_vals, ood_vals in [
        (axes[0], "KDE", kde_id, kde_ood),
        (axes[1], "Emp-SD-KDE", emp_id, emp_ood),
    ]:
        hist_range = (float(min(id_vals.min(), ood_vals.min())), float(max(id_vals.max(), ood_vals.max())))
        ax.hist(id_vals, bins=bins, range=hist_range, alpha=0.6, label="MNIST ID")
        ax.hist(ood_vals, bins=bins, range=hist_range, alpha=0.6, label="Fashion OOD")
        ax.set_title(title)
        ax.set_xlabel("log density")
        ax.grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False)
    fig.suptitle("Log-density histograms (largest n_train)")
    _save_fig(fig, output_dir / "fig_log_density_hist.pdf", dpi)
    _save_fig(fig, output_dir / "fig_log_density_hist.png", dpi)
    plt.close(fig)


def plot_auc_vs_n_train(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = [int(n) for n in results["n_train_list"]]
    n_train_sorted = sorted(n_train)

    roc_kde = [results["metrics"][str(n)]["kde"]["roc_auc"] for n in n_train_sorted]
    roc_emp = [results["metrics"][str(n)]["emp_sd_kde"]["roc_auc"] for n in n_train_sorted]
    pr_kde = [results["metrics"][str(n)]["kde"]["pr_auc"] for n in n_train_sorted]
    pr_emp = [results["metrics"][str(n)]["emp_sd_kde"]["pr_auc"] for n in n_train_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(n_train_sorted, roc_kde, marker="o", label="KDE", lw=2)
    axes[0].plot(n_train_sorted, roc_emp, marker="o", label="Emp-SD-KDE", lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_train")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("ROC AUC vs n_train")
    axes[0].grid(alpha=0.2, linestyle="--")

    axes[1].plot(n_train_sorted, pr_kde, marker="o", label="KDE", lw=2)
    axes[1].plot(n_train_sorted, pr_emp, marker="o", label="Emp-SD-KDE", lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_train")
    axes[1].set_ylabel("PR AUC")
    axes[1].set_title("PR AUC vs n_train")
    axes[1].grid(alpha=0.2, linestyle="--")

    axes[0].legend(frameon=False)
    _save_fig(fig, output_dir / "fig_auc_vs_n_train.pdf", dpi)
    _save_fig(fig, output_dir / "fig_auc_vs_n_train.png", dpi)
    plt.close(fig)


def plot_runtime_vs_n_train(output_dir: Path, results: Dict, *, dpi: int) -> None:
    n_train = [int(n) for n in results["n_train_list"]]
    n_train_sorted = sorted(n_train)

    kde_eval_id = [results["runtime_sec"][str(n)]["kde"]["eval_id_sec"] for n in n_train_sorted]
    kde_eval_ood = [results["runtime_sec"][str(n)]["kde"]["eval_ood_sec"] for n in n_train_sorted]

    emp_score = [results["runtime_sec"][str(n)]["emp_sd_kde"]["score_sec"] for n in n_train_sorted]
    emp_shift = [results["runtime_sec"][str(n)]["emp_sd_kde"]["shift_sec"] for n in n_train_sorted]
    emp_eval_id = [results["runtime_sec"][str(n)]["emp_sd_kde"]["eval_id_sec"] for n in n_train_sorted]
    emp_eval_ood = [results["runtime_sec"][str(n)]["emp_sd_kde"]["eval_ood_sec"] for n in n_train_sorted]

    x = np.arange(len(n_train_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    ax.bar(x - width / 2, kde_eval_id, width, label="KDE eval ID")
    ax.bar(x - width / 2, kde_eval_ood, width, bottom=kde_eval_id, label="KDE eval OOD")

    bottom_emp = np.zeros_like(x, dtype=float)
    ax.bar(x + width / 2, emp_score, width, label="Emp score")
    bottom_emp += np.array(emp_score)
    ax.bar(x + width / 2, emp_shift, width, bottom=bottom_emp, label="Emp shift")
    bottom_emp += np.array(emp_shift)
    ax.bar(x + width / 2, emp_eval_id, width, bottom=bottom_emp, label="Emp eval ID")
    bottom_emp += np.array(emp_eval_id)
    ax.bar(x + width / 2, emp_eval_ood, width, bottom=bottom_emp, label="Emp eval OOD")

    ax.set_xticks(x, [str(n) for n in n_train_sorted])
    ax.set_xlabel("n_train")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Runtime breakdown vs n_train")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    _save_fig(fig, output_dir / "fig_runtime_vs_n_train.pdf", dpi)
    _save_fig(fig, output_dir / "fig_runtime_vs_n_train.png", dpi)
    plt.close(fig)


def main() -> None:
    config = MnistFashionOodPlotConfig()
    results_dir = _resolve_results_dir(config)
    output_dir = results_dir / config.output_subdir
    results, densities = _load_results(results_dir)
    _setup_style()

    plot_roc_curves(output_dir, densities, dpi=config.dpi)
    plot_log_density_hist(output_dir, densities, dpi=config.dpi, bins=config.hist_bins)
    plot_auc_vs_n_train(output_dir, results, dpi=config.dpi)
    plot_runtime_vs_n_train(output_dir, results, dpi=config.dpi)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

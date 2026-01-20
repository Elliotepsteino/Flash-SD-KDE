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


def _select_density(densities: Dict, backend_name: str, key: str) -> np.ndarray:
    pref_key = f"{backend_name}_{key}"
    if pref_key in densities:
        return densities[pref_key]
    return densities[key]


def _get_backend_metrics(results: Dict, backend_name: str) -> Dict:
    if "backend_variants" in results:
        return results["metrics"][backend_name]
    return results["metrics"]


def _get_backend_runtime(results: Dict, backend_name: str) -> Dict:
    if "backend_variants" in results:
        return results["runtime_sec"][backend_name]
    return results["runtime_sec"]


def _resolve_backend_names(results: Dict, config: MnistFashionOodPlotConfig) -> list[str]:
    if config.compare_backend_names is not None:
        return list(config.compare_backend_names)
    if "backend_variants" in results:
        return list(results["backend_variants"].keys())
    return ["default"]


def _total_kde_runtime(runtime_entry: Dict) -> float:
    return runtime_entry["kde"]["eval_id_sec"] + runtime_entry["kde"]["eval_ood_sec"]


def _total_emp_runtime(runtime_entry: Dict) -> float:
    emp = runtime_entry["emp_sd_kde"]
    return emp["score_sec"] + emp["shift_sec"] + emp["eval_id_sec"] + emp["eval_ood_sec"]


def plot_roc_curves(output_dir: Path, densities: Dict, *, backend_name: str, dpi: int) -> None:
    id_kde = _select_density(densities, backend_name, "kde_id")
    ood_kde = _select_density(densities, backend_name, "kde_ood")
    id_emp = _select_density(densities, backend_name, "emp_id")
    ood_emp = _select_density(densities, backend_name, "emp_ood")

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
    ax.set_title(f"OOD ROC (MNIST vs Fashion-MNIST) [{backend_name}]")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle="--")
    _save_fig(fig, output_dir / "fig_roc_curves.pdf", dpi)
    _save_fig(fig, output_dir / "fig_roc_curves.png", dpi)
    plt.close(fig)


def plot_log_density_hist(
    output_dir: Path, densities: Dict, *, backend_name: str, dpi: int, bins: int
) -> None:
    kde_id = np.log(_select_density(densities, backend_name, "kde_id") + DEFAULT_EPS)
    kde_ood = np.log(_select_density(densities, backend_name, "kde_ood") + DEFAULT_EPS)
    emp_id = np.log(_select_density(densities, backend_name, "emp_id") + DEFAULT_EPS)
    emp_ood = np.log(_select_density(densities, backend_name, "emp_ood") + DEFAULT_EPS)

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
    fig.suptitle(f"Log-density histograms (largest n_train) [{backend_name}]")
    _save_fig(fig, output_dir / "fig_log_density_hist.pdf", dpi)
    _save_fig(fig, output_dir / "fig_log_density_hist.png", dpi)
    plt.close(fig)


def plot_auc_vs_n_train(output_dir: Path, results: Dict, *, backend_name: str, dpi: int) -> None:
    n_train = [int(n) for n in results["n_train_list"]]
    n_train_sorted = sorted(n_train)

    metrics = _get_backend_metrics(results, backend_name)
    roc_kde = [metrics[str(n)]["kde"]["roc_auc"] for n in n_train_sorted]
    roc_emp = [metrics[str(n)]["emp_sd_kde"]["roc_auc"] for n in n_train_sorted]
    pr_kde = [metrics[str(n)]["kde"]["pr_auc"] for n in n_train_sorted]
    pr_emp = [metrics[str(n)]["emp_sd_kde"]["pr_auc"] for n in n_train_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(n_train_sorted, roc_kde, marker="o", label="KDE", lw=2)
    axes[0].plot(n_train_sorted, roc_emp, marker="o", label="Emp-SD-KDE", lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_train")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title(f"ROC AUC vs n_train [{backend_name}]")
    axes[0].grid(alpha=0.2, linestyle="--")

    axes[1].plot(n_train_sorted, pr_kde, marker="o", label="KDE", lw=2)
    axes[1].plot(n_train_sorted, pr_emp, marker="o", label="Emp-SD-KDE", lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_train")
    axes[1].set_ylabel("PR AUC")
    axes[1].set_title(f"PR AUC vs n_train [{backend_name}]")
    axes[1].grid(alpha=0.2, linestyle="--")

    axes[0].legend(frameon=False)
    _save_fig(fig, output_dir / "fig_auc_vs_n_train.pdf", dpi)
    _save_fig(fig, output_dir / "fig_auc_vs_n_train.png", dpi)
    plt.close(fig)


def plot_runtime_vs_n_train(output_dir: Path, results: Dict, *, backend_name: str, dpi: int) -> None:
    n_train = [int(n) for n in results["n_train_list"]]
    n_train_sorted = sorted(n_train)

    runtime = _get_backend_runtime(results, backend_name)
    kde_eval_id = [runtime[str(n)]["kde"]["eval_id_sec"] for n in n_train_sorted]
    kde_eval_ood = [runtime[str(n)]["kde"]["eval_ood_sec"] for n in n_train_sorted]

    emp_score = [runtime[str(n)]["emp_sd_kde"]["score_sec"] for n in n_train_sorted]
    emp_shift = [runtime[str(n)]["emp_sd_kde"]["shift_sec"] for n in n_train_sorted]
    emp_eval_id = [runtime[str(n)]["emp_sd_kde"]["eval_id_sec"] for n in n_train_sorted]
    emp_eval_ood = [runtime[str(n)]["emp_sd_kde"]["eval_ood_sec"] for n in n_train_sorted]

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
    ax.set_title(f"Runtime breakdown vs n_train [{backend_name}]")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    _save_fig(fig, output_dir / "fig_runtime_vs_n_train.pdf", dpi)
    _save_fig(fig, output_dir / "fig_runtime_vs_n_train.png", dpi)
    plt.close(fig)


def plot_backend_comparison(output_dir: Path, results: Dict, backend_names: list[str], *, dpi: int) -> None:
    n_train_max = max(int(n) for n in results["n_train_list"])
    key = str(n_train_max)

    roc_kde = []
    roc_emp = []
    runtime_kde = []
    runtime_emp = []

    for backend_name in backend_names:
        metrics = _get_backend_metrics(results, backend_name)
        runtime = _get_backend_runtime(results, backend_name)
        roc_kde.append(metrics[key]["kde"]["roc_auc"])
        roc_emp.append(metrics[key]["emp_sd_kde"]["roc_auc"])
        runtime_kde.append(_total_kde_runtime(runtime[key]))
        runtime_emp.append(_total_emp_runtime(runtime[key]))

    x = np.arange(len(backend_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    axes[0].bar(x - width / 2, roc_kde, width, label="KDE")
    axes[0].bar(x + width / 2, roc_emp, width, label="Emp-SD-KDE")
    axes[0].set_xticks(x, backend_names, rotation=30, ha="right")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title(f"Backend ROC AUC (n_train={n_train_max})")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, linestyle="--", axis="y")

    axes[1].bar(x - width / 2, runtime_kde, width, label="KDE total")
    axes[1].bar(x + width / 2, runtime_emp, width, label="Emp total")
    axes[1].set_xticks(x, backend_names, rotation=30, ha="right")
    axes[1].set_ylabel("runtime (s)")
    axes[1].set_title(f"Backend runtime (n_train={n_train_max})")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2, linestyle="--", axis="y")

    fig.tight_layout()
    _save_fig(fig, output_dir / "fig_backend_comparison.pdf", dpi)
    _save_fig(fig, output_dir / "fig_backend_comparison.png", dpi)
    plt.close(fig)


def plot_speedup_vs_n_train(
    output_dir: Path,
    results: Dict,
    *,
    baseline_name: str,
    compare_names: list[str],
    dpi: int,
) -> None:
    if baseline_name not in compare_names:
        compare_names = [baseline_name] + compare_names

    n_train_sorted = sorted(int(n) for n in results["n_train_list"])
    runtime_base = _get_backend_runtime(results, baseline_name)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for backend_name in compare_names:
        if backend_name == baseline_name:
            continue
        runtime = _get_backend_runtime(results, backend_name)
        kde_speedup = []
        emp_speedup = []
        for n in n_train_sorted:
            key = str(n)
            base_kde = _total_kde_runtime(runtime_base[key])
            base_emp = _total_emp_runtime(runtime_base[key])
            this_kde = _total_kde_runtime(runtime[key])
            this_emp = _total_emp_runtime(runtime[key])
            kde_speedup.append(base_kde / max(this_kde, 1e-12))
            emp_speedup.append(base_emp / max(this_emp, 1e-12))

        axes[0].plot(n_train_sorted, kde_speedup, marker="o", label=backend_name)
        axes[1].plot(n_train_sorted, emp_speedup, marker="o", label=backend_name)

    for ax, title in zip(axes, ["KDE total speedup", "Emp-SD-KDE total speedup"]):
        ax.set_xscale("log")
        ax.set_xlabel("n_train")
        ax.set_ylabel("speedup vs baseline")
        ax.set_title(f\"{title} (baseline={baseline_name})\")
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, output_dir / "fig_speedup_vs_n_train.pdf", dpi)
    _save_fig(fig, output_dir / "fig_speedup_vs_n_train.png", dpi)
    plt.close(fig)


def main() -> None:
    config = MnistFashionOodPlotConfig()
    results_dir = _resolve_results_dir(config)
    output_dir = results_dir / config.output_subdir
    results, densities = _load_results(results_dir)
    _setup_style()

    backend_names = _resolve_backend_names(results, config)
    primary_backend = config.primary_backend_name
    if primary_backend not in backend_names:
        primary_backend = backend_names[0]

    plot_roc_curves(output_dir, densities, backend_name=primary_backend, dpi=config.dpi)
    plot_log_density_hist(
        output_dir, densities, backend_name=primary_backend, dpi=config.dpi, bins=config.hist_bins
    )
    plot_auc_vs_n_train(output_dir, results, backend_name=primary_backend, dpi=config.dpi)
    plot_runtime_vs_n_train(output_dir, results, backend_name=primary_backend, dpi=config.dpi)
    if len(backend_names) > 1:
        plot_backend_comparison(output_dir, results, backend_names, dpi=config.dpi)
        if config.speedup_baseline_name not in backend_names:
            raise ValueError("speedup_baseline_name not found in backend results.")
        missing = [name for name in config.speedup_backend_names if name not in backend_names]
        if missing:
            raise ValueError(f"speedup_backend_names missing in backend results: {missing}")
        plot_speedup_vs_n_train(
            output_dir,
            results,
            baseline_name=config.speedup_baseline_name,
            compare_names=list(config.speedup_backend_names),
            dpi=config.dpi,
        )

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

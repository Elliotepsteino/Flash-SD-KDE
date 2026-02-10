from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision.utils import make_grid

from flash_sd_kde.utils import ensure_repo
from globals import FILE_STORAGE_ROOT
from plots.save_density_ranked_grids_config import DensityGridConfig


def _resolve_results_dir(config: DensityGridConfig) -> Path:
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


def _load_densities(results_dir: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(results_dir / "densities_curves.npz"))


def _load_test_sets(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    mnist_test = datasets.MNIST(root=str(root), train=False, download=True)
    fashion_test = datasets.FashionMNIST(root=str(root), train=False, download=True)
    mnist = mnist_test.data.float() / 255.0
    fashion = fashion_test.data.float() / 255.0
    return mnist, fashion


def _save_grid(images: torch.Tensor, path: Path, *, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.unsqueeze(1), nrow=5, padding=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _rank_indices(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    top_idx = np.argsort(scores)[-top_k:][::-1].copy()
    low_idx = np.argsort(scores)[:top_k].copy()
    return top_idx, low_idx


def _select_density(densities: Dict[str, np.ndarray], backend_name: str, key: str) -> np.ndarray:
    pref_key = f"{backend_name}_{key}"
    if pref_key in densities:
        return densities[pref_key]
    return densities[key]


def save_grids(config: DensityGridConfig) -> Path:
    results_dir = _resolve_results_dir(config)
    densities = _load_densities(results_dir)
    output_dir = results_dir / config.output_subdir

    data_root = results_dir / "datasets"
    mnist, fashion = _load_test_sets(data_root)

    for method in ["kde", "emp"]:
        id_scores = _select_density(densities, config.backend_name, f"{method}_id")
        ood_scores = _select_density(densities, config.backend_name, f"{method}_ood")

        id_top, id_low = _rank_indices(id_scores, config.top_k)
        ood_top, _ = _rank_indices(ood_scores, config.top_k)

        _save_grid(
            mnist[id_top],
            output_dir / f"mnist_top_{method}.png",
            title=f"MNIST top {config.top_k} ({method}, {config.backend_name})",
            dpi=config.dpi,
        )
        _save_grid(
            mnist[id_low],
            output_dir / f"mnist_low_{method}.png",
            title=f"MNIST low {config.top_k} ({method}, {config.backend_name})",
            dpi=config.dpi,
        )
        _save_grid(
            fashion[ood_top],
            output_dir / f"fashion_top_{method}.png",
            title=f"Fashion top {config.top_k} ({method}, {config.backend_name})",
            dpi=config.dpi,
        )

    return output_dir


def main() -> None:
    config = DensityGridConfig()
    output_dir = save_grids(config)
    print(f"Density grids saved to {output_dir}")


if __name__ == "__main__":
    main()

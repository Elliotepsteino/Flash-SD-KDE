from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass
from typing import Sequence

from globals import (
    DEFAULT_EMP_SCORE_BACKEND,
    DEFAULT_KDE_BACKEND,
    DEFAULT_PRECISION_MODE,
    ND_FEATURES,
)


@dataclass(frozen=True)
class MnistFashionOodConfig:
    seed: int = 0
    device: str = "cuda"
    pca_components: int = ND_FEATURES
    n_train_list: Sequence[int] = (2000, 4000, 8000, 16000, 32000)
    n_val: int = 10_000
    use_val_bandwidth: bool = True
    bandwidth_multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5)
    precision_mode: str = DEFAULT_PRECISION_MODE
    kde_backend: str = DEFAULT_KDE_BACKEND
    emp_score_backend: str | None = None
    use_precomputed_norms: bool = True
    autotune: bool = True
    save_density_arrays: bool = True
    output_tag: str = "benchmarks/mnist_fashion_pca16_ood"

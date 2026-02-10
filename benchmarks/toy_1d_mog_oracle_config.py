from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass
from typing import Sequence

from globals import DEFAULT_KDE_BACKEND, DEFAULT_PRECISION_MODE


@dataclass(frozen=True)
class Toy1dMoGOracleConfig:
    seed: int = 0
    device: str = "cuda"
    n_train_list: Sequence[int] = (128, 256, 512, 1024, 2048, 4096)
    n_repeats: int = 3
    n_grid: int = 2048
    grid_min: float = -6.0
    grid_max: float = 6.0
    bandwidth_multiplier: float = 1.0
    mixture_weights: Sequence[float] = (0.3, 0.4, 0.3)
    mixture_means: Sequence[float] = (-2.0, 0.0, 2.0)
    mixture_stds: Sequence[float] = (0.4, 0.7, 0.3)
    kde_backend: str = DEFAULT_KDE_BACKEND
    precision_mode: str = DEFAULT_PRECISION_MODE
    use_precomputed_norms: bool = True
    autotune: bool = True
    timing_repeats: int = 3
    timing_warmup: int = 1
    emp_chunk_size: int = 1024
    laplace_chunk_size: int = 1024
    save_density_curves: bool = True
    output_tag: str = "benchmarks/toy_1d_mog_oracle"

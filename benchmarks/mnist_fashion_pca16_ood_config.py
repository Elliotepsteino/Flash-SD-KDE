from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass
from typing import Sequence

from globals import (
    BACKEND_FLASH_SPLITK_ORDERED,
    BACKEND_FLASH_SPLITK_SYM,
    BACKEND_FLASH_SPLITK_SYM_LINEARIZED,
    BACKEND_NON_FLASH_ATOMIC_SYM,
    DEFAULT_EMP_SCORE_BACKEND,
    DEFAULT_KDE_BACKEND,
    DEFAULT_PRECISION_MODE,
    EMP_SD_KDE_VARIANT_EXACT,
    EMP_SD_KDE_VARIANT_LINEARIZED,
    EMP_SCORE_BACKEND_ORDERED_SPLITK,
    KDE_BACKEND_ATOMIC,
    ND_FEATURES,
)


@dataclass(frozen=True)
class BackendVariant:
    name: str
    kde_backend: str
    emp_score_backend: str
    emp_sd_kde_variant: str = EMP_SD_KDE_VARIANT_EXACT
    precision_mode: str = DEFAULT_PRECISION_MODE
    use_precomputed_norms: bool = True
    autotune: bool = True


@dataclass(frozen=True)
class MnistFashionOodConfig:
    seed: int = 0
    device: str = "cuda"
    pca_components: int = ND_FEATURES
    n_train_list: Sequence[int] = (2000, 4000, 8000, 16000, 32000)
    n_val: int = 10_000
    use_val_bandwidth: bool = True
    bandwidth_multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5)
    backend_variants: Sequence[BackendVariant] = (
        BackendVariant(
            name=BACKEND_FLASH_SPLITK_SYM,
            kde_backend=DEFAULT_KDE_BACKEND,
            emp_score_backend=DEFAULT_EMP_SCORE_BACKEND,
        ),
        BackendVariant(
            name=BACKEND_FLASH_SPLITK_SYM_LINEARIZED,
            kde_backend=DEFAULT_KDE_BACKEND,
            emp_score_backend=DEFAULT_EMP_SCORE_BACKEND,
            emp_sd_kde_variant=EMP_SD_KDE_VARIANT_LINEARIZED,
        ),
        BackendVariant(
            name=BACKEND_FLASH_SPLITK_ORDERED,
            kde_backend=DEFAULT_KDE_BACKEND,
            emp_score_backend=EMP_SCORE_BACKEND_ORDERED_SPLITK,
        ),
        BackendVariant(
            name=BACKEND_NON_FLASH_ATOMIC_SYM,
            kde_backend=KDE_BACKEND_ATOMIC,
            emp_score_backend=DEFAULT_EMP_SCORE_BACKEND,
        ),
    )
    primary_backend_name: str = BACKEND_FLASH_SPLITK_SYM
    bandwidth_backend_name: str = BACKEND_FLASH_SPLITK_SYM
    save_density_arrays: bool = True
    output_tag: str = "benchmarks/mnist_fashion_pca16_ood"

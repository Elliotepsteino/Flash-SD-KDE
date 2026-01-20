from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass
from typing import Sequence

from globals import (
    BACKEND_FLASH_SPLITK_ORDERED,
    BACKEND_FLASH_SPLITK_SYM,
    BACKEND_FLASH_SPLITK_SYM_LINEARIZED,
    BACKEND_NON_FLASH_ATOMIC_SYM,
)


@dataclass(frozen=True)
class MnistFashionOodPlotConfig:
    results_dir: str | None = None
    output_subdir: str = "figures"
    dpi: int = 300
    hist_bins: int = 60
    primary_backend_name: str = BACKEND_FLASH_SPLITK_SYM
    compare_backend_names: Sequence[str] | None = None
    speedup_baseline_name: str = BACKEND_NON_FLASH_ATOMIC_SYM
    speedup_backend_names: Sequence[str] = (
        BACKEND_FLASH_SPLITK_SYM,
        BACKEND_FLASH_SPLITK_ORDERED,
        BACKEND_FLASH_SPLITK_SYM_LINEARIZED,
    )

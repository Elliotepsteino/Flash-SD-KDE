from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass

from globals import BACKEND_FLASH_SPLITK_SYM


@dataclass(frozen=True)
class DensityGridConfig:
    results_dir: str | None = None
    output_subdir: str = "grids"
    top_k: int = 25
    dpi: int = 200
    backend_name: str = BACKEND_FLASH_SPLITK_SYM

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass


@dataclass(frozen=True)
class DensityGridConfig:
    results_dir: str | None = None
    output_subdir: str = "grids"
    top_k: int = 25
    dpi: int = 200

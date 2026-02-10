from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass


@dataclass(frozen=True)
class Toy1dMoGOraclePlotConfig:
    results_dir: str | None = None
    output_subdir: str = "figures"
    dpi: int = 300

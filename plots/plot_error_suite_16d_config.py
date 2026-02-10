from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSuitePlotConfig:
    results_dir: str | None = None
    output_subdir: str = "figures"
    dpi: int = 300
    font_size: int = 9
    column_width_in: float = 3.3
    wide_width_in: float = 6.8
    error_metric: str = "max_abs_log_err"
    speed_metric: str = "speedup"
    time_metric: str = "time_flash_ms"
    experiment_kind: str | None = None
    status_filter: str = "ok"
    output_filter: str = "log_density"
    make_table: bool = True
    table_filename: str = "table_error_suite_16d.txt"

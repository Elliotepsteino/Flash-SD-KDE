from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.error_suite_16d.run import run_from_config_path


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


@pytest.mark.small
def test_error_suite_16d_smoke(tmp_path: Path) -> None:
    config_path = Path("configs/error_suite_16d/smoke.yaml")
    cfg = _load_config(config_path)

    if not torch.cuda.is_available():
        cfg["suite"]["device"] = "cpu"
        cfg["suite"]["require_gpu_name_contains"] = None
        cfg["flash_impl"]["enabled"] = False
        cfg["reference_impl"]["enabled"] = True
        cfg["reference_impl"]["params"]["dtype"] = "fp32"

    temp_cfg = tmp_path / "smoke.yaml"
    _write_config(temp_cfg, cfg)

    out_dir = run_from_config_path(temp_cfg)
    results_path = out_dir / "results.csv"
    assert results_path.exists()

    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows
    row = rows[0]
    if row.get("status") == "ok":
        max_abs = float(row["max_abs_log_err"])
        assert np.isfinite(max_abs)
        if row.get("nll_hat"):
            assert np.isfinite(float(row["nll_hat"]))

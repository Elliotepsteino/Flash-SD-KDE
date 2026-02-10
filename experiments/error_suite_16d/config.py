from __future__ import annotations

from gitbud.gitbud import get_exp_info, inject_repo_into_sys_path

inject_repo_into_sys_path()

import copy
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from flash_sd_kde.utils import ensure_repo

DEFAULT_CONFIG: dict[str, Any] = {
    "suite": {
        "name": "error_suite_16d",
        "out_dir": "file_storage/error_suite_16d/${timestamp}",
        "seed": 0,
        "device": "cuda",
        "require_gpu_name_contains": "A100",
        "hard_fail_if_not_a100": False,
        "deterministic": False,
        "enable_tf32": False,
        "matmul_precision": "high",
        "cudnn_benchmark": True,
    },
    "experiment": {
        "kind": "kde_correctness_logspace",
        "repeats": 5,
        "warmup": 3,
        "timing_repeats": 5,
    },
    "data": {
        "dataset": "gm_diag_16d",
        "dim": 16,
        "standardize": True,
        "n_train": 200000,
        "n_test": 50000,
        "distribution_params": {
            "n_components": 8,
            "component_std": 1.0,
            "mean_scale": 2.0,
            "weights": "uniform",
        },
    },
    "kde": {
        "kernel": "gaussian",
        "output": "log_density",
        "bandwidth": {
            "mode": "rule_times_scale",
            "rule": "silverman_diag",
            "scale": 1.0,
            "fixed_h": None,
        },
        "chunking": {
            "enabled": True,
            "query_chunk_size": 32768,
            "sample_chunk_size": 65536,
            "auto_target_fraction": 0.6,
        },
    },
    "flash_impl": {
        "enabled": True,
        "emp_score_backend": "flash_sd_kde",
        "params": {
            "compute_dtype": "bf16",
            "accumulate_dtype": "fp32",
            "block_q": 128,
            "block_n": 128,
            "num_warps": 4,
            "num_stages": 2,
        },
        "precision_mode": "fast_tf32",
        "kde_backend": "splitk_stream",
        "use_precomputed_norms": True,
        "autotune": True,
    },
    "reference_impl": {
        "enabled": True,
        "type": "torch_chunked_fp64_logspace",
        "params": {
            "dtype": "fp64",
            "use_logsumexp": True,
            "tier2_fp32_if_too_slow": True,
            "tier2_max_pairs": 2e9,
            "tier1_subsample": {
                "enabled": True,
                "n_ref": 50000,
                "n_test_ref": 20000,
            },
        },
    },
    "statistical": {
        "enabled": True,
        "metrics": {
            "compute_nll": True,
            "compute_kl": True,
            "compute_log_mse": True,
            "compute_ise": True,
        },
        "ise": {
            "proposal": "from_true",
            "proposal_params": {
                "wide_gaussian_std": 2.5,
                "inflate_factor": 2.0,
            },
            "clip_log_ratio": 60.0,
        },
    },
    "checks": {
        "enabled": True,
        "max_abs_log_err": 1e-2,
        "rmse_log_err": 1e-3,
        "min_finite_fraction_flash": 0.999,
    },
    "logging": {
        "write_csv": True,
        "write_json": True,
        "write_debug_tensors": False,
    },
    "oracle": {
        "method": "kde",
        "clamp_nonnegative": True,
        "clamp_eps": 1e-30,
        "nonfused_device": "cpu",
        "nonfused_dtype": "fp64",
    },
    "summary_plots": [
        "fig_oracle_error_vs_n_16d",
        "pareto_kl_vs_throughput",
        "pareto_nll_vs_throughput",
    ],
    "sweep": {
        "parameters": {},
        "fail_fast": False,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def apply_defaults(config: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    if defaults is None:
        defaults = DEFAULT_CONFIG
    merged = copy.deepcopy(defaults)
    return _deep_update(merged, config)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"config at {path} must be a mapping at the top level.")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    return apply_defaults(raw)


def resolve_out_dir(config: dict[str, Any]) -> Path:
    repo_root = Path(ensure_repo())
    template = str(config.get("suite", {}).get("out_dir", DEFAULT_CONFIG["suite"]["out_dir"]))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_info = get_exp_info()
    resolved = template.replace("${timestamp}", timestamp).replace("${exp_info}", exp_info)
    out_dir = Path(resolved)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    return out_dir


def set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def expand_grid(parameters: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    items = list(parameters.items())
    if not items:
        return [{}]
    grids: list[dict[str, Any]] = [{}]
    for path, values in items:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise ValueError(f"sweep parameter {path} must be a list of values")
        next_grids: list[dict[str, Any]] = []
        for value in values:
            for base in grids:
                merged = dict(base)
                merged[path] = value
                next_grids.append(merged)
        grids = next_grids
    return grids

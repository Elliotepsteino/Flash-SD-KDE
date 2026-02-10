from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import json
import csv
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.error_suite_16d.adapters import (
    flash_available,
    run_flash_linearized_log_density,
    run_flash_log_density,
    run_sd_kde_log_density,
    sd_kde_available,
)
from experiments.error_suite_16d.config import load_config, resolve_out_dir
from experiments.error_suite_16d.datasets import make_dataset
from experiments.error_suite_16d.metrics import (
    compute_log_error_metrics,
    compute_oracle_error_metrics,
    compute_statistical_metrics,
)
from experiments.error_suite_16d.numerics import exp_diagnostics
from experiments.error_suite_16d.reference import (
    reference_linearized_density,
    reference_log_density,
    silverman_diag_bandwidth,
)
from flash_sd_kde.utils import get_repo_state, write_json
from globals import ND_FEATURES


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    suite_cfg = cfg.get("suite", {})
    deterministic = bool(suite_cfg.get("deterministic", False))
    cudnn_benchmark = bool(suite_cfg.get("cudnn_benchmark", True))
    enable_tf32 = bool(suite_cfg.get("enable_tf32", False))
    matmul_precision = suite_cfg.get("matmul_precision", None)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        cudnn_benchmark = False

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))

    return {
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "enable_tf32": enable_tf32,
        "matmul_precision": matmul_precision,
    }


def _gpu_info(device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type != "cuda":
        return info
    name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    capability = torch.cuda.get_device_capability(device)
    info.update(
        {
            "gpu_name": name,
            "gpu_total_vram_bytes": int(total_mem),
            "gpu_sm_major": int(capability[0]),
            "gpu_sm_minor": int(capability[1]),
            "gpu_sm": f"{capability[0]}.{capability[1]}",
        }
    )

    required = cfg.get("suite", {}).get("require_gpu_name_contains")
    strict = bool(cfg.get("suite", {}).get("hard_fail_if_not_a100", False))
    if required and required not in name:
        info["gpu_name_mismatch"] = True
        msg = f"GPU name '{name}' does not include required substring '{required}'."
        if strict:
            raise RuntimeError(msg)
        info["gpu_name_warning"] = msg
    return info


def _resolve_chunking(cfg: dict[str, Any], *, n_queries: int, n_samples: int, device: torch.device) -> dict[str, Any]:
    chunk_cfg = dict(cfg.get("kde", {}).get("chunking", {}) or {})
    if not chunk_cfg.get("enabled", True):
        return {"query_chunk_size": n_queries, "sample_chunk_size": n_samples}

    query_chunk_size = chunk_cfg.get("query_chunk_size")
    sample_chunk_size = chunk_cfg.get("sample_chunk_size")

    if sample_chunk_size in (None, 0, "auto"):
        sample_chunk_size = n_samples

    if query_chunk_size in (None, 0, "auto"):
        if device.type == "cuda":
            free_mem, _ = torch.cuda.mem_get_info(device)
            target_fraction = float(chunk_cfg.get("auto_target_fraction", 0.6))
            bytes_per = 4
            max_q = int((free_mem * target_fraction) // (max(1, int(sample_chunk_size)) * bytes_per))
            query_chunk_size = max(1, min(n_queries, max_q))
        else:
            query_chunk_size = n_queries

    return {
        "query_chunk_size": int(query_chunk_size),
        "sample_chunk_size": int(sample_chunk_size),
    }


def _resolve_bandwidth(cfg: dict[str, Any], samples: torch.Tensor) -> tuple[float, dict[str, Any]]:
    bw_cfg = cfg.get("kde", {}).get("bandwidth", {})
    mode = bw_cfg.get("mode", "fixed")
    meta = {"bandwidth_mode": mode}

    if mode == "rule_times_scale":
        rule = bw_cfg.get("rule")
        scale = float(bw_cfg.get("scale", 1.0))
        if rule == "silverman_diag":
            h = silverman_diag_bandwidth(samples)
        else:
            raise ValueError(f"unsupported bandwidth rule: {rule}")
        meta.update({"bandwidth_rule": rule, "bandwidth_scale": scale})
        return h * scale, meta

    if mode == "fixed":
        fixed = bw_cfg.get("fixed_h")
        if fixed is None:
            raise ValueError("fixed_h must be set for fixed bandwidth mode")
        meta.update({"bandwidth_value": float(fixed)})
        return float(fixed), meta

    raise ValueError(f"unsupported bandwidth mode: {mode}")


def _time_call(fn, *, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device)
        return result, start.elapsed_time(end) / 1000.0
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _time_repeats(fn, *, repeats: int, device: torch.device) -> tuple[Any, float, float]:
    durations = []
    result = None
    for _ in range(max(1, repeats)):
        result, elapsed = _time_call(fn, device=device)
        durations.append(elapsed)
    durations = sorted(durations)
    median = durations[len(durations) // 2]
    p90 = durations[int(0.9 * (len(durations) - 1))]
    return result, median, p90


def _distance_diagnostics(samples: torch.Tensor, queries: torch.Tensor, bandwidth: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_pairs = min(1024, samples.shape[0], queries.shape[0])
    s_idx = rng.choice(samples.shape[0], size=n_pairs, replace=False)
    q_idx = rng.choice(queries.shape[0], size=n_pairs, replace=False)
    diffs = samples[s_idx].detach().cpu().numpy() - queries[q_idx].detach().cpu().numpy()
    dist2 = np.sum(diffs * diffs, axis=1)
    median_dist2 = float(np.median(dist2))
    exponent_scale = float(-median_dist2 / (2.0 * bandwidth * bandwidth))
    return {"median_dist2": median_dist2, "median_exponent_scale": exponent_scale}


def _normalization_check(
    samples: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    flash_params: dict[str, Any],
) -> dict[str, Any]:
    n_samples = min(samples.shape[0], 256)
    n_queries = min(queries.shape[0], 1)
    sub_samples = samples[:n_samples]
    sub_queries = queries[:n_queries]

    # Keep the reference/manual computation on CPU to avoid device mismatch.
    sub_samples_cpu = sub_samples.detach().cpu()
    sub_queries_cpu = sub_queries.detach().cpu()

    ref_logp = reference_log_density(
        sub_samples_cpu,
        sub_queries_cpu,
        bandwidth=bandwidth,
        chunk_cfg={"query_chunk_size": n_queries, "sample_chunk_size": n_samples},
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    diff = sub_queries_cpu[:, None, :] - sub_samples_cpu[None, :, :]
    dist2 = torch.sum(diff * diff, dim=-1)
    exponent = -0.5 * dist2 / (bandwidth * bandwidth)
    max_exp = torch.max(exponent, dim=1, keepdim=True).values
    logsum = torch.log(torch.sum(torch.exp(exponent - max_exp), dim=1)) + max_exp.squeeze(1)
    log_const = -math.log(n_samples) - ND_FEATURES * math.log(bandwidth) - 0.5 * ND_FEATURES * math.log(
        2.0 * math.pi
    )
    manual_logp = logsum + log_const

    check = {"ref_manual_abs_diff": float(torch.max(torch.abs(ref_logp - manual_logp)).item())}

    if flash_params.get("enabled", True) and flash_available():
        flash_logp = run_flash_log_density(
            sub_samples,
            sub_queries,
            bandwidth=bandwidth,
            kernel="gaussian",
            device=device,
            precision_mode=flash_params.get("precision_mode", "fast_tf32"),
            kde_backend=flash_params.get("kde_backend", "splitk_stream"),
            use_precomputed_norms=flash_params.get("use_precomputed_norms", True),
            autotune=flash_params.get("autotune", True),
            compute_dtype=flash_params.get("params", {}).get("compute_dtype"),
        )
        check["flash_ref_abs_diff"] = float(torch.max(torch.abs(flash_logp.cpu() - ref_logp)).item())
    return check


def _apply_checks(metrics: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = cfg.get("checks", {})
    if not checks.get("enabled", True):
        return False, []

    reasons = []
    max_abs = checks.get("max_abs_log_err")
    if max_abs is not None and metrics.get("max_abs_log_err") is not None:
        if metrics["max_abs_log_err"] > float(max_abs):
            reasons.append(f"max_abs_log_err>{max_abs}")

    rmse = checks.get("rmse_log_err")
    if rmse is not None and metrics.get("rmse_log_err") is not None:
        if metrics["rmse_log_err"] > float(rmse):
            reasons.append(f"rmse_log_err>{rmse}")

    min_finite = checks.get("min_finite_fraction_flash")
    if min_finite is not None and metrics.get("finite_fraction_flash") is not None:
        if metrics["finite_fraction_flash"] < float(min_finite):
            reasons.append(f"finite_fraction_flash<{min_finite}")

    return bool(reasons), reasons


def run_one(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    suite_cfg = cfg.get("suite", {})
    exp_cfg = cfg.get("experiment", {})
    flash_cfg = cfg.get("flash_impl", {})
    ref_cfg = cfg.get("reference_impl", {})
    stat_cfg = cfg.get("statistical", {})
    oracle_cfg = cfg.get("oracle", {})

    device = torch.device(suite_cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested.")

    seed = int(suite_cfg.get("seed", 0))
    _set_seed(seed)
    torch_cfg = _configure_torch(cfg, device)
    gpu_info = _gpu_info(device, cfg)

    train, test, truth, data_stats = make_dataset(cfg, seed=seed, device=device, dtype=torch.float32)
    truth_meta = truth.metadata()

    bandwidth, bw_meta = _resolve_bandwidth(cfg, train)
    chunk_cfg = _resolve_chunking(cfg, n_queries=test.shape[0], n_samples=train.shape[0], device=device)

    diagnostics = _distance_diagnostics(train, test, bandwidth, seed)
    norm_check = _normalization_check(train, test, bandwidth, device=device, flash_params=flash_cfg)

    results: dict[str, Any] = {
        "status": "ok",
        "error_type": "",
        "error_msg": "",
        "seed": seed,
        "experiment_kind": exp_cfg.get("kind"),
        "method": oracle_cfg.get("method"),
        "dataset": cfg.get("data", {}).get("dataset"),
        "n_train": int(cfg.get("data", {}).get("n_train", 0)),
        "n_test": int(cfg.get("data", {}).get("n_test", 0)),
        "dim": int(cfg.get("data", {}).get("dim", 0)),
        "kernel": cfg.get("kde", {}).get("kernel"),
        "output": cfg.get("kde", {}).get("output"),
        "bandwidth": bandwidth,
        "bandwidth_mode": bw_meta.get("bandwidth_mode"),
        "bandwidth_rule": bw_meta.get("bandwidth_rule"),
        "bandwidth_scale": bw_meta.get("bandwidth_scale"),
        "bandwidth_value": bw_meta.get("bandwidth_value"),
        "compute_dtype": flash_cfg.get("params", {}).get("compute_dtype"),
        "accumulate_dtype": flash_cfg.get("params", {}).get("accumulate_dtype"),
        "block_q": flash_cfg.get("params", {}).get("block_q"),
        "block_n": flash_cfg.get("params", {}).get("block_n"),
        "num_warps": flash_cfg.get("params", {}).get("num_warps"),
        "num_stages": flash_cfg.get("params", {}).get("num_stages"),
        "precision_mode": flash_cfg.get("precision_mode"),
        "kde_backend": flash_cfg.get("kde_backend"),
        "enable_tf32": bool(suite_cfg.get("enable_tf32", False)),
        "clamped_fraction_flash": None,
        "clamped_fraction_laplace": None,
    }

    meta = {
        "seed": seed,
        "suite": suite_cfg,
        "experiment": exp_cfg,
        "torch": torch_cfg,
        "gpu": gpu_info,
        "repo": get_repo_state(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "data_stats": data_stats,
        "data_cfg": cfg.get("data", {}),
        "truth": truth_meta,
        "distance_diagnostics": diagnostics,
        "norm_check": norm_check,
    }

    if exp_cfg.get("kind") == "oracle_error_16d":
        method = str(oracle_cfg.get("method", "kde"))
        clamp_nonneg = bool(oracle_cfg.get("clamp_nonnegative", True))
        clamp_eps = float(oracle_cfg.get("clamp_eps", 1e-30))

        logp_true = truth.true_log_density(test).detach().cpu()

        logp_hat = None
        if method == "kde":
            logp_hat = run_flash_log_density(
                train,
                test,
                bandwidth=bandwidth,
                kernel="gaussian",
                device=device,
                precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
                kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
                use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
                autotune=flash_cfg.get("autotune", True),
                compute_dtype=flash_cfg.get("params", {}).get("compute_dtype"),
            ).detach().cpu()

        elif method == "flash_laplace":
            logp_hat, aux = run_flash_linearized_log_density(
                train,
                test,
                bandwidth=bandwidth,
                device=device,
                precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
                kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
                use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
                autotune=flash_cfg.get("autotune", True),
                compute_dtype=flash_cfg.get("params", {}).get("compute_dtype"),
                eps=clamp_eps,
                return_aux=True,
            )
            if clamp_nonneg:
                results["clamped_fraction_laplace"] = aux.get("clamped_fraction")
            logp_hat = logp_hat.detach().cpu()

        elif method == "nonfused_laplace":
            nf_device = torch.device(oracle_cfg.get("nonfused_device", "cpu"))
            dtype_name = str(oracle_cfg.get("nonfused_dtype", "fp64"))
            nf_dtype = torch.float64 if dtype_name == "fp64" else torch.float32
            if nf_device.type == "cuda" and not torch.cuda.is_available():
                nf_device = torch.device("cpu")
            density = reference_linearized_density(
                train,
                test,
                bandwidth=bandwidth,
                chunk_cfg=chunk_cfg,
                dtype=nf_dtype,
                device=nf_device,
            )
            if clamp_nonneg:
                nonpos = (density <= 0).float().mean().item()
                density = torch.clamp(density, min=clamp_eps)
                results["clamped_fraction_laplace"] = float(nonpos)
            logp_hat = torch.log(density).detach().cpu()

        elif method == "emp_sd_kde":
            if not sd_kde_available():
                results["status"] = "skipped"
                results["error_type"] = "sd_kde_unavailable"
            else:
                logp_hat = run_sd_kde_log_density(
                    train,
                    test,
                    bandwidth=bandwidth,
                    device=device,
                    precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
                    emp_score_backend=flash_cfg.get("emp_score_backend"),
                    use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
                    autotune=flash_cfg.get("autotune", True),
                    kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
                ).detach().cpu()
        else:
            raise ValueError(f"unsupported oracle method: {method}")

        if logp_hat is not None:
            oracle_metrics = compute_oracle_error_metrics(logp_hat, logp_true)
            results.update(oracle_metrics)
        return results, meta

    flash_logp = None
    flash_aux = {}
    if flash_cfg.get("enabled", True) and flash_available():
        warmup = int(exp_cfg.get("warmup", 3))
        for _ in range(max(0, warmup)):
            _ = run_flash_log_density(
                train,
                test[: min(test.shape[0], 1)],
                bandwidth=bandwidth,
                kernel="gaussian",
                device=device,
                precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
                kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
                use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
                autotune=flash_cfg.get("autotune", True),
                compute_dtype=flash_cfg.get("params", {}).get("compute_dtype"),
            )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def flash_call():
            return run_flash_log_density(
                train,
                test,
                bandwidth=bandwidth,
                kernel="gaussian",
                device=device,
                precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
                kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
                use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
                autotune=flash_cfg.get("autotune", True),
                compute_dtype=flash_cfg.get("params", {}).get("compute_dtype"),
                return_aux=True,
            )

        (flash_logp, flash_aux), flash_time, flash_p90 = _time_repeats(
            flash_call,
            repeats=int(exp_cfg.get("timing_repeats", 5)),
            device=device,
        )
        if flash_aux:
            results["clamped_fraction_flash"] = flash_aux.get("clamped_fraction")
        results["time_flash_ms"] = flash_time * 1000.0
        results["time_flash_p90_ms"] = flash_p90 * 1000.0
        if device.type == "cuda":
            results["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            results["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        results["throughput_qps"] = test.shape[0] / (flash_time if flash_time > 0 else 1e-9)
    else:
        results["status"] = "skipped"
        results["error_type"] = "flash_unavailable"

    ref_logp = None
    ref_used_dtype = None
    if ref_cfg.get("enabled", True):
        ref_params = ref_cfg.get("params", {})
        dtype_name = ref_params.get("dtype", "fp64")
        dtype = torch.float64 if dtype_name == "fp64" else torch.float32
        ref_device = torch.device(ref_cfg.get("device", device))

        max_pairs = float(ref_params.get("tier2_max_pairs", 2e9))
        if ref_params.get("tier2_fp32_if_too_slow", False):
            if train.shape[0] * test.shape[0] > max_pairs:
                dtype = torch.float32
        ref_used_dtype = dtype

        def ref_call():
            return reference_log_density(
                train,
                test,
                bandwidth=bandwidth,
                chunk_cfg=chunk_cfg,
                dtype=dtype,
                device=ref_device,
            )

        ref_logp, ref_time, ref_p90 = _time_repeats(
            ref_call,
            repeats=int(exp_cfg.get("timing_repeats", 5)),
            device=ref_device,
        )
        results["time_ref_ms"] = ref_time * 1000.0
        results["time_ref_p90_ms"] = ref_p90 * 1000.0

    if flash_logp is not None and ref_logp is not None:
        ref_logp = ref_logp.to(device=flash_logp.device)
        err_metrics = compute_log_error_metrics(flash_logp, ref_logp)
        results.update(err_metrics)
        if results.get("time_flash_ms") and results.get("time_ref_ms"):
            results["speedup"] = results["time_ref_ms"] / results["time_flash_ms"]

        if err_metrics.get("finite_fraction_flash", 1.0) < 1.0:
            results["status"] = "failed"
            results["error_type"] = "nan_inf"

        failed, reasons = _apply_checks(err_metrics, cfg)
        if failed:
            results["status"] = "failed"
            results["error_type"] = "check_failed"
            results["error_msg"] = ";".join(reasons)

    if stat_cfg.get("enabled", False) and flash_logp is not None:
        logp_true = truth.true_log_density(test)
        stat_metrics = compute_statistical_metrics(
            flash_logp.detach().cpu(),
            logp_true.detach().cpu(),
            ise_cfg={"enabled": stat_cfg.get("metrics", {}).get("compute_ise", False)},
        )
        results.update(stat_metrics)

    tier1_cfg = ref_cfg.get("params", {}).get("tier1_subsample", {})
    if flash_logp is not None and tier1_cfg.get("enabled", False):
        n_ref = int(tier1_cfg.get("n_ref", 50000))
        n_test_ref = int(tier1_cfg.get("n_test_ref", 20000))
        sub_train = train[:n_ref]
        sub_test = test[:n_test_ref]
        tier_ref = reference_log_density(
            sub_train,
            sub_test,
            bandwidth=bandwidth,
            chunk_cfg={"query_chunk_size": min(n_test_ref, sub_test.shape[0]), "sample_chunk_size": min(n_ref, sub_train.shape[0])},
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        tier_flash = run_flash_log_density(
            sub_train,
            sub_test,
            bandwidth=bandwidth,
            kernel="gaussian",
            device=device,
            precision_mode=flash_cfg.get("precision_mode", "fast_tf32"),
            kde_backend=flash_cfg.get("kde_backend", "splitk_stream"),
            use_precomputed_norms=flash_cfg.get("use_precomputed_norms", True),
            autotune=flash_cfg.get("autotune", True),
            compute_dtype=flash_cfg.get("params", {}).get("compute_dtype"),
        )
        tier_metrics = compute_log_error_metrics(tier_flash.detach().cpu(), tier_ref.detach().cpu())
        results.update({f"tier1_{k}": v for k, v in tier_metrics.items()})

    meta["flash_aux"] = flash_aux
    meta["exp_diagnostics"] = exp_diagnostics(flash_logp) if flash_logp is not None else {}
    meta["reference_dtype_used"] = str(ref_used_dtype)

    return results, meta


def run_from_config_path(config_path: Path, *, out_dir_override: Path | None = None) -> Path:
    cfg = load_config(config_path)
    out_dir = out_dir_override or resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    results, meta = run_one(cfg)
    if cfg.get("logging", {}).get("write_csv", True):
        results_path = out_dir / "results.csv"
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)

    if cfg.get("logging", {}).get("write_json", True):
        write_json(out_dir / "metrics.json", {"results": results, "meta": meta})
        write_json(out_dir / "metadata.json", meta)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error suite A100 16D")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir_override = Path(args.out_dir) if args.out_dir else None
    try:
        out_dir = run_from_config_path(config_path, out_dir_override=out_dir_override)
        print(f"Outputs written to {out_dir}")
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

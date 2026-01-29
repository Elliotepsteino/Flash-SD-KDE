from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from benchmarks.toy_1d_mog_oracle_config import Toy1dMoGOracleConfig
from flash_sd_kde.kde import kde_eval, kde_eval_linearized, kde_eval_linearized_nonfused
from flash_sd_kde.reference import kde_eval_1d_linearized_numpy, kde_eval_1d_numpy, silverman_bandwidth_1d
from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json
from globals import DEFAULT_EPS

_METHODS = ("kde", "linearized", "linearized_nonfused", "emp_sd_kde")


def _mog_sample(rng: np.random.Generator, n: int, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    comp = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(loc=means[comp], scale=stds[comp]).astype(np.float32, copy=False)


def _mog_pdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    total = np.zeros_like(x)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for w, mu, sigma in zip(weights, means, stds):
        z = (x - mu) / sigma
        total += w * (inv_sqrt_2pi / sigma) * np.exp(-0.5 * z * z)
    return total


def _oracle_errors(x_grid: np.ndarray, est: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    diff = est - true
    ise = float(np.trapezoid(diff * diff, x_grid))
    iae = float(np.trapezoid(np.abs(diff), x_grid))
    max_abs = float(np.max(np.abs(diff)))
    neg_mass = float(np.trapezoid(np.abs(np.minimum(est, 0.0)), x_grid))
    return {
        "ise": ise,
        "iae": iae,
        "max_abs": max_abs,
        "neg_mass": neg_mass,
    }


def _kde_eval_1d_linearized_nonfused_numpy(
    queries: np.ndarray,
    data: np.ndarray,
    bandwidth: float,
    *,
    chunk_size: int,
) -> np.ndarray:
    base = kde_eval_1d_numpy(queries, data, bandwidth)
    n = data.size
    if n == 0:
        raise ValueError("data must contain at least one element.")
    inv_h = 1.0 / bandwidth
    sum_phi_scaled = np.zeros_like(queries, dtype=np.float32)
    chunk = max(int(chunk_size), 1)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        data_chunk = data[start:end]
        diff = (queries[:, None] - data_chunk[None, :]) * np.float32(inv_h)
        scaled = diff * diff
        phi = np.exp(-0.5 * scaled)
        sum_phi_scaled += (phi * scaled).sum(axis=1)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * bandwidth * n)
    return base * (1.0 + 0.5) - 0.5 * norm * sum_phi_scaled


def _time_call(fn, *, device: torch.device) -> Tuple[object, float]:
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


def _time_repeated(fn, *, device: torch.device, warmup: int, repeats: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        _, elapsed = _time_call(fn, device=device)
        timings.append(elapsed)
    return float(np.mean(timings)), float(np.std(timings))


def _emp_sd_kde_transform_1d_numpy(data: np.ndarray, bandwidth: float, *, chunk_size: int) -> np.ndarray:
    n = data.shape[0]
    pdf_sum = np.zeros((n,), dtype=np.float32)
    weighted_sum = np.zeros((n,), dtype=np.float32)
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = data[start:end]
        diff = (data[:, None] - chunk[None, :]) / bandwidth
        phi = np.exp(-0.5 * diff * diff)
        pdf_sum += phi.sum(axis=1)
        weighted_sum += (phi * chunk[None, :]).sum(axis=1)
    score = (weighted_sum / (pdf_sum + DEFAULT_EPS) - data) * inv_h2
    delta = 0.5 * (bandwidth ** 2)
    return data + delta * score


def _emp_sd_kde_transform_1d_torch(
    data: torch.Tensor, bandwidth: float, *, chunk_size: int, eps: float
) -> torch.Tensor:
    n = data.shape[0]
    pdf_sum = torch.zeros((n,), device=data.device, dtype=torch.float32)
    weighted_sum = torch.zeros((n,), device=data.device, dtype=torch.float32)
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = data[start:end]
        diff = (data[:, None] - chunk[None, :]) / bandwidth
        phi = torch.exp(-0.5 * diff * diff)
        pdf_sum += phi.sum(dim=1)
        weighted_sum += (phi * chunk[None, :]).sum(dim=1)
    score = (weighted_sum / (pdf_sum + eps) - data) * inv_h2
    delta = 0.5 * (bandwidth ** 2)
    return data + delta * score


def _ensure_device(config: Toy1dMoGOracleConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but was requested.")
        return device
    if device.type == "cpu":
        return device
    raise ValueError(f"unsupported device: {config.device}")


def run_benchmark(config: Toy1dMoGOracleConfig) -> Path:
    device = _ensure_device(config)

    rng = np.random.default_rng(config.seed)
    weights = np.asarray(config.mixture_weights, dtype=np.float32)
    weights = weights / weights.sum()
    means = np.asarray(config.mixture_means, dtype=np.float32)
    stds = np.asarray(config.mixture_stds, dtype=np.float32)

    x_grid = np.linspace(config.grid_min, config.grid_max, config.n_grid, dtype=np.float32)
    true_density = _mog_pdf(x_grid, weights, means, stds)

    run_dir = make_run_dir(tag=config.output_tag)

    results = {
        "config": asdict(config),
        "meta": get_repo_state(),
        "n_train_list": list(config.n_train_list),
        "mixture": {
            "weights": weights.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
        },
        "grid": {
            "min": float(config.grid_min),
            "max": float(config.grid_max),
            "n": int(config.n_grid),
        },
        "metrics": {method: {} for method in _METHODS},
        "runtime_sec": {method: {} for method in _METHODS},
        "runtime_breakdown_sec": {"emp_sd_kde": {}},
    }

    density_curves = {}
    max_n = max(config.n_train_list)

    for repeat_id in range(config.n_repeats):
        train_all = _mog_sample(rng, max_n, weights, means, stds)

        for n_train in config.n_train_list:
            data_np = train_all[:n_train]
            h = silverman_bandwidth_1d(data_np) * config.bandwidth_multiplier
            n_key = str(n_train)

            if device.type == "cuda":
                data_t = torch.as_tensor(data_np, device=device, dtype=torch.float32)
                grid_t = torch.as_tensor(x_grid, device=device, dtype=torch.float32)

                def kde_fn():
                    return kde_eval(
                        data_t,
                        grid_t,
                        h,
                        device=device,
                        precision_mode=config.precision_mode,
                        kde_backend=config.kde_backend,
                        use_precomputed_norms=config.use_precomputed_norms,
                        autotune=config.autotune,
                    )

                def lin_fn():
                    return kde_eval_linearized(
                        data_t,
                        grid_t,
                        h,
                        device=device,
                        precision_mode=config.precision_mode,
                        kde_backend=config.kde_backend,
                        use_precomputed_norms=config.use_precomputed_norms,
                        autotune=config.autotune,
                    )

                def lin_nf_fn():
                    return kde_eval_linearized_nonfused(
                        data_t,
                        grid_t,
                        h,
                        device=device,
                        precision_mode=config.precision_mode,
                        kde_backend=config.kde_backend,
                        use_precomputed_norms=config.use_precomputed_norms,
                        autotune=config.autotune,
                        chunk_size=config.laplace_chunk_size,
                    )

                def emp_score_shift():
                    return _emp_sd_kde_transform_1d_torch(
                        data_t, h, chunk_size=config.emp_chunk_size, eps=DEFAULT_EPS
                    )

                def emp_eval(debiased: torch.Tensor):
                    return kde_eval(
                        debiased,
                        grid_t,
                        h,
                        device=device,
                        precision_mode=config.precision_mode,
                        kde_backend=config.kde_backend,
                        use_precomputed_norms=config.use_precomputed_norms,
                        autotune=config.autotune,
                    )

                _time_repeated(kde_fn, device=device, warmup=config.timing_warmup, repeats=1)
                _time_repeated(lin_fn, device=device, warmup=config.timing_warmup, repeats=1)
                _time_repeated(lin_nf_fn, device=device, warmup=config.timing_warmup, repeats=1)
                _time_repeated(emp_score_shift, device=device, warmup=config.timing_warmup, repeats=1)

                dens_kde = kde_fn().detach().cpu().numpy()
                dens_lin = lin_fn().detach().cpu().numpy()
                dens_lin_nf = lin_nf_fn().detach().cpu().numpy()
                debiased = emp_score_shift()
                dens_emp = emp_eval(debiased).detach().cpu().numpy()

                kde_mean, kde_std = _time_repeated(
                    kde_fn,
                    device=device,
                    warmup=0,
                    repeats=config.timing_repeats,
                )
                lin_mean, lin_std = _time_repeated(
                    lin_fn,
                    device=device,
                    warmup=0,
                    repeats=config.timing_repeats,
                )
                lin_nf_mean, lin_nf_std = _time_repeated(
                    lin_nf_fn,
                    device=device,
                    warmup=0,
                    repeats=config.timing_repeats,
                )
                emp_score_mean, emp_score_std = _time_repeated(
                    emp_score_shift,
                    device=device,
                    warmup=0,
                    repeats=config.timing_repeats,
                )

                def emp_full():
                    return emp_eval(emp_score_shift())

                emp_total_mean, emp_total_std = _time_repeated(
                    emp_full,
                    device=device,
                    warmup=0,
                    repeats=config.timing_repeats,
                )
                emp_eval_mean = max(emp_total_mean - emp_score_mean, 0.0)
                emp_eval_std = emp_total_std
            else:
                dens_kde = kde_eval_1d_numpy(x_grid, data_np, h)
                dens_lin = kde_eval_1d_linearized_numpy(x_grid, data_np, h)
                dens_lin_nf = _kde_eval_1d_linearized_nonfused_numpy(
                    x_grid, data_np, h, chunk_size=config.laplace_chunk_size
                )
                debiased = _emp_sd_kde_transform_1d_numpy(
                    data_np, h, chunk_size=config.emp_chunk_size
                )
                dens_emp = kde_eval_1d_numpy(x_grid, debiased, h)

                def kde_fn():
                    return kde_eval_1d_numpy(x_grid, data_np, h)

                def lin_fn():
                    return kde_eval_1d_linearized_numpy(x_grid, data_np, h)

                def lin_nf_fn():
                    return _kde_eval_1d_linearized_nonfused_numpy(
                        x_grid, data_np, h, chunk_size=config.laplace_chunk_size
                    )

                def emp_score_shift():
                    return _emp_sd_kde_transform_1d_numpy(
                        data_np, h, chunk_size=config.emp_chunk_size
                    )

                def emp_full():
                    deb = emp_score_shift()
                    return kde_eval_1d_numpy(x_grid, deb, h)

                kde_mean, kde_std = _time_repeated(
                    kde_fn,
                    device=device,
                    warmup=config.timing_warmup,
                    repeats=config.timing_repeats,
                )
                lin_mean, lin_std = _time_repeated(
                    lin_fn,
                    device=device,
                    warmup=config.timing_warmup,
                    repeats=config.timing_repeats,
                )
                lin_nf_mean, lin_nf_std = _time_repeated(
                    lin_nf_fn,
                    device=device,
                    warmup=config.timing_warmup,
                    repeats=config.timing_repeats,
                )
                emp_score_mean, emp_score_std = _time_repeated(
                    emp_score_shift,
                    device=device,
                    warmup=config.timing_warmup,
                    repeats=config.timing_repeats,
                )
                emp_total_mean, emp_total_std = _time_repeated(
                    emp_full,
                    device=device,
                    warmup=config.timing_warmup,
                    repeats=config.timing_repeats,
                )
                emp_eval_mean = max(emp_total_mean - emp_score_mean, 0.0)
                emp_eval_std = emp_total_std

            for method, dens in [
                ("kde", dens_kde),
                ("linearized", dens_lin),
                ("linearized_nonfused", dens_lin_nf),
                ("emp_sd_kde", dens_emp),
            ]:
                metrics = _oracle_errors(x_grid, dens, true_density)
                entry = results["metrics"][method].setdefault(n_key, {"repeat": []})
                entry["repeat"].append(metrics)

            runtime_entry = results["runtime_sec"]["kde"].setdefault(n_key, {"repeat": []})
            runtime_entry["repeat"].append({"mean": kde_mean, "std": kde_std})
            runtime_entry = results["runtime_sec"]["linearized"].setdefault(n_key, {"repeat": []})
            runtime_entry["repeat"].append({"mean": lin_mean, "std": lin_std})
            runtime_entry = results["runtime_sec"]["linearized_nonfused"].setdefault(
                n_key, {"repeat": []}
            )
            runtime_entry["repeat"].append({"mean": lin_nf_mean, "std": lin_nf_std})
            runtime_entry = results["runtime_sec"]["emp_sd_kde"].setdefault(n_key, {"repeat": []})
            runtime_entry["repeat"].append({"mean": emp_total_mean, "std": emp_total_std})
            breakdown_entry = results["runtime_breakdown_sec"]["emp_sd_kde"].setdefault(
                n_key, {"repeat": []}
            )
            breakdown_entry["repeat"].append(
                {
                    "score_shift_mean": emp_score_mean,
                    "score_shift_std": emp_score_std,
                    "eval_mean": emp_eval_mean,
                    "eval_std": emp_eval_std,
                }
            )

            if config.save_density_curves and n_train == max_n and repeat_id == 0:
                density_curves = {
                    "x_grid": x_grid,
                    "true_density": true_density,
                    "kde_density": dens_kde,
                    "linearized_density": dens_lin,
                    "linearized_nonfused_density": dens_lin_nf,
                    "emp_sd_kde_density": dens_emp,
                    "n_train": np.asarray([n_train], dtype=np.int32),
                }

    for method in _METHODS:
        for n_key, entry in results["metrics"][method].items():
            repeats = entry["repeat"]
            for metric in ["ise", "iae", "max_abs", "neg_mass"]:
                vals = np.array([rep[metric] for rep in repeats], dtype=np.float64)
                entry[f"{metric}_mean"] = float(vals.mean())
                entry[f"{metric}_std"] = float(vals.std())
            del entry["repeat"]

    for method in _METHODS:
        for n_key, entry in results["runtime_sec"][method].items():
            repeats = entry["repeat"]
            means = np.array([rep["mean"] for rep in repeats], dtype=np.float64)
            stds = np.array([rep["std"] for rep in repeats], dtype=np.float64)
            entry["mean"] = float(means.mean())
            entry["std"] = float(stds.mean())
            del entry["repeat"]

    for n_key, entry in results["runtime_breakdown_sec"]["emp_sd_kde"].items():
        repeats = entry["repeat"]
        for metric in ["score_shift_mean", "score_shift_std", "eval_mean", "eval_std"]:
            vals = np.array([rep[metric] for rep in repeats], dtype=np.float64)
            entry[metric] = float(vals.mean())
        del entry["repeat"]

    write_json(run_dir / "results.json", results)

    if config.save_density_curves and density_curves:
        np.savez_compressed(run_dir / "densities_curves.npz", **density_curves)

    return run_dir


def main() -> None:
    config = Toy1dMoGOracleConfig()
    run_dir = run_benchmark(config)
    print(f"Benchmark complete. Results in {run_dir}")


if __name__ == "__main__":
    main()

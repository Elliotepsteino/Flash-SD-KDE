"""Benchmark Triton KDE vs the existing Silverman implementation."""

from __future__ import annotations

import argparse
import time
from typing import List, Sequence

import numpy as np
import torch

try:
    from sklearn.neighbors import KernelDensity
except Exception:  # pragma: no cover - optional dependency
    KernelDensity = None

from kde_utils import (
    kde_pdf_eval,
    mixture_params_list,
    one_step_debiased_data_emp_kde,
    sample_from_mixture,
    silverman_bandwidth,
)
from triton_kde import gaussian_kde_triton, empirical_sd_kde_triton


def _parse_seeds(seeds: Sequence[int] | str) -> List[int]:
    if isinstance(seeds, str):
        return [int(tok.strip()) for tok in seeds.split(",") if tok.strip()]
    return list(seeds)


def _cpu_kde(train: np.ndarray, queries: np.ndarray, bandwidth: float):
    start = time.perf_counter()
    densities = kde_pdf_eval(queries, train, bandwidth)
    elapsed = time.perf_counter() - start
    return densities, elapsed


def _sklearn_kde(
    train: np.ndarray, queries: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, float]:
    if KernelDensity is None:
        raise RuntimeError("scikit-learn is not installed; cannot run sklearn baseline.")

    start = time.perf_counter()
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(train.reshape(-1, 1))
    log_dens = kde.score_samples(queries.reshape(-1, 1))
    densities = np.exp(log_dens)
    elapsed = time.perf_counter() - start
    return densities, elapsed


def _gpu_kde(
    train: np.ndarray,
    queries: np.ndarray,
    bandwidth: float,
    *,
    device: str,
    warmup_done: bool,
) -> tuple[np.ndarray, float]:
    torch_device = torch.device(device)
    if not warmup_done:
        gaussian_kde_triton(train, queries, bandwidth, device=device)

    torch.cuda.synchronize(torch_device)
    start = time.perf_counter()
    densities = gaussian_kde_triton(
        train, queries, bandwidth, device=device, synchronize=False
    )
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - start
    return densities.detach().cpu().numpy(), elapsed


def _cpu_empirical_sd_kde(
    train: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    x_emp, h_emp = one_step_debiased_data_emp_kde(train)
    densities = kde_pdf_eval(queries, x_emp, h_emp)
    elapsed = time.perf_counter() - start
    return densities, elapsed


def _gpu_empirical_sd_kde(
    train: np.ndarray,
    queries: np.ndarray,
    *,
    device: str,
    warmup_done: bool,
) -> tuple[np.ndarray, float]:
    torch_device = torch.device(device)
    if not warmup_done:
        warm_x, warm_h = empirical_sd_kde_triton(
            train, device=device, return_tensor=True, synchronize=True
        )
        gaussian_kde_triton(
            warm_x, queries, warm_h, device=device, synchronize=True
        )

    torch.cuda.synchronize(torch_device)
    start = time.perf_counter()
    x_emp_gpu, h_emp = empirical_sd_kde_triton(
        train, device=device, return_tensor=True, synchronize=False
    )
    densities = gaussian_kde_triton(
        x_emp_gpu, queries, h_emp, device=device, synchronize=False
    )
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - start
    return densities.detach().cpu().numpy(), elapsed


def run_benchmark(
    mixture_index: int,
    seeds: Sequence[int],
    n_train: int,
    n_test: int,
    *,
    device: str = "cuda",
    enable_gpu: bool = True,
    enable_sklearn: bool = True,
    enable_emp_gpu: bool = True,
):
    params = mixture_params_list[mixture_index]
    timings_cpu = []
    timings_gpu = []
    timings_sklearn = []
    timings_emp_cpu = []
    timings_emp_gpu = []
    deltas_gpu = []
    deltas_sklearn = []
    deltas_emp_gpu = []

    warmup_done = False
    emp_warmup_done = False

    for seed in seeds:
        np.random.seed(seed)
        train = sample_from_mixture(n_train, params)

        np.random.seed(seed + 10_000)
        queries = sample_from_mixture(n_test, params)

        bandwidth = silverman_bandwidth(train)

        cpu_vals, cpu_time = _cpu_kde(train, queries, bandwidth)
        timings_cpu.append(cpu_time)
        seed_msgs = [f"Seed {seed}: CPU={cpu_time*1e3:.3f} ms"]

        sklearn_time = None
        if enable_sklearn:
            sklearn_vals, sklearn_time = _sklearn_kde(train, queries, bandwidth)
            timings_sklearn.append(sklearn_time)
            sk_max = np.max(np.abs(cpu_vals - sklearn_vals))
            sk_mean = np.mean(np.abs(cpu_vals - sklearn_vals))
            sk_rel = np.linalg.norm(cpu_vals - sklearn_vals) / (
                np.linalg.norm(cpu_vals) + 1e-12
            )
            deltas_sklearn.append((sk_max, sk_mean, sk_rel))
            seed_msgs.append(
                f"sklearn={sklearn_time*1e3:.3f} ms (Δmax={sk_max:.3e}, rel-L2={sk_rel:.3e})"
            )

        if enable_gpu:
            gpu_vals, gpu_time = _gpu_kde(
                train, queries, bandwidth, device=device, warmup_done=warmup_done
            )
            warmup_done = True

            gpu_max = np.max(np.abs(cpu_vals - gpu_vals))
            gpu_mean = np.mean(np.abs(cpu_vals - gpu_vals))
            gpu_rel = np.linalg.norm(cpu_vals - gpu_vals) / (
                np.linalg.norm(cpu_vals) + 1e-12
            )

            timings_gpu.append(gpu_time)
            deltas_gpu.append((gpu_max, gpu_mean, gpu_rel))
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            seed_msgs.append(
                f"GPU={gpu_time*1e3:.3f} ms (speedup={speedup:.2f}x, Δmax={gpu_max:.3e}, rel-L2={gpu_rel:.3e})"
            )

        emp_cpu_vals, emp_cpu_time = _cpu_empirical_sd_kde(train, queries)
        timings_emp_cpu.append(emp_cpu_time)
        seed_msgs.append(f"EmpCPU={emp_cpu_time*1e3:.3f} ms")

        if enable_emp_gpu:
            emp_gpu_vals, emp_gpu_time = _gpu_empirical_sd_kde(
                train, queries, device=device, warmup_done=emp_warmup_done
            )
            emp_warmup_done = True
            timings_emp_gpu.append(emp_gpu_time)
            emp_max = np.max(np.abs(emp_cpu_vals - emp_gpu_vals))
            emp_mean = np.mean(np.abs(emp_cpu_vals - emp_gpu_vals))
            emp_rel = np.linalg.norm(emp_cpu_vals - emp_gpu_vals) / (
                np.linalg.norm(emp_cpu_vals) + 1e-12
            )
            deltas_emp_gpu.append((emp_max, emp_mean, emp_rel))
            emp_speedup_sklearn = (
                (sklearn_time / emp_gpu_time)
                if (sklearn_time is not None and emp_gpu_time > 0)
                else None
            )
            speedup_str = (
                f", speedup vs sklearn={emp_speedup_sklearn:.2f}x"
                if emp_speedup_sklearn is not None
                else ", speedup vs sklearn=N/A"
            )
            seed_msgs.append(
                f"EmpGPU={emp_gpu_time*1e3:.3f} ms (Δmax={emp_max:.3e}, rel-L2={emp_rel:.3e}{speedup_str})"
            )

        print(" | ".join(seed_msgs))

    avg_cpu = np.mean(timings_cpu)
    avg_emp_cpu = np.mean(timings_emp_cpu)
    print("-" * 80)
    seed_count = len(seeds)
    print(
        f"(n_train={n_train}, n_test={n_test}) Silverman CPU KDE (avg over {seed_count} seeds): {avg_cpu*1e3:.3f} ms"
    )

    avg_sklearn = None
    if enable_sklearn and timings_sklearn:
        avg_sklearn = np.mean(timings_sklearn)
        avg_max_sk = np.mean([d[0] for d in deltas_sklearn])
        avg_mean_sk = np.mean([d[1] for d in deltas_sklearn])
        avg_rel_sk = np.mean([d[2] for d in deltas_sklearn])
        print(
            f"(n_train={n_train}, n_test={n_test}) sklearn KDE (avg over {seed_count} seeds): {avg_sklearn*1e3:.3f} ms "
            f"(Δmax={avg_max_sk:.3e}, mean Δ={avg_mean_sk:.3e}, rel-L2={avg_rel_sk:.3e})"
        )

    if enable_gpu and timings_gpu:
        avg_gpu = np.mean(timings_gpu)
        avg_speedup = avg_cpu / avg_gpu if avg_gpu > 0 else float("inf")
        avg_max = np.mean([d[0] for d in deltas_gpu])
        avg_mean = np.mean([d[1] for d in deltas_gpu])
        avg_rel = np.mean([d[2] for d in deltas_gpu])

        print(
            f"(n_train={n_train}, n_test={n_test}) Silverman GPU KDE (avg over {seed_count} seeds): {avg_gpu*1e3:.3f} ms "
            f"(speedup={avg_speedup:.2f}x, Δmax={avg_max:.3e}, mean Δ={avg_mean:.3e}, rel-L2={avg_rel:.3e})"
        )
    elif enable_gpu:
        print("Silverman GPU KDE: skipped (no timings collected).")

    print(
        f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE CPU (avg over {seed_count} seeds): {avg_emp_cpu*1e3:.3f} ms"
    )

    if enable_emp_gpu and timings_emp_gpu:
        avg_emp_gpu = np.mean(timings_emp_gpu)
        avg_emp_max = np.mean([d[0] for d in deltas_emp_gpu])
        avg_emp_mean = np.mean([d[1] for d in deltas_emp_gpu])
        avg_emp_rel = np.mean([d[2] for d in deltas_emp_gpu])
        summary = (
            f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE GPU (avg over {seed_count} seeds): {avg_emp_gpu*1e3:.3f} ms "
            f"(Δmax={avg_emp_max:.3e}, mean Δ={avg_emp_mean:.3e}, rel-L2={avg_emp_rel:.3e})"
        )
        if avg_sklearn is not None and avg_emp_gpu > 0:
            sk_speedup = avg_sklearn / avg_emp_gpu
            summary += f" | speedup vs sklearn: {sk_speedup:.2f}x"
        else:
            summary += " | speedup vs sklearn: N/A"
        print(summary)
    elif enable_emp_gpu:
        print("Empirical SD-KDE GPU: skipped (no timings collected).")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Triton KDE with the Silverman baseline."
    )
    parser.add_argument("--mixture-index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip the Triton GPU run (useful when CUDA is unavailable).",
    )
    parser.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Skip the scikit-learn KDE baseline.",
    )
    parser.add_argument(
        "--skip-emp-gpu",
        action="store_true",
        help="Skip the GPU empirical SD-KDE baseline.",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not args.cpu_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for the Triton KDE benchmark.")

    print(
        f"Running benchmark on mixture {args.mixture_index} with "
        f"{len(seeds)} seeds, n_train={args.n_train}, n_test={args.n_test} "
        f"using device {args.device}."
    )
    run_benchmark(
        mixture_index=args.mixture_index,
        seeds=seeds,
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
        enable_gpu=not args.cpu_only,
        enable_sklearn=not args.skip_sklearn,
        enable_emp_gpu=(not args.skip_emp_gpu) and (not args.cpu_only),
    )


if __name__ == "__main__":
    main()

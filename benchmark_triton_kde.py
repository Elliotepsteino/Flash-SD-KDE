"""Benchmark Triton KDE vs the existing Silverman implementation."""

from __future__ import annotations

import argparse
import time
from typing import List, Sequence
import math

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


def _gpu_kde_torch_naive(
    train: np.ndarray,
    queries: np.ndarray,
    bandwidth: float,
    *,
    device: str,
) -> tuple[np.ndarray, float]:
    """Non-tiled PyTorch implementation on GPU, mirroring the basic numpy KDE."""
    torch_device = torch.device(device)
    x = torch.as_tensor(queries, device=torch_device, dtype=torch.float32)
    d = torch.as_tensor(train, device=torch_device, dtype=torch.float32)
    h = float(bandwidth)
    inv_h = 1.0 / h
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    torch.cuda.synchronize(torch_device)
    start = time.perf_counter()
    z = (x[:, None] - d[None, :]) * inv_h
    pdf_mat = torch.exp(-0.5 * z * z) * inv_sqrt_2pi
    densities = pdf_mat.mean(dim=1) / h
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - start
    return densities.detach().cpu().numpy(), elapsed


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


def _gpu_empirical_sd_kde_torch_naive(
    train: np.ndarray,
    queries: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, float]:
    """Empirical SD-KDE using straightforward PyTorch broadcasting on the GPU."""
    torch_device = torch.device(device)
    x = torch.as_tensor(train, device=torch_device, dtype=torch.float32)
    q = torch.as_tensor(queries, device=torch_device, dtype=torch.float32)

    n = x.numel()
    if n == 0:
        raise ValueError("train must contain at least one element.")

    # Bandwidth and step-size as in the numpy empirical version
    x_np = train.astype(np.float32, copy=False)
    std_dev = float(np.std(x_np))
    iqr = float(np.percentile(x_np, 75) - np.percentile(x_np, 25))
    sigma = min(std_dev, iqr / 1.34)
    h = float(0.4 * sigma * n ** (-1 / 9))
    delta = 0.5 * h * h

    inv_h = 1.0 / h
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    torch.cuda.synchronize(torch_device)
    start = time.perf_counter()

    # Full pairwise z_{ij} = (x_i - x_j)/h on the GPU
    diff = (x[:, None] - x[None, :]) * inv_h  # (n, n)
    phi = torch.exp(-0.5 * diff * diff)

    pdf_acc = phi.sum(dim=1)
    deriv_acc = (-diff * phi).sum(dim=1)

    scale_pdf = inv_sqrt_2pi / (n * h)
    scale_deriv = inv_sqrt_2pi / (n * h * h)
    pdf_vals = scale_pdf * pdf_acc
    deriv_vals = scale_deriv * deriv_acc
    score = deriv_vals / (pdf_vals + 1e-12)

    x_deb = x + delta * score

    # KDE on debiased samples at query points
    z = (q[:, None] - x_deb[None, :]) * inv_h
    pdf_mat = torch.exp(-0.5 * z * z) * inv_sqrt_2pi
    densities = pdf_mat.mean(dim=1) / h

    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - start
    return densities.detach().cpu().numpy(), elapsed


def _gpu_empirical_sd_kde_torch_optimized(
    train: np.ndarray,
    queries: np.ndarray,
    *,
    device: str,
    block_size: int = 4096,
) -> tuple[np.ndarray, float]:
    """Empirical SD-KDE using a blocked Torch implementation (avoids full n×n)."""
    torch_device = torch.device(device)
    x = torch.as_tensor(train, device=torch_device, dtype=torch.float32)
    q = torch.as_tensor(queries, device=torch_device, dtype=torch.float32)

    n = x.numel()
    if n == 0:
        raise ValueError("train must contain at least one element.")

    x_np = train.astype(np.float32, copy=False)
    std_dev = float(np.std(x_np))
    iqr = float(np.percentile(x_np, 75) - np.percentile(x_np, 25))
    sigma = min(std_dev, iqr / 1.34)
    h = float(0.4 * sigma * n ** (-1 / 9))
    delta = 0.5 * h * h

    inv_h = 1.0 / h
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    pdf_acc = torch.zeros_like(x)
    deriv_acc = torch.zeros_like(x)

    torch.cuda.synchronize(torch_device)
    start = time.perf_counter()

    # Blocked accumulation of phi(z) and -z*phi(z) for the score
    for start_idx in range(0, n, block_size):
        db = x[start_idx:start_idx + block_size]  # (B,)
        diff = (x[:, None] - db[None, :]) * inv_h  # (n, B)
        phi = torch.exp(-0.5 * diff * diff)
        pdf_acc += phi.sum(dim=1)
        deriv_acc += (-diff * phi).sum(dim=1)

    scale_pdf = inv_sqrt_2pi / (n * h)
    scale_deriv = inv_sqrt_2pi / (n * h * h)
    pdf_vals = scale_pdf * pdf_acc
    deriv_vals = scale_deriv * deriv_acc
    score = deriv_vals / (pdf_vals + 1e-12)

    x_deb = x + delta * score

    # KDE on debiased samples at query points (full m×n is fine)
    z = (q[:, None] - x_deb[None, :]) * inv_h
    pdf_mat = torch.exp(-0.5 * z * z) * inv_sqrt_2pi
    densities = pdf_mat.mean(dim=1) / h

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
    timings_emp_gpu_triton = []
    timings_emp_gpu_torch_naive = []
    timings_emp_gpu_torch_opt = []
    timings_torch_naive = []
    deltas_gpu = []
    deltas_sklearn = []
    deltas_emp_gpu_triton = []
    deltas_emp_gpu_torch_naive = []
    deltas_emp_gpu_torch_opt = []
    deltas_torch_naive = []

    warmup_done = False
    emp_warmup_done = False
    torch_warmup_done = False
    emp_torch_naive_warmup_done = False
    emp_torch_opt_warmup_done = False

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

            if not torch_warmup_done:
                _gpu_kde_torch_naive(train, queries, bandwidth, device=device)
                torch_warmup_done = True
            torch_naive_vals, torch_naive_time = _gpu_kde_torch_naive(
                train, queries, bandwidth, device=device
            )
            timings_torch_naive.append(torch_naive_time)
            tn_max = np.max(np.abs(cpu_vals - torch_naive_vals))
            tn_mean = np.mean(np.abs(cpu_vals - torch_naive_vals))
            tn_rel = np.linalg.norm(cpu_vals - torch_naive_vals) / (
                np.linalg.norm(cpu_vals) + 1e-12
            )
            deltas_torch_naive.append((tn_max, tn_mean, tn_rel))
            tn_speedup = cpu_time / torch_naive_time if torch_naive_time > 0 else float("inf")
            seed_msgs.append(
                f"TorchGPU={torch_naive_time*1e3:.3f} ms (speedup={tn_speedup:.2f}x, Δmax={tn_max:.3e}, rel-L2={tn_rel:.3e})"
            )

        emp_cpu_vals, emp_cpu_time = _cpu_empirical_sd_kde(train, queries)
        timings_emp_cpu.append(emp_cpu_time)
        seed_msgs.append(f"EmpCPU={emp_cpu_time*1e3:.3f} ms")

        if enable_emp_gpu:
            emp_gpu_vals_triton, emp_gpu_time_triton = _gpu_empirical_sd_kde(
                train, queries, device=device, warmup_done=emp_warmup_done
            )
            emp_warmup_done = True
            timings_emp_gpu_triton.append(emp_gpu_time_triton)
            emp_max = np.max(np.abs(emp_cpu_vals - emp_gpu_vals_triton))
            emp_mean = np.mean(np.abs(emp_cpu_vals - emp_gpu_vals_triton))
            emp_rel = np.linalg.norm(emp_cpu_vals - emp_gpu_vals_triton) / (
                np.linalg.norm(emp_cpu_vals) + 1e-12
            )
            deltas_emp_gpu_triton.append((emp_max, emp_mean, emp_rel))
            emp_speedup_sklearn = (
                (sklearn_time / emp_gpu_time_triton)
                if (sklearn_time is not None and emp_gpu_time_triton > 0)
                else None
            )
            speedup_str = (
                f", speedup vs sklearn={emp_speedup_sklearn:.2f}x"
                if emp_speedup_sklearn is not None
                else ", speedup vs sklearn=N/A"
            )
            seed_msgs.append(
                f"EmpGPU(Triton)={emp_gpu_time_triton*1e3:.3f} ms (Δmax={emp_max:.3e}, rel-L2={emp_rel:.3e}{speedup_str})"
            )

            # Torch empirical SD-KDE (naive n×n)
            if not emp_torch_naive_warmup_done:
                _gpu_empirical_sd_kde_torch_naive(train, queries, device=device)
                emp_torch_naive_warmup_done = True
            emp_gpu_vals_torch_naive, emp_gpu_time_torch_naive = _gpu_empirical_sd_kde_torch_naive(
                train, queries, device=device
            )
            timings_emp_gpu_torch_naive.append(emp_gpu_time_torch_naive)
            emp_t_max = np.max(np.abs(emp_cpu_vals - emp_gpu_vals_torch_naive))
            emp_t_mean = np.mean(np.abs(emp_cpu_vals - emp_gpu_vals_torch_naive))
            emp_t_rel = np.linalg.norm(emp_cpu_vals - emp_gpu_vals_torch_naive) / (
                np.linalg.norm(emp_cpu_vals) + 1e-12
            )
            deltas_emp_gpu_torch_naive.append((emp_t_max, emp_t_mean, emp_t_rel))
            emp_speedup_sklearn_torch = (
                (sklearn_time / emp_gpu_time_torch_naive)
                if (sklearn_time is not None and emp_gpu_time_torch_naive > 0)
                else None
            )
            speedup_str_torch = (
                f", speedup vs sklearn={emp_speedup_sklearn_torch:.2f}x"
                if emp_speedup_sklearn_torch is not None
                else ", speedup vs sklearn=N/A"
            )
            seed_msgs.append(
                f"EmpGPU(Torch)={emp_gpu_time_torch_naive*1e3:.3f} ms (Δmax={emp_t_max:.3e}, rel-L2={emp_t_rel:.3e}{speedup_str_torch})"
            )

            # Torch empirical SD-KDE (optimized, blocked)
            if not emp_torch_opt_warmup_done:
                _gpu_empirical_sd_kde_torch_optimized(train, queries, device=device)
                emp_torch_opt_warmup_done = True
            emp_gpu_vals_torch_opt, emp_gpu_time_torch_opt = _gpu_empirical_sd_kde_torch_optimized(
                train, queries, device=device
            )
            timings_emp_gpu_torch_opt.append(emp_gpu_time_torch_opt)
            emp_to_max = np.max(np.abs(emp_cpu_vals - emp_gpu_vals_torch_opt))
            emp_to_mean = np.mean(np.abs(emp_cpu_vals - emp_gpu_vals_torch_opt))
            emp_to_rel = np.linalg.norm(emp_cpu_vals - emp_gpu_vals_torch_opt) / (
                np.linalg.norm(emp_cpu_vals) + 1e-12
            )
            deltas_emp_gpu_torch_opt.append((emp_to_max, emp_to_mean, emp_to_rel))
            emp_speedup_sklearn_torch_opt = (
                (sklearn_time / emp_gpu_time_torch_opt)
                if (sklearn_time is not None and emp_gpu_time_torch_opt > 0)
                else None
            )
            speedup_str_torch_opt = (
                f", speedup vs sklearn={emp_speedup_sklearn_torch_opt:.2f}x"
                if emp_speedup_sklearn_torch_opt is not None
                else ", speedup vs sklearn=N/A"
            )
            seed_msgs.append(
                f"EmpGPU(TorchOpt)={emp_gpu_time_torch_opt*1e3:.3f} ms (Δmax={emp_to_max:.3e}, rel-L2={emp_to_rel:.3e}{speedup_str_torch_opt})"
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

    if enable_gpu and timings_torch_naive:
        avg_tn = np.mean(timings_torch_naive)
        avg_tn_speedup = avg_cpu / avg_tn if avg_tn > 0 else float("inf")
        avg_tn_max = np.mean([d[0] for d in deltas_torch_naive])
        avg_tn_mean = np.mean([d[1] for d in deltas_torch_naive])
        avg_tn_rel = np.mean([d[2] for d in deltas_torch_naive])
        print(
            f"(n_train={n_train}, n_test={n_test}) Torch GPU KDE (avg over {seed_count} seeds): {avg_tn*1e3:.3f} ms "
            f"(speedup={avg_tn_speedup:.2f}x, Δmax={avg_tn_max:.3e}, mean Δ={avg_tn_mean:.3e}, rel-L2={avg_tn_rel:.3e})"
        )

    print(
        f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE CPU (avg over {seed_count} seeds): {avg_emp_cpu*1e3:.3f} ms"
    )

    if enable_emp_gpu and timings_emp_gpu_triton:
        avg_emp_gpu_triton = np.mean(timings_emp_gpu_triton)
        avg_emp_max_triton = np.mean([d[0] for d in deltas_emp_gpu_triton])
        avg_emp_mean_triton = np.mean([d[1] for d in deltas_emp_gpu_triton])
        avg_emp_rel_triton = np.mean([d[2] for d in deltas_emp_gpu_triton])
        summary_triton = (
            f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE GPU (Triton, avg over {seed_count} seeds): {avg_emp_gpu_triton*1e3:.3f} ms "
            f"(Δmax={avg_emp_max_triton:.3e}, mean Δ={avg_emp_mean_triton:.3e}, rel-L2={avg_emp_rel_triton:.3e})"
        )
        if avg_sklearn is not None and avg_emp_gpu_triton > 0:
            sk_speedup = avg_sklearn / avg_emp_gpu_triton
            summary_triton += f" | speedup vs sklearn: {sk_speedup:.2f}x"
        else:
            summary_triton += " | speedup vs sklearn: N/A"
        print(summary_triton)

    if enable_emp_gpu and timings_emp_gpu_torch_naive:
        avg_emp_gpu_torch = np.mean(timings_emp_gpu_torch_naive)
        avg_emp_max_torch = np.mean([d[0] for d in deltas_emp_gpu_torch_naive])
        avg_emp_mean_torch = np.mean([d[1] for d in deltas_emp_gpu_torch_naive])
        avg_emp_rel_torch = np.mean([d[2] for d in deltas_emp_gpu_torch_naive])
        summary_torch = (
            f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE GPU (Torch, avg over {seed_count} seeds): {avg_emp_gpu_torch*1e3:.3f} ms "
            f"(Δmax={avg_emp_max_torch:.3e}, mean Δ={avg_emp_mean_torch:.3e}, rel-L2={avg_emp_rel_torch:.3e})"
        )
        if avg_sklearn is not None and avg_emp_gpu_torch > 0:
            sk_speedup_torch = avg_sklearn / avg_emp_gpu_torch
            summary_torch += f" | speedup vs sklearn: {sk_speedup_torch:.2f}x"
        else:
            summary_torch += " | speedup vs sklearn: N/A"
        print(summary_torch)

    if enable_emp_gpu and timings_emp_gpu_torch_opt:
        avg_emp_gpu_torch_opt = np.mean(timings_emp_gpu_torch_opt)
        avg_emp_max_torch_opt = np.mean([d[0] for d in deltas_emp_gpu_torch_opt])
        avg_emp_mean_torch_opt = np.mean([d[1] for d in deltas_emp_gpu_torch_opt])
        avg_emp_rel_torch_opt = np.mean([d[2] for d in deltas_emp_gpu_torch_opt])
        summary_torch_opt = (
            f"(n_train={n_train}, n_test={n_test}) Empirical SD-KDE GPU (TorchOpt, avg over {seed_count} seeds): {avg_emp_gpu_torch_opt*1e3:.3f} ms "
            f"(Δmax={avg_emp_max_torch_opt:.3e}, mean Δ={avg_emp_mean_torch_opt:.3e}, rel-L2={avg_emp_rel_torch_opt:.3e})"
        )
        if avg_sklearn is not None and avg_emp_gpu_torch_opt > 0:
            sk_speedup_torch_opt = avg_sklearn / avg_emp_gpu_torch_opt
            summary_torch_opt += f" | speedup vs sklearn: {sk_speedup_torch_opt:.2f}x"
        else:
            summary_torch_opt += " | speedup vs sklearn: N/A"
        print(summary_torch_opt)

    if enable_emp_gpu and not (timings_emp_gpu_triton or timings_emp_gpu_torch_naive or timings_emp_gpu_torch_opt):
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

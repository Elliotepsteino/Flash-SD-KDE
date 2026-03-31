"""Benchmark rebuttal Figure 1 runtimes for 16-D KDE / SD-KDE.

This sweep measures five methods across powers of two in ``n_train``:
  - sklearn KDE
  - SD-KDE (Torch)
  - SD-KDE (Torch compile)
  - SD-KDE (PyKeOps)
  - Flash-SD-KDE

For each point, ``n_test = n_train / 8`` and the bandwidth is chosen via the
same 16-D Silverman rule used elsewhere in the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from sklearn.neighbors import KernelDensity
except Exception:  # pragma: no cover - optional dependency
    KernelDensity = None

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd


def _require_pykeops():
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pykeops is not installed or importable. Install `pykeops` to run this benchmark."
        ) from exc
    return LazyTensor


def _torch_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    q_norm = (queries * queries).sum(dim=1, keepdim=True)
    d_norm = (train * train).sum(dim=1, keepdim=True).T
    dot = queries @ train.T
    dist = torch.clamp(q_norm + d_norm - 2.0 * dot, min=0.0)
    phi = torch.exp(-0.5 * dist * inv_h2)
    dim = train.shape[1]
    norm = 1.0 / (((2.0 * math.pi) ** (dim / 2.0)) * (bandwidth ** dim) * train.shape[0])
    return norm * phi.sum(dim=1)


def _torch_empirical_sd_kde_nd(
    train: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    x_norm = (train * train).sum(dim=1, keepdim=True)
    gram = train @ train.T
    dist = torch.clamp(x_norm + x_norm.T - 2.0 * gram, min=0.0)
    phi = torch.exp(-0.5 * dist * inv_h2)
    phi_sum = phi.sum(dim=1, keepdim=True)
    weighted = phi @ train
    score = (weighted / (phi_sum + 1e-12) - train) * inv_h2
    return train + 0.5 * (bandwidth ** 2) * score


def _torch_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    debiased = _torch_empirical_sd_kde_nd(train, bandwidth)
    return _torch_kde_nd(debiased, queries, bandwidth)


def _torch_compile_available() -> bool:
    return hasattr(torch, "compile")


def _pykeops_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    LazyTensor = _require_pykeops()
    x_i = LazyTensor(queries[:, None, :])
    y_j = LazyTensor(train[None, :, :])
    d_ij = ((x_i - y_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()
    n_train = float(train.shape[0])
    dim = float(train.shape[1])
    norm = (1.0 / bandwidth) ** dim
    norm /= ((2.0 * math.pi) ** (dim / 2.0)) * n_train
    return (norm * k_ij.sum(dim=1)).view(-1)


def _pykeops_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    LazyTensor = _require_pykeops()
    x_i = LazyTensor(train[:, None, :])
    x_j = LazyTensor(train[None, :, :])
    d_ij = ((x_i - x_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()
    pdf_sum = k_ij.sum(dim=1)
    weighted_sum = (k_ij * x_j).sum(dim=1)
    score = (weighted_sum / (pdf_sum + 1e-12) - train) * (1.0 / (bandwidth * bandwidth))
    debiased = train + 0.5 * (bandwidth ** 2) * score
    return _pykeops_kde_nd(debiased, queries, bandwidth)


def _flash_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    debiased, _ = empirical_sd_kde_triton_nd(
        train,
        bandwidth,
        device=device,
        return_tensor=True,
        synchronize=False,
    )
    return gaussian_kde_triton_nd(
        debiased,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
    )


def _time_cpu_ms(
    fn: Callable[[], np.ndarray],
    *,
    repeats: int,
) -> tuple[float, float, float, np.ndarray]:
    values = None
    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        values = fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    assert values is not None
    arr = np.asarray(times_ms, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), values


def _time_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, np.ndarray]:
    for _ in range(max(warmup, 0)):
        _ = fn()
        torch.cuda.synchronize(device)

    values = None
    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        values = fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    assert values is not None
    arr = np.asarray(times_ms, dtype=float)
    return (
        float(arr.mean()),
        float(arr.std(ddof=0)),
        float(arr.min()),
        values.detach().cpu().numpy().ravel(),
    )


def _sklearn_kde_nd(
    train: np.ndarray,
    queries: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    if KernelDensity is None:
        raise RuntimeError("scikit-learn is not installed; cannot run sklearn KDE.")
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(train)
    return np.exp(kde.score_samples(queries)).astype(np.float32, copy=False)


def benchmark_point(
    *,
    n_train: int,
    n_test: int,
    device: str,
    seed: int,
    warmup: int,
    flash_repeats: int,
    baseline_repeats: int,
) -> dict[str, float]:
    if KernelDensity is None:
        raise RuntimeError("scikit-learn is required for this benchmark.")
    _require_pykeops()

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run this benchmark.")

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()
    queries_t = torch.as_tensor(queries_np, device=torch_device, dtype=torch.float32).contiguous()

    sk_mean, sk_std, sk_min, _ = _time_cpu_ms(
        lambda: _sklearn_kde_nd(train_np, queries_np, bandwidth),
        repeats=baseline_repeats,
    )
    torch_mean, torch_std, torch_min, torch_vals = _time_cuda_ms(
        lambda: _torch_sd_kde_nd(train_t, queries_t, bandwidth),
        device=torch_device,
        warmup=warmup,
        repeats=baseline_repeats,
    )
    if _torch_compile_available():
        compiled_impl = torch.compile(
            lambda x, q: _torch_sd_kde_nd(x, q, bandwidth),
            mode="reduce-overhead",
            fullgraph=False,
        )
        compile_mean, compile_std, compile_min, compile_vals = _time_cuda_ms(
            lambda: compiled_impl(train_t, queries_t),
            device=torch_device,
            warmup=warmup,
            repeats=baseline_repeats,
        )
    else:
        compile_mean = float("nan")
        compile_std = float("nan")
        compile_min = float("nan")
        compile_vals = torch_vals
    pykeops_mean, pykeops_std, pykeops_min, pykeops_vals = _time_cuda_ms(
        lambda: _pykeops_sd_kde_nd(train_t, queries_t, bandwidth),
        device=torch_device,
        warmup=warmup,
        repeats=baseline_repeats,
    )
    flash_mean, flash_std, flash_min, flash_vals = _time_cuda_ms(
        lambda: _flash_sd_kde_nd(train_t, queries_t, bandwidth, device=torch_device),
        device=torch_device,
        warmup=warmup,
        repeats=flash_repeats,
    )

    delta_flash_torch = float(np.max(np.abs(flash_vals - torch_vals)))
    delta_flash_compile = float(np.max(np.abs(flash_vals - compile_vals)))
    delta_flash_pykeops = float(np.max(np.abs(flash_vals - pykeops_vals)))
    rel_l2_flash_torch = float(
        np.linalg.norm(flash_vals - torch_vals) / (np.linalg.norm(torch_vals) + 1e-12)
    )
    rel_l2_flash_compile = float(
        np.linalg.norm(flash_vals - compile_vals) / (np.linalg.norm(compile_vals) + 1e-12)
    )
    rel_l2_flash_pykeops = float(
        np.linalg.norm(flash_vals - pykeops_vals) / (np.linalg.norm(pykeops_vals) + 1e-12)
    )

    row = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "seed": float(seed),
        "bandwidth": bandwidth,
        "sklearn_kde_ms": sk_mean,
        "sklearn_kde_ms_std": sk_std,
        "sklearn_kde_ms_min": sk_min,
        "sd_torch_ms": torch_mean,
        "sd_torch_ms_std": torch_std,
        "sd_torch_ms_min": torch_min,
        "sd_torch_compile_ms": compile_mean,
        "sd_torch_compile_ms_std": compile_std,
        "sd_torch_compile_ms_min": compile_min,
        "sd_pykeops_ms": pykeops_mean,
        "sd_pykeops_ms_std": pykeops_std,
        "sd_pykeops_ms_min": pykeops_min,
        "flash_sd_kde_ms": flash_mean,
        "flash_sd_kde_ms_std": flash_std,
        "flash_sd_kde_ms_min": flash_min,
        "speedup_flash_vs_torch": (torch_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "speedup_flash_vs_torch_compile": (compile_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "speedup_flash_vs_pykeops": (pykeops_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "speedup_flash_vs_sklearn": (sk_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "delta_max_flash_vs_torch": delta_flash_torch,
        "delta_max_flash_vs_torch_compile": delta_flash_compile,
        "delta_max_flash_vs_pykeops": delta_flash_pykeops,
        "rel_l2_flash_vs_torch": rel_l2_flash_torch,
        "rel_l2_flash_vs_torch_compile": rel_l2_flash_compile,
        "rel_l2_flash_vs_pykeops": rel_l2_flash_pykeops,
    }

    print(
        f"[Rebuttal Fig1 16D] n_train={n_train}, n_test={n_test}, h={bandwidth:.6e} | "
        f"sklearn KDE={sk_mean:.2f} ms | SD-KDE Torch={torch_mean:.2f} ms | "
        f"SD-KDE Torch compile={compile_mean:.2f} ms | "
        f"SD-KDE PyKeOps={pykeops_mean:.2f} ms | Flash-SD-KDE={flash_mean:.2f} ms"
    )
    print(
        "  Flash agreement:"
        f" Torch Δmax={delta_flash_torch:.3e}, rel-L2={rel_l2_flash_torch:.3e} |"
        f" Torch compile Δmax={delta_flash_compile:.3e}, rel-L2={rel_l2_flash_compile:.3e} |"
        f" PyKeOps Δmax={delta_flash_pykeops:.3e}, rel-L2={rel_l2_flash_pykeops:.3e}"
    )
    return row


def benchmark_sweep(
    *,
    start_power: int,
    end_power: int,
    device: str,
    seed: int,
    warmup: int,
    flash_repeats: int,
    baseline_repeats: int,
) -> dict[str, object]:
    rows = []
    for power in range(start_power, end_power + 1):
        n_train = 1 << power
        n_test = max(1, n_train // 8)
        rows.append(
            benchmark_point(
                n_train=n_train,
                n_test=n_test,
                device=device,
                seed=seed,
                warmup=warmup,
                flash_repeats=flash_repeats,
                baseline_repeats=baseline_repeats,
            )
        )
    return {
        "figure": "rebuttal_figure1_16d_runtime",
        "device": device,
        "seed": seed,
        "start_power": start_power,
        "end_power": end_power,
        "warmup": warmup,
        "flash_repeats": flash_repeats,
        "baseline_repeats": baseline_repeats,
        "rows": rows,
    }


def format_markdown_table(payload: dict[str, object]) -> str:
    rows = list(payload["rows"])
    header = [
        "| n_train | n_test | sklearn KDE (ms) | SD-KDE (Torch) (ms) | SD-KDE (Torch compile) (ms) | SD-KDE (PyKeOps) (ms) | Flash-SD-KDE (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = [
        (
            f"| {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{row['sklearn_kde_ms']:.2f} | {row['sd_torch_ms']:.2f} | "
            f"{row['sd_torch_compile_ms']:.2f} | "
            f"{row['sd_pykeops_ms']:.2f} | {row['flash_sd_kde_ms']:.2f} |"
        )
        for row in rows
    ]
    notes = [
        "",
        "Lower is better. Measurements use 16-D Gaussian-mixture data with "
        "`n_test = n_train / 8` and exclude one-time warmup / JIT costs.",
    ]
    return "\n".join(header + body + notes) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rebuttal Figure 1 16-D KDE / SD-KDE runtimes."
    )
    parser.add_argument("--start-power", type=int, default=11)
    parser.add_argument("--end-power", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--flash-repeats", type=int, default=10)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    args = parser.parse_args()

    payload = benchmark_sweep(
        start_power=args.start_power,
        end_power=args.end_power,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        flash_repeats=args.flash_repeats,
        baseline_repeats=args.baseline_repeats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote rebuttal Figure 1 benchmark JSON to {args.output}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown_table(payload), encoding="utf-8")
        print(f"Wrote rebuttal Figure 1 markdown table to {args.markdown_output}")


if __name__ == "__main__":
    main()

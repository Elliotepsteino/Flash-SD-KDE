"""Compare 16-D Flash-SD-KDE with PyKeOps KDE and SD-KDE runtimes.

This script times full 16-D Gaussian density evaluation using:
  - Flash-SD-KDE (Triton empirical debias + Triton KDE), and
  - PyKeOps LazyTensor KDE / SD-KDE implementations.
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

from flash_sd_kde.reference import silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd

_ND_FEATURES = 16


def _require_pykeops():
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pykeops is not installed or importable. Install `pykeops` to run this benchmark."
        ) from exc
    return LazyTensor


def _pykeops_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Gaussian KDE in d dimensions using PyKeOps LazyTensors (normalized)."""
    LazyTensor = _require_pykeops()

    x_i = LazyTensor(queries[:, None, :])  # (m, 1, d)
    y_j = LazyTensor(train[None, :, :])  # (1, n, d)

    d_ij = ((x_i - y_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()

    n_train = float(train.shape[0])
    dim = float(train.shape[1])
    norm = (1.0 / bandwidth) ** dim
    norm /= ((2.0 * math.pi) ** (dim / 2.0)) * n_train

    # KeOps reduction returns shape (m, 1); flatten to (m,).
    return (norm * k_ij.sum(dim=1)).view(-1)


def _pykeops_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Full SD-KDE using PyKeOps for both score computation and final KDE."""
    LazyTensor = _require_pykeops()

    x = train
    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])

    d_ij = ((x_i - x_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()

    pdf_sum = k_ij.sum(dim=1)
    weighted_sum = (k_ij * x_j).sum(dim=1)
    eps = 1e-12
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    score = (weighted_sum / (pdf_sum + eps) - x) * inv_h2
    debiased = x + 0.5 * (bandwidth * bandwidth) * score

    return _pykeops_kde_nd(debiased, queries, bandwidth)


def _time_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    for _ in range(max(warmup, 0)):
        _ = fn()
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(times_ms, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min())


def _format_table(summary: dict[str, float]) -> str:
    flash_ms = summary["flash_sd_kde_ms"]
    kde_ms = summary["pykeops_kde_ms"]
    sd_ms = summary["pykeops_sd_kde_ms"]
    rel_kde = kde_ms / flash_ms if flash_ms > 0 else float("inf")
    rel_sd = sd_ms / flash_ms if flash_ms > 0 else float("inf")
    return "\n".join(
        [
            "Method\tRuntime (ms)\tRel. to Flash-SD-KDE",
            f"16-D Flash-SD-KDE\t{flash_ms:.2f}\t1.00x",
            f"PyKeOps 16-D KDE\t{kde_ms:.2f}\t{rel_kde:.2f}x",
            f"PyKeOps 16-D SD-KDE\t{sd_ms:.2f}\t{rel_sd:.2f}x",
        ]
    )


def benchmark_pykeops_vs_flash_sd_kde_16d(
    *,
    n_train: int,
    n_test: int,
    device: str,
    seed: int,
    warmup: int,
    flash_repeats: int,
    baseline_repeats: int,
) -> dict[str, float]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run this benchmark.")

    print(
        f"[16D Flash-SD-KDE vs PyKeOps] n_train={n_train}, n_test={n_test}, "
        f"device={device}, dim={_ND_FEATURES}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = torch.randn(
        (n_train, _ND_FEATURES),
        device=torch_device,
        dtype=torch.float32,
    )
    x_test = torch.randn(
        (n_test, _ND_FEATURES),
        device=torch_device,
        dtype=torch.float32,
    )

    bw = float(
        silverman_bandwidth_nd(
            x_train.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    )
    print(f"  Silverman bandwidth h={bw:.6e}")

    def run_flash_sd_kde() -> torch.Tensor:
        x_emp, _ = empirical_sd_kde_triton_nd(
            x_train,
            bw,
            device=torch_device,
            return_tensor=True,
            synchronize=False,
        )
        return gaussian_kde_triton_nd(
            x_emp,
            x_test,
            bw,
            device=torch_device,
            synchronize=False,
        )

    def run_pykeops_kde() -> torch.Tensor:
        return _pykeops_kde_nd(x_train, x_test, bw)

    def run_pykeops_sd_kde() -> torch.Tensor:
        return _pykeops_sd_kde_nd(x_train, x_test, bw)

    # Warm once in full pipeline order to include JIT/init before timed repeats.
    _ = run_flash_sd_kde()
    _ = run_pykeops_kde()
    pykeops_sd_values = run_pykeops_sd_kde()
    torch.cuda.synchronize(torch_device)

    flash_mean, flash_std, flash_min = _time_cuda_ms(
        run_flash_sd_kde, device=torch_device, warmup=warmup, repeats=flash_repeats
    )
    kde_mean, kde_std, kde_min = _time_cuda_ms(
        run_pykeops_kde, device=torch_device, warmup=warmup, repeats=baseline_repeats
    )
    sd_mean, sd_std, sd_min = _time_cuda_ms(
        run_pykeops_sd_kde, device=torch_device, warmup=warmup, repeats=baseline_repeats
    )

    flash_values = run_flash_sd_kde().detach().cpu().numpy().ravel()
    sd_values = pykeops_sd_values.detach().cpu().numpy().ravel()
    max_delta = float(np.max(np.abs(flash_values - sd_values)))
    rel_l2 = float(np.linalg.norm(flash_values - sd_values) / (np.linalg.norm(sd_values) + 1e-12))

    summary = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "seed": float(seed),
        "warmup": float(warmup),
        "flash_repeats": float(flash_repeats),
        "baseline_repeats": float(baseline_repeats),
        "bandwidth": bw,
        "flash_sd_kde_ms": flash_mean,
        "flash_sd_kde_ms_std": flash_std,
        "flash_sd_kde_ms_min": flash_min,
        "pykeops_kde_ms": kde_mean,
        "pykeops_kde_ms_std": kde_std,
        "pykeops_kde_ms_min": kde_min,
        "pykeops_sd_kde_ms": sd_mean,
        "pykeops_sd_kde_ms_std": sd_std,
        "pykeops_sd_kde_ms_min": sd_min,
        "speedup_flash_vs_pykeops_kde": (kde_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "speedup_flash_vs_pykeops_sd_kde": (sd_mean / flash_mean) if flash_mean > 0 else float("inf"),
        "delta_max_flash_vs_pykeops_sd_kde": max_delta,
        "rel_l2_flash_vs_pykeops_sd_kde": rel_l2,
    }

    print(f"  Flash-SD-KDE:       {flash_mean:8.2f} ms (std={flash_std:.2f}, min={flash_min:.2f})")
    print(f"  PyKeOps 16-D KDE:   {kde_mean:8.2f} ms (std={kde_std:.2f}, min={kde_min:.2f})")
    print(f"  PyKeOps 16-D SD-KDE:{sd_mean:8.2f} ms (std={sd_std:.2f}, min={sd_min:.2f})")
    print(
        "  Speedup Flash-SD-KDE vs PyKeOps:"
        f" KDE={summary['speedup_flash_vs_pykeops_kde']:.2f}x,"
        f" SD-KDE={summary['speedup_flash_vs_pykeops_sd_kde']:.2f}x"
    )
    print(f"  Δmax={max_delta:.3e}, rel-L2={rel_l2:.3e}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 16-D Flash-SD-KDE against PyKeOps KDE/SD-KDE."
    )
    parser.add_argument("--n-train", type=int, default=32768)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--flash-repeats", type=int, default=10)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Optional plain-text table output path.",
    )
    args = parser.parse_args()

    summary = benchmark_pykeops_vs_flash_sd_kde_16d(
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        flash_repeats=args.flash_repeats,
        baseline_repeats=args.baseline_repeats,
    )

    table_text = _format_table(summary)
    print("\n" + table_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote JSON summary to {args.output}")

    if args.table_output is not None:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(table_text + "\n", encoding="utf-8")
        print(f"Wrote table text to {args.table_output}")


if __name__ == "__main__":
    main()

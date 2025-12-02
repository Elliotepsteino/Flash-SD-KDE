"""Compare 16-D Triton KDE with a PyKeOps KDE implementation.

This script times the full Gaussian KDE evaluation in 16 dimensions using:
  - our Triton kernel (`gaussian_kde_triton_nd`), and
  - a PyKeOps LazyTensor implementation.

It is intended for quick, end-to-end runtime comparisons on a single GPU.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from kde_utils import silverman_bandwidth_nd
from triton_kde import (
    gaussian_kde_triton_nd,
    empirical_sd_kde_triton_nd,
    _ND_FEATURES,
)


def _pykeops_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Gaussian KDE in d dimensions using PyKeOps LazyTensors (normalized)."""
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pykeops is not installed or not importable; "
            "install `pykeops` in the flash-sd-kde environment to run this benchmark."
        ) from exc

    x = queries  # (m, d)
    y = train    # (n, d)
    h = float(bandwidth)

    x_i = LazyTensor(x[:, None, :])   # (m, 1, d)
    y_j = LazyTensor(y[None, :, :])   # (1, n, d)

    # Squared distances and Gaussian kernel with bandwidth h.
    D_ij = ((x_i - y_j) ** 2).sum(dim=2)
    K_ij = (-0.5 * D_ij / (h * h)).exp()

    # Match the multivariate Gaussian normalization used by gaussian_kde_triton_nd.
    n_train = float(train.shape[0])
    d = float(train.shape[1])
    inv_h = 1.0 / h
    norm = (inv_h ** d) / (((2.0 * math.pi) ** (d / 2.0)) * n_train)

    return norm * K_ij.sum(dim=1)  # (m, 1)


def _pykeops_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Full SD-KDE using PyKeOps both for score computation and final KDE."""
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:
        raise RuntimeError(
            "pykeops is not installed or not importable; "
            "install `pykeops` in the flash-sd-kde environment to run this benchmark."
        ) from exc

    x = train  # (n, d)
    h = float(bandwidth)

    x_i = LazyTensor(x[:, None, :])
    x_j = LazyTensor(x[None, :, :])

    D_ij = ((x_i - x_j) ** 2).sum(dim=2)
    K_ij = (-0.5 * D_ij / (h * h)).exp()

    n = float(train.shape[0])
    d = float(train.shape[1])
    inv_h = 1.0 / h
    norm_pdf = (inv_h ** d) / (((2.0 * math.pi) ** (d / 2.0)) * n)

    pdf_vals = norm_pdf * K_ij.sum(dim=1)
    weighted_sum = (K_ij * x_j).sum(dim=1)
    eps = 1e-12
    score = (weighted_sum / (K_ij.sum(dim=1) + eps) - x) * (inv_h * inv_h)
    delta = 0.5 * (h * h)
    debiased = x + delta * score

    return _pykeops_kde_nd(debiased, queries, h)


def benchmark_pykeops_vs_triton_nd(
    n_train: int,
    n_test: int,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run ND Triton/PyKeOps benchmark.")

    print(
        f"[16D KDE Triton vs PyKeOps] n_train={n_train}, n_test={n_test}, "
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

    # Bandwidth from Silverman's ND rule of thumb.
    bw = silverman_bandwidth_nd(x_train.detach().cpu().numpy().astype(np.float32, copy=False))
    print(f"  Using Silverman ND bandwidth h={bw:.4e}")

    print("  Warming up Triton SD-KDE...")
    deb_warm, _ = empirical_sd_kde_triton_nd(
        x_train,
        bw,
        device=torch_device,
        return_tensor=True,
        synchronize=True,
    )
    gaussian_kde_triton_nd(
        deb_warm,
        x_test,
        bw,
        device=torch_device,
        synchronize=True,
    )

    print("  Warming up PyKeOps KDE + SD-KDE...")
    kde_pykeops = _pykeops_kde_nd(x_train, x_test, bw)
    sd_pykeops = _pykeops_sd_kde_nd(x_train, x_test, bw)
    torch.cuda.synchronize(torch_device)

    torch.cuda.synchronize(torch_device)
    t0 = time.perf_counter()
    tri_deb, _ = empirical_sd_kde_triton_nd(
        x_train,
        bw,
        device=torch_device,
        return_tensor=True,
        synchronize=False,
    )
    densities_triton = gaussian_kde_triton_nd(
        tri_deb,
        x_test,
        bw,
        device=torch_device,
        synchronize=False,
    )
    torch.cuda.synchronize(torch_device)
    t_triton_sd = time.perf_counter() - t0

    torch.cuda.synchronize(torch_device)
    t0 = time.perf_counter()
    kde_pykeops = _pykeops_kde_nd(x_train, x_test, bw)
    torch.cuda.synchronize(torch_device)
    t_pykeops_kde = time.perf_counter() - t0

    torch.cuda.synchronize(torch_device)
    t0 = time.perf_counter()
    sd_pykeops = _pykeops_sd_kde_nd(x_train, x_test, bw)
    torch.cuda.synchronize(torch_device)
    t_pykeops_sd = time.perf_counter() - t0

    tri = densities_triton.detach().cpu().numpy().ravel()
    sd = sd_pykeops.detach().cpu().numpy().ravel()

    max_delta = float(np.max(np.abs(tri - sd)))
    rel_l2 = float(np.linalg.norm(tri - sd) / (np.linalg.norm(sd) + 1e-12))

    print(f"  Triton 16D SD-KDE: {t_triton_sd*1e3:8.2f} ms")
    print(f"  PyKeOps 16D KDE:   {t_pykeops_kde*1e3:8.2f} ms")
    print(f"  PyKeOps 16D SD-KDE:{t_pykeops_sd*1e3:8.2f} ms")
    speedup = t_pykeops_sd / t_triton_sd if t_triton_sd > 0 else float("inf")
    print(f"  Speedup Triton SD-KDE vs PyKeOps SD-KDE: {speedup:6.2f}x")
    print(f"  Δmax={max_delta:.3e}, rel-L2={rel_l2:.3e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 16-D Triton KDE with a PyKeOps KDE."
    )
    parser.add_argument("--n-train", type=int, default=32768)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark_pykeops_vs_triton_nd(
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

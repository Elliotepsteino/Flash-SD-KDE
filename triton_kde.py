"""Triton-based Gaussian KDE implementation for flash-sd-kde.

This module provides a GPU-accelerated normal (Gaussian) KDE evaluation
that mirrors the scalar Silverman-style estimator used elsewhere in the
repository.  The Triton kernel computes partial sums over tiles of the
training data and relies on atomic adds to accumulate the contributions
for each query point.  This keeps the kernel simple while remaining fast
enough for the current problem sizes (n up to a few hundred thousand).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _gaussian_kde_kernel(
    data_ptr,
    query_ptr,
    out_ptr,
    n_data,
    n_query,
    inv_bandwidth,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute Gaussian KDE contributions for a tile of queries vs data."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    query_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    data_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    query_mask = query_offsets < n_query
    data_mask = data_offsets < n_data

    query_vals = tl.load(query_ptr + query_offsets, mask=query_mask, other=0.0)
    data_vals = tl.load(data_ptr + data_offsets, mask=data_mask, other=0.0)

    query_vals = tl.reshape(query_vals, (BLOCK_M, 1))
    data_vals = tl.reshape(data_vals, (1, BLOCK_N))

    diff = (query_vals - data_vals) * inv_bandwidth
    inv_sqrt_2pi = 0.3989422804014327  # simple literal avoids global constexpr issues
    contrib = tl.exp(-0.5 * diff * diff) * inv_sqrt_2pi

    data_mask_matrix = tl.reshape(data_mask, (1, BLOCK_N))
    contrib = tl.where(data_mask_matrix, contrib, 0.0)

    block_sum = tl.sum(contrib, axis=1)

    tl.atomic_add(out_ptr + query_offsets, block_sum, mask=query_mask)


@triton.jit
def _gaussian_kde_pdf_deriv_kernel(
    data_ptr,
    query_ptr,
    pdf_ptr,
    deriv_ptr,
    n_data,
    n_query,
    inv_bandwidth,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate phi(z) and -z*phi(z) sums for KDE score evaluation."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    query_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    data_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    query_mask = query_offsets < n_query
    data_mask = data_offsets < n_data

    query_vals = tl.load(query_ptr + query_offsets, mask=query_mask, other=0.0)
    data_vals = tl.load(data_ptr + data_offsets, mask=data_mask, other=0.0)

    query_vals = tl.reshape(query_vals, (BLOCK_M, 1))
    data_vals = tl.reshape(data_vals, (1, BLOCK_N))

    diff = (query_vals - data_vals) * inv_bandwidth
    inv_sqrt_2pi = 0.3989422804014327
    phi = tl.exp(-0.5 * diff * diff) * inv_sqrt_2pi
    deriv = -diff * phi

    data_mask_matrix = tl.reshape(data_mask, (1, BLOCK_N))
    phi = tl.where(data_mask_matrix, phi, 0.0)
    deriv = tl.where(data_mask_matrix, deriv, 0.0)

    phi_sum = tl.sum(phi, axis=1)
    deriv_sum = tl.sum(deriv, axis=1)

    tl.atomic_add(pdf_ptr + query_offsets, phi_sum, mask=query_mask)
    tl.atomic_add(deriv_ptr + query_offsets, deriv_sum, mask=query_mask)


def _to_torch_tensor(
    array_like: Sequence[float] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert array-like data to a contiguous float32 torch tensor on device."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    return tensor.contiguous()


def gaussian_kde_triton(
    data: Sequence[float] | torch.Tensor,
    queries: Sequence[float] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
) -> torch.Tensor:
    """Evaluate the Gaussian KDE on the GPU using Triton.

    Args:
        data: Training samples (1D).
        queries: Query locations (1D).
        bandwidth: KDE bandwidth.
        block_m: Number of queries per Triton program.
        block_n: Number of data points per Triton program.
        num_warps: Triton launch parameter.
        num_stages: Triton launch parameter.
        device: CUDA device to use.

    Returns:
        Torch tensor (float32, on the requested device) containing densities.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("gaussian_kde_triton currently requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for Triton KDE.")

    train = _to_torch_tensor(data, device)
    query = _to_torch_tensor(queries, device)

    n_data = train.numel()
    n_query = query.numel()

    output = torch.zeros_like(query)

    grid_m = triton.cdiv(n_query, block_m)
    grid_n = triton.cdiv(n_data, block_n)
    grid = (grid_m, grid_n)

    _gaussian_kde_kernel[grid](
        train,
        query,
        output,
        n_data,
        n_query,
        1.0 / bandwidth,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if synchronize:
        torch.cuda.synchronize(device)
    output.mul_(1.0 / (bandwidth * n_data))
    return output


def gaussian_kde_triton_numpy(
    data: Sequence[float],
    queries: Sequence[float],
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Convenience wrapper returning CPU numpy arrays."""
    densities = gaussian_kde_triton(
        data=data, queries=queries, bandwidth=bandwidth, device=device
    )
    return densities.detach().cpu().numpy()


def gaussian_kde_score_triton(
    data: Sequence[float] | torch.Tensor,
    queries: Sequence[float] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KDE scores (pdf + derivative) using Triton."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("gaussian_kde_score_triton requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for Triton KDE.")

    train = _to_torch_tensor(data, device)
    query = _to_torch_tensor(queries, device)

    n_data = train.numel()
    n_query = query.numel()

    pdf_acc = torch.zeros_like(query)
    deriv_acc = torch.zeros_like(query)

    grid_m = triton.cdiv(n_query, block_m)
    grid_n = triton.cdiv(n_data, block_n)
    grid = (grid_m, grid_n)

    _gaussian_kde_pdf_deriv_kernel[grid](
        train,
        query,
        pdf_acc,
        deriv_acc,
        n_data,
        n_query,
        1.0 / bandwidth,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if synchronize:
        torch.cuda.synchronize(device)

    inv_norm = 1.0 / (n_data * bandwidth)
    pdf = pdf_acc * inv_norm
    deriv = deriv_acc * (inv_norm / bandwidth)
    score = deriv / (pdf + 1e-12)
    return score, pdf, deriv


def empirical_sd_kde_triton(
    data: Sequence[float] | torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    return_tensor: bool = False,
    synchronize: bool = True,
) -> tuple[torch.Tensor | np.ndarray, float]:
    """One-step empirical SD-KDE debiasing performed on the GPU."""
    if isinstance(data, torch.Tensor):
        host_array = data.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        host_array = np.asarray(data, dtype=np.float32)

    if host_array.size == 0:
        raise ValueError("data must contain at least one element.")

    n = host_array.size
    std_dev = float(host_array.std())
    iqr = float(np.percentile(host_array, 75) - np.percentile(host_array, 25))
    sigma = min(std_dev, iqr / 1.34)

    h = 0.4 * sigma * (n ** (-1 / 9))
    delta = 0.5 * (h ** 2)

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("empirical_sd_kde_triton requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but empirical SD-KDE was requested.")

    train = _to_torch_tensor(host_array, device)

    scores, _, _ = gaussian_kde_score_triton(
        train,
        train,
        h,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        device=device,
        synchronize=synchronize,
    )

    debiased = train + delta * scores
    if synchronize:
        torch.cuda.synchronize(device)

    if return_tensor:
        return debiased, h
    return debiased.detach().cpu().numpy(), h

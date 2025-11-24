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
import warnings
from typing import Sequence

import torch
import triton
import triton.language as tl
import numpy as np

_CUDA_MAX_GRID_DIM_X = 65_535
_CUDA_MAX_GRID_DIM_Y = 65_535
_MAX_BLOCK_TILE = 1024
_ND_FEATURES = 16


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


@triton.jit
def _gaussian_kde_kernel_nd(
    data_ptr,
    query_ptr,
    out_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    inv_h2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tensor-core-friendly KDE evaluation for fixed dimensionality."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    query_mask = offs_m < n_query
    data_mask = offs_n < n_data

    q_ptrs = query_ptr + (offs_m[:, None] * stride_query + offs_k[None, :])
    d_ptrs = data_ptr + (offs_n[:, None] * stride_data + offs_k[None, :])
    q_block = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
    d_block = tl.load(d_ptrs, mask=data_mask[:, None], other=0.0)

    q_norm = tl.sum(q_block * q_block, axis=1)
    d_norm = tl.sum(d_block * d_block, axis=1)

    dot = tl.dot(q_block, tl.trans(d_block), allow_tf32=True)
    #dot = tl.dot(q_block, tl.trans(d_block), input_precision="ieee")
    dist = tl.maximum(q_norm[:, None] + d_norm[None, :] - 2.0 * dot, 0.0)
    contrib = tl.exp(-0.5 * dist * inv_h2)

    data_mask_matrix = data_mask[None, :]
    contrib = tl.where(data_mask_matrix, contrib, 0.0)

    block_sum = tl.sum(contrib, axis=1)
    tl.atomic_add(out_ptr + offs_m, block_sum, mask=query_mask)


@triton.jit
def _empirical_sd_kde_kernel_nd(
    data_ptr,
    query_ptr,
    pdf_ptr,
    weighted_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_weighted,
    inv_h2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Accumulate phi sums and phi-weighted data sums for SD-KDE debiasing."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    query_mask = offs_m < n_query
    data_mask = offs_n < n_data

    q_ptrs = query_ptr + (offs_m[:, None] * stride_query + offs_k[None, :])
    d_ptrs = data_ptr + (offs_n[:, None] * stride_data + offs_k[None, :])
    q_block = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
    d_block = tl.load(d_ptrs, mask=data_mask[:, None], other=0.0)

    q_norm = tl.sum(q_block * q_block, axis=1)
    d_norm = tl.sum(d_block * d_block, axis=1)

    dot = tl.dot(q_block, tl.trans(d_block), allow_tf32=True)
    #dot = tl.dot(q_block, tl.trans(d_block), input_precision="ieee")
    dist = tl.maximum(q_norm[:, None] + d_norm[None, :] - 2.0 * dot, 0.0)
    phi = tl.exp(-0.5 * dist * inv_h2)

    data_mask_matrix = data_mask[None, :]
    phi = tl.where(data_mask_matrix, phi, 0.0)

    phi_sum = tl.sum(phi, axis=1)
    weighted = tl.dot(phi, d_block, allow_tf32=True)
    #weighted = tl.dot(phi, d_block, input_precision="ieee")

    tl.atomic_add(pdf_ptr + offs_m, phi_sum, mask=query_mask)

    w_ptrs = weighted_ptr + (offs_m[:, None] * stride_weighted + offs_k[None, :])
    tl.atomic_add(w_ptrs, weighted, mask=query_mask[:, None])


def _to_torch_tensor(
    array_like: Sequence[float] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert array-like data to a contiguous float32 torch tensor on device."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    return tensor.contiguous()


def _to_matrix_tensor(
    array_like: Sequence[Sequence[float]] | torch.Tensor,
    device: torch.device,
    *,
    dim: int,
) -> torch.Tensor:
    """Convert array-like data to (n, dim) float32 tensor on device."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != dim:
        raise ValueError(f"expected tensor with shape (n, {dim}), got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length())


def _resolve_launch_shape(
    *,
    n_query: int,
    n_data: int,
    block_m: int,
    block_n: int,
    kernel_name: str,
) -> tuple[int, int, int, int]:
    """Ensure Triton grid dimensions stay within CUDA limits."""
    bm = int(block_m)
    bn = int(block_n)

    if bm <= 0 or bn <= 0:
        raise ValueError(f"{kernel_name}: block sizes must be positive, got ({bm}, {bn}).")

    grid_m = triton.cdiv(n_query, bm)
    if grid_m > _CUDA_MAX_GRID_DIM_X:
        required_bm = math.ceil(n_query / _CUDA_MAX_GRID_DIM_X)
        promoted_bm = max(bm, _next_power_of_two(required_bm))
        promoted_bm = min(_MAX_BLOCK_TILE, promoted_bm)
        promoted_bm = min(promoted_bm, n_query)
        grid_m = triton.cdiv(n_query, promoted_bm)
        if grid_m > _CUDA_MAX_GRID_DIM_X:
            raise ValueError(
                f"{kernel_name}: even block_m={promoted_bm} yields grid_m={grid_m} "
                f"> {_CUDA_MAX_GRID_DIM_X}; split the queries before launching."
            )
        if promoted_bm != bm:
            warnings.warn(
                f"{kernel_name}: promoting block_m from {bm} to {promoted_bm} so that "
                f"grid_m stays within CUDA limit {_CUDA_MAX_GRID_DIM_X}.",
                RuntimeWarning,
                stacklevel=3,
            )
            bm = promoted_bm

    grid_n = triton.cdiv(n_data, bn)
    if grid_n > _CUDA_MAX_GRID_DIM_Y:
        required_bn = math.ceil(n_data / _CUDA_MAX_GRID_DIM_Y)
        promoted_bn = max(bn, _next_power_of_two(required_bn))
        promoted_bn = min(_MAX_BLOCK_TILE, promoted_bn)
        promoted_bn = min(promoted_bn, n_data)
        grid_n = triton.cdiv(n_data, promoted_bn)
        if grid_n > _CUDA_MAX_GRID_DIM_Y:
            raise ValueError(
                f"{kernel_name}: even block_n={promoted_bn} yields grid_n={grid_n} "
                f"> {_CUDA_MAX_GRID_DIM_Y}; split the training data before launching."
            )
        if promoted_bn != bn:
            warnings.warn(
                f"{kernel_name}: promoting block_n from {bn} to {promoted_bn} so that "
                f"grid_n stays within CUDA limit {_CUDA_MAX_GRID_DIM_Y}.",
                RuntimeWarning,
                stacklevel=3,
            )
            bn = promoted_bn

    return bm, bn, grid_m, grid_n


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

    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    # Chunk queries so each launch keeps gridDim.x within CUDA limits.
    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        query_chunk = query[q_start:q_end]
        output_chunk = output[q_start:q_end]
        chunk_n_query = query_chunk.numel()

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="gaussian_kde_triton",
        )
        grid = (grid_m, grid_n)

        _gaussian_kde_kernel[grid](
            train,
            query_chunk,
            output_chunk,
            n_data,
            chunk_n_query,
            1.0 / bandwidth,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
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


def gaussian_kde_triton_nd(
    data: Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
) -> torch.Tensor:
    """Evaluate 16-D Gaussian KDE using a Tensor-Core-friendly Triton kernel."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("gaussian_kde_triton_nd requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for Triton KDE.")

    train = _to_matrix_tensor(data, device, dim=_ND_FEATURES)
    query = _to_matrix_tensor(queries, device, dim=_ND_FEATURES)

    n_data = train.shape[0]
    n_query = query.shape[0]
    if n_data == 0 or n_query == 0:
        raise ValueError("data and queries must contain at least one sample.")

    output = torch.zeros(n_query, device=device, dtype=torch.float32)
    inv_bandwidth = 1.0 / bandwidth
    inv_h2 = inv_bandwidth * inv_bandwidth

    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)
    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        query_chunk = query[q_start:q_end]
        output_chunk = output[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="gaussian_kde_triton_nd",
        )
        grid = (grid_m, grid_n)

        _gaussian_kde_kernel_nd[grid](
            train,
            query_chunk,
            output_chunk,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            inv_h2,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=_ND_FEATURES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if synchronize:
        torch.cuda.synchronize(device)

    norm = (inv_bandwidth ** _ND_FEATURES) / (
        ((2.0 * math.pi) ** (_ND_FEATURES / 2.0)) * n_data
    )
    output.mul_(norm)
    return output


def gaussian_kde_triton_nd_numpy(
    data: Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Convenience wrapper returning CPU numpy arrays for the 16-D KDE."""
    densities = gaussian_kde_triton_nd(
        data=data, queries=queries, bandwidth=bandwidth, device=device
    )
    return densities.detach().cpu().numpy()


def empirical_sd_kde_triton_nd(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    return_tensor: bool = False,
    synchronize: bool = True,
) -> tuple[torch.Tensor | np.ndarray, float]:
    """Empirical SD-KDE debiasing in 16-D using Tensor-Core-accelerated Triton kernels."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("empirical_sd_kde_triton_nd requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for SD-KDE.")

    train = _to_matrix_tensor(data, device, dim=_ND_FEATURES)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must contain at least one element.")

    pdf_acc = torch.zeros(n_data, device=device, dtype=torch.float32)
    weighted_acc = torch.zeros(
        (n_data, _ND_FEATURES), device=device, dtype=torch.float32
    )

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)

    for q_start in range(0, n_data, max_queries_per_launch):
        q_end = min(n_data, q_start + max_queries_per_launch)
        query_chunk = train[q_start:q_end]
        pdf_chunk = pdf_acc[q_start:q_end]
        weighted_chunk = weighted_acc[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="empirical_sd_kde_triton_nd",
        )
        grid = (grid_m, grid_n)

        _empirical_sd_kde_kernel_nd[grid](
            train,
            query_chunk,
            pdf_chunk,
            weighted_chunk,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            weighted_chunk.stride(0),
            inv_h2,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=_ND_FEATURES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if synchronize:
        torch.cuda.synchronize(device)

    eps = 1e-12
    phi_sum = pdf_acc
    phi_y = weighted_acc
    ratio = phi_y / (phi_sum.unsqueeze(1) + eps)
    inv_h2_scalar = inv_h2
    score = (ratio - train) * inv_h2_scalar
    delta = 0.5 * (bandwidth ** 2)
    debiased = train + delta * score

    if return_tensor:
        return debiased, bandwidth
    return debiased.detach().cpu().numpy(), bandwidth


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

    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    # Chunk queries so each launch keeps gridDim.x within CUDA limits.
    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        query_chunk = query[q_start:q_end]
        pdf_chunk = pdf_acc[q_start:q_end]
        deriv_chunk = deriv_acc[q_start:q_end]
        chunk_n_query = query_chunk.numel()

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="gaussian_kde_score_triton",
        )
        grid = (grid_m, grid_n)

        _gaussian_kde_pdf_deriv_kernel[grid](
            train,
            query_chunk,
            pdf_chunk,
            deriv_chunk,
            n_data,
            chunk_n_query,
            1.0 / bandwidth,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
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
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 1,
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

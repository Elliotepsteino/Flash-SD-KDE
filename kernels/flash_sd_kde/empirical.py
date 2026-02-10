# Copyright (c) 2026, Elliot Epstein.
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from .common import (
    _CUDA_MAX_GRID_DIM_X,
    _ND_FEATURES,
    _resolve_launch_shape,
    _to_matrix_tensor,
    _to_torch_tensor,
)
from .kde import gaussian_kde_score_triton


@triton.jit
def _empirical_sd_kde_kernel_nd_old(
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
    dist = q_norm[:, None] + d_norm[None, :] - 2.0 * dot
    phi = tl.exp(-0.5 * dist * inv_h2)

    data_mask_matrix = data_mask[None, :]
    phi = tl.where(data_mask_matrix, phi, 0.0)

    phi_sum = tl.sum(phi, axis=1)
    weighted = tl.dot(phi, d_block, allow_tf32=True)

    tl.atomic_add(pdf_ptr + offs_m, phi_sum, mask=query_mask)

    w_ptrs = weighted_ptr + (offs_m[:, None] * stride_weighted + offs_k[None, :])
    tl.atomic_add(w_ptrs, weighted, mask=query_mask[:, None])


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
    BLOCK_N_CHUNK: tl.constexpr = 16,
):
    """Accumulate phi sums and phi-weighted data sums for SD-KDE debiasing,
    using streaming over the data dimension to reduce shared-memory usage.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)

    query_mask = offs_m < n_query

    q_ptrs = query_ptr + (offs_m[:, None] * stride_query + offs_k[None, :])
    q_block = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
    q_norm = tl.sum(q_block * q_block, axis=1)

    phi_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    weighted = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, BLOCK_N, BLOCK_N_CHUNK):
        offs_n_chunk = offs_n_base + n_start + tl.arange(0, BLOCK_N_CHUNK)
        data_mask_chunk = offs_n_chunk < n_data

        d_ptrs_chunk = data_ptr + (offs_n_chunk[:, None] * stride_data + offs_k[None, :])
        d_chunk = tl.load(d_ptrs_chunk, mask=data_mask_chunk[:, None], other=0.0)

        d_norm_chunk = tl.sum(d_chunk * d_chunk, axis=1)

        dot_chunk = tl.dot(q_block, tl.trans(d_chunk), allow_tf32=True)
        dist_chunk = q_norm[:, None] + d_norm_chunk[None, :] - 2.0 * dot_chunk

        phi_chunk = tl.exp(-0.5 * dist_chunk * inv_h2)

        phi_chunk = tl.where(data_mask_chunk[None, :], phi_chunk, 0.0)
        phi_chunk = tl.where(query_mask[:, None], phi_chunk, 0.0)

        phi_sum += tl.sum(phi_chunk, axis=1)
        weighted += tl.dot(phi_chunk, d_chunk, allow_tf32=True)

    tl.atomic_add(pdf_ptr + offs_m, phi_sum, mask=query_mask)

    w_ptrs = weighted_ptr + (offs_k[:, None] * stride_weighted + offs_m[None, :])
    tl.atomic_add(w_ptrs, tl.trans(weighted), mask=query_mask[None, :])


def emp_score_16d_flash_sd_kde(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 2048,
    num_warps: int = 2,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 16D empirical-score accumulators used by SD-KDE debiasing.

    For each training point `x_i`, this computes:
    - `pdf_sum[i] = sum_j exp(-||x_i - x_j||^2 / (2 h^2))`
    - `weighted_sum[i] = sum_j exp(-||x_i - x_j||^2 / (2 h^2)) * x_j`

    Arguments:
        data: Training samples with shape `(n_train, 16)`.
        bandwidth: Positive scalar bandwidth `h`.
        block_m: Query tile size.
        block_n: Training tile size.
        num_warps: Triton launch parameter.
        num_stages: Triton launch parameter.
        device: CUDA device.
        synchronize: If True, calls `torch.cuda.synchronize` before returning.

    Return:
        pdf_sum: Tensor `(n_train,)`.
        weighted_sum: Tensor `(n_train, 16)`.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("emp_score_16d_flash_sd_kde requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for SD-KDE.")

    train = _to_matrix_tensor(data, device, dim=_ND_FEATURES)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must contain at least one element.")

    pdf_acc = torch.zeros(n_data, device=device, dtype=torch.float32)
    weighted_acc = torch.zeros((_ND_FEATURES, n_data), device=device, dtype=torch.float32)

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
            kernel_name="emp_score_16d_flash_sd_kde",
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

    return pdf_acc, weighted_acc.t()


def empirical_sd_kde_triton_nd(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 2048,
    num_warps: int = 2,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    return_tensor: bool = False,
    synchronize: bool = True,
) -> tuple[torch.Tensor | np.ndarray, float]:
    """Run one-step 16D empirical SD-KDE debiasing on CUDA.

    Arguments:
        data: Training samples with shape `(n_train, 16)`.
        bandwidth: Positive scalar bandwidth `h`.
        block_m: Query tile size.
        block_n: Training tile size.
        num_warps: Triton launch parameter.
        num_stages: Triton launch parameter.
        device: CUDA device.
        return_tensor: If True, return CUDA tensor output; else return numpy.
        synchronize: If True, calls `torch.cuda.synchronize` before returning.

    Return:
        debiased_data: Shape `(n_train, 16)` as tensor or numpy array.
        bandwidth: The same input bandwidth `h`.
    """
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

    pdf_sum, weighted_sum = emp_score_16d_flash_sd_kde(
        train,
        bandwidth,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        device=device,
        synchronize=synchronize,
    )

    eps = 1e-12
    score = (weighted_sum / (pdf_sum[:, None] + eps) - train) * (1.0 / (bandwidth * bandwidth))
    delta = 0.5 * (bandwidth ** 2)
    debiased = train + delta * score

    if return_tensor:
        return debiased, bandwidth
    return debiased.detach().cpu().numpy(), bandwidth


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
    """Run one-step 1D empirical SD-KDE debiasing on CUDA.

    This path estimates a 1D bandwidth from input data, computes KDE score at
    training points, and applies the one-step SD correction.

    Arguments:
        data: Training samples with shape `(n_train,)`.
        block_m: Query tile size.
        block_n: Training tile size.
        num_warps: Triton launch parameter.
        num_stages: Triton launch parameter.
        device: CUDA device.
        return_tensor: If True, return CUDA tensor output; else return numpy.
        synchronize: If True, calls `torch.cuda.synchronize` before returning.

    Return:
        debiased_data: Shape `(n_train,)` as tensor or numpy array.
        bandwidth: Scalar Silverman-style bandwidth used internally.
    """
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

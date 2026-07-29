from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from globals import PRECISION_FAST_TF32, PRECISION_FP32_IEEE

from .common import (
    _CUDA_MAX_GRID_DIM_X,
    _GENERAL_TILE_K,
    _pad_feature_matrix,
    _resolve_launch_shape,
    _to_feature_matrix_tensor,
)


@triton.jit
def _gaussian_kde_padded_atomic_kernel(
    data_ptr,
    query_ptr,
    out_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    inv_h2,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    APPLY_LAPLACIAN: tl.constexpr,
    DIM_TRUE: tl.constexpr,
    DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    query_mask = offs_m < n_query
    data_mask = offs_n < n_data

    q_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    d_norm = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, DIM_PAD, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        q_ptrs = query_ptr + offs_m[:, None] * stride_query + offs_k[None, :]
        d_ptrs = data_ptr + offs_n[:, None] * stride_data + offs_k[None, :]
        q_chunk = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
        d_chunk = tl.load(d_ptrs, mask=data_mask[:, None], other=0.0)
        q_norm += tl.sum(q_chunk * q_chunk, axis=1)
        d_norm += tl.sum(d_chunk * d_chunk, axis=1)
        if USE_IEEE:
            dot_acc += tl.dot(q_chunk, tl.trans(d_chunk), input_precision="ieee")
        else:
            dot_acc += tl.dot(q_chunk, tl.trans(d_chunk), allow_tf32=ALLOW_TF32)

    dist = tl.maximum(q_norm[:, None] + d_norm[None, :] - 2.0 * dot_acc, 0.0)
    scaled = dist * inv_h2
    phi = tl.exp(-0.5 * scaled)
    if APPLY_LAPLACIAN:
        phi = phi * (1.0 + 0.5 * DIM_TRUE - 0.5 * scaled)
    phi = tl.where(query_mask[:, None] & data_mask[None, :], phi, 0.0)
    tl.atomic_add(out_ptr + offs_m, tl.sum(phi, axis=1), mask=query_mask)


def _resolve_block_k(block_k: int, dim_pad: int, kernel_name: str) -> int:
    """Clamp block_k to dim_pad and require exact K-tiling."""
    bk = min(int(block_k), dim_pad)
    if bk <= 0 or dim_pad % bk != 0:
        raise ValueError(
            f"{kernel_name}: block_k={block_k} must be positive and divide dim_pad={dim_pad}."
        )
    return bk


def gaussian_kde_triton_padded_nd(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = _GENERAL_TILE_K,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
    precision_mode: str = PRECISION_FAST_TF32,
    apply_laplacian_correction: bool = False,
    return_unnormalized: bool = False,
) -> torch.Tensor:
    """Evaluate Gaussian KDE on CUDA for any feature dimension via zero padding."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("gaussian_kde_triton_padded_nd requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for Triton KDE.")
    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    train_in = _to_feature_matrix_tensor(data, device)
    query_in = _to_feature_matrix_tensor(queries, device)
    if train_in.shape[1] != query_in.shape[1]:
        raise ValueError(
            f"data and queries must share feature dimension, got {train_in.shape[1]} and {query_in.shape[1]}."
        )

    train, dim_true, dim_pad = _pad_feature_matrix(train_in, tile_multiple=_GENERAL_TILE_K)
    query, query_dim_true, _ = _pad_feature_matrix(query_in, tile_multiple=_GENERAL_TILE_K)
    if query_dim_true != dim_true:
        raise ValueError(
            f"data and queries must share feature dimension, got {dim_true} and {query_dim_true}."
        )

    n_data = train.shape[0]
    n_query = query.shape[0]
    if n_data == 0 or n_query == 0:
        raise ValueError("data and queries must contain at least one sample.")

    output = torch.zeros(n_query, device=device, dtype=torch.float32)
    inv_bandwidth = 1.0 / bandwidth
    inv_h2 = inv_bandwidth * inv_bandwidth

    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32
    resolved_block_k = _resolve_block_k(block_k, dim_pad, "gaussian_kde_triton_padded_nd")
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
            kernel_name="gaussian_kde_triton_padded_nd",
        )
        grid = (grid_m, grid_n)
        _gaussian_kde_padded_atomic_kernel[grid](
            train,
            query_chunk,
            output_chunk,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            inv_h2,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            APPLY_LAPLACIAN=apply_laplacian_correction,
            DIM_TRUE=dim_true,
            DIM_PAD=dim_pad,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=resolved_block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if synchronize:
        torch.cuda.synchronize(device)
    if return_unnormalized:
        return output

    norm = (inv_bandwidth ** dim_true) / (((2.0 * math.pi) ** (dim_true / 2.0)) * n_data)
    output.mul_(norm)
    return output


def gaussian_kde_triton_padded_nd_numpy(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = PRECISION_FAST_TF32,
    apply_laplacian_correction: bool = False,
) -> np.ndarray:
    densities = gaussian_kde_triton_padded_nd(
        data=data,
        queries=queries,
        bandwidth=bandwidth,
        device=device,
        precision_mode=precision_mode,
        apply_laplacian_correction=apply_laplacian_correction,
    )
    return densities.detach().cpu().numpy()


@triton.jit
def _emp_score_padded_atomic_kernel(
    data_ptr,
    query_ptr,
    pdf_ptr,
    weighted_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_weighted_query,
    stride_weighted_k,
    inv_h2,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    query_mask = offs_m < n_query
    data_mask = offs_n < n_data

    q_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    d_norm = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, DIM_PAD, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        q_ptrs = query_ptr + offs_m[:, None] * stride_query + offs_k[None, :]
        d_ptrs = data_ptr + offs_n[:, None] * stride_data + offs_k[None, :]
        q_chunk = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
        d_chunk = tl.load(d_ptrs, mask=data_mask[:, None], other=0.0)
        q_norm += tl.sum(q_chunk * q_chunk, axis=1)
        d_norm += tl.sum(d_chunk * d_chunk, axis=1)
        if USE_IEEE:
            dot_acc += tl.dot(q_chunk, tl.trans(d_chunk), input_precision="ieee")
        else:
            dot_acc += tl.dot(q_chunk, tl.trans(d_chunk), allow_tf32=ALLOW_TF32)

    dist = tl.maximum(q_norm[:, None] + d_norm[None, :] - 2.0 * dot_acc, 0.0)
    phi = tl.exp(-0.5 * dist * inv_h2)
    phi = tl.where(query_mask[:, None] & data_mask[None, :], phi, 0.0)

    tl.atomic_add(pdf_ptr + offs_m, tl.sum(phi, axis=1), mask=query_mask)

    for k_start in range(0, DIM_PAD, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        d_ptrs = data_ptr + offs_n[:, None] * stride_data + offs_k[None, :]
        d_chunk = tl.load(d_ptrs, mask=data_mask[:, None], other=0.0)
        if USE_IEEE:
            weighted_chunk = tl.dot(phi, d_chunk, input_precision="ieee")
        else:
            weighted_chunk = tl.dot(phi, d_chunk, allow_tf32=ALLOW_TF32)
        w_ptrs = (
            weighted_ptr
            + offs_m[:, None] * stride_weighted_query
            + offs_k[None, :] * stride_weighted_k
        )
        tl.atomic_add(w_ptrs, weighted_chunk, mask=query_mask[:, None])


@triton.jit
def _emp_score_padded_chunked_kernel(
    data_ptr,
    query_ptr,
    pdf_ptr,
    weighted_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_weighted_query,
    stride_weighted_k,
    inv_h2,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_CHUNK: tl.constexpr,
):
    """General-d port of the specialized 16-D score kernel structure.

    The query block and the running `weighted` accumulator stay register
    resident for a whole BLOCK_N macro-tile while the data streams through in
    BLOCK_N_CHUNK-row slices; each phi chunk is consumed immediately by both
    accumulations, so the data is read once and one atomic add is issued per
    macro-tile. DIM_PAD is the K dimension of both dots.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = pid_n * BLOCK_N
    offs_k = tl.arange(0, DIM_PAD)
    query_mask = offs_m < n_query

    q_ptrs = query_ptr + offs_m[:, None] * stride_query + offs_k[None, :]
    q_block = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)
    q_norm = tl.sum(q_block * q_block, axis=1)

    phi_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    weighted = tl.zeros((BLOCK_M, DIM_PAD), dtype=tl.float32)

    for n_start in range(0, BLOCK_N, BLOCK_N_CHUNK):
        offs_n_chunk = offs_n_base + n_start + tl.arange(0, BLOCK_N_CHUNK)
        data_mask_chunk = offs_n_chunk < n_data

        d_ptrs = data_ptr + offs_n_chunk[:, None] * stride_data + offs_k[None, :]
        d_chunk = tl.load(d_ptrs, mask=data_mask_chunk[:, None], other=0.0)
        d_norm_chunk = tl.sum(d_chunk * d_chunk, axis=1)

        if USE_IEEE:
            dot_chunk = tl.dot(q_block, tl.trans(d_chunk), input_precision="ieee")
        else:
            dot_chunk = tl.dot(q_block, tl.trans(d_chunk), allow_tf32=ALLOW_TF32)
        dist_chunk = tl.maximum(
            q_norm[:, None] + d_norm_chunk[None, :] - 2.0 * dot_chunk, 0.0
        )
        phi_chunk = tl.exp(-0.5 * dist_chunk * inv_h2)
        phi_chunk = tl.where(
            query_mask[:, None] & data_mask_chunk[None, :], phi_chunk, 0.0
        )

        phi_sum += tl.sum(phi_chunk, axis=1)
        if USE_IEEE:
            weighted += tl.dot(phi_chunk, d_chunk, input_precision="ieee")
        else:
            weighted += tl.dot(phi_chunk, d_chunk, allow_tf32=ALLOW_TF32)

    tl.atomic_add(pdf_ptr + offs_m, phi_sum, mask=query_mask)

    w_ptrs = (
        weighted_ptr
        + offs_m[:, None] * stride_weighted_query
        + offs_k[None, :] * stride_weighted_k
    )
    tl.atomic_add(w_ptrs, weighted, mask=query_mask[:, None])


def emp_score_padded_nd_flash_sd_kde(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = _GENERAL_TILE_K,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
    precision_mode: str = PRECISION_FAST_TF32,
    chunked: bool = False,
    block_n_chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute empirical-score accumulators for any dimension via padded Triton kernels.

    With ``chunked=True`` the launch uses the single-pass chunk-streaming kernel
    (the general-d port of the specialized 16-D structure); ``block_n`` must then
    be a multiple of ``block_n_chunk``, and ``block_k`` is ignored.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("emp_score_padded_nd_flash_sd_kde requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for SD-KDE.")
    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    train_in = _to_feature_matrix_tensor(data, device)
    train, dim_true, dim_pad = _pad_feature_matrix(train_in, tile_multiple=_GENERAL_TILE_K)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must contain at least one sample.")

    pdf_sum = torch.zeros((n_data,), device=device, dtype=torch.float32)
    weighted_sum = torch.zeros((n_data, dim_pad), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32
    resolved_block_k = _resolve_block_k(block_k, dim_pad, "emp_score_padded_nd_flash_sd_kde")
    if chunked and (block_n_chunk < 16 or block_n % block_n_chunk != 0):
        raise ValueError(
            f"chunked launch requires block_n_chunk >= 16 dividing block_n, got "
            f"{block_n_chunk} and {block_n}."
        )
    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)
    for q_start in range(0, n_data, max_queries_per_launch):
        q_end = min(n_data, q_start + max_queries_per_launch)
        query_chunk = train[q_start:q_end]
        pdf_chunk = pdf_sum[q_start:q_end]
        weighted_chunk = weighted_sum[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]
        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="emp_score_padded_nd_flash_sd_kde",
        )
        grid = (grid_m, grid_n)
        if chunked:
            _emp_score_padded_chunked_kernel[grid](
                train,
                query_chunk,
                pdf_chunk,
                weighted_chunk,
                n_data,
                chunk_n_query,
                stride_data,
                query_chunk.stride(0),
                weighted_chunk.stride(0),
                weighted_chunk.stride(1),
                inv_h2,
                USE_IEEE=use_ieee,
                ALLOW_TF32=allow_tf32,
                DIM_PAD=dim_pad,
                BLOCK_M=chunk_block_m,
                BLOCK_N=chunk_block_n,
                BLOCK_N_CHUNK=block_n_chunk,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            continue
        _emp_score_padded_atomic_kernel[grid](
            train,
            query_chunk,
            pdf_chunk,
            weighted_chunk,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            weighted_chunk.stride(0),
            weighted_chunk.stride(1),
            inv_h2,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            DIM_PAD=dim_pad,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=resolved_block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if synchronize:
        torch.cuda.synchronize(device)
    return pdf_sum, weighted_sum[:, :dim_true].contiguous()


def empirical_sd_kde_triton_padded_nd(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = _GENERAL_TILE_K,
    num_warps: int = 4,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    return_tensor: bool = False,
    synchronize: bool = True,
    precision_mode: str = PRECISION_FAST_TF32,
    chunked: bool = False,
    block_n_chunk: int = 16,
) -> tuple[torch.Tensor | np.ndarray, float]:
    """Run one-step empirical SD-KDE debiasing on CUDA for any feature dimension."""
    device = torch.device(device)
    train = _to_feature_matrix_tensor(data, device)
    pdf_sum, weighted_sum = emp_score_padded_nd_flash_sd_kde(
        train,
        bandwidth,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        device=device,
        synchronize=synchronize,
        precision_mode=precision_mode,
        chunked=chunked,
        block_n_chunk=block_n_chunk,
    )
    eps = 1e-12
    score = (weighted_sum / (pdf_sum[:, None] + eps) - train) * (1.0 / (bandwidth * bandwidth))
    delta = 0.5 * (bandwidth ** 2)
    debiased = train + delta * score
    if return_tensor:
        return debiased, bandwidth
    return debiased.detach().cpu().numpy(), bandwidth

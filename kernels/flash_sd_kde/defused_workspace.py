# Copyright (c) 2026, Elliot Epstein.
from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from .common import (
    _CUDA_MAX_GRID_DIM_X,
    _ND_FEATURES,
    _resolve_launch_shape,
    _to_matrix_tensor,
)


@triton.jit
def _empirical_sd_kde_workspace_kernel_nd(
    data_ptr,
    query_ptr,
    pdf_workspace_ptr,
    weighted_workspace_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_pdf_tile,
    stride_pdf_query,
    stride_weighted_tile,
    stride_weighted_query,
    stride_weighted_k,
    inv_h2,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_CHUNK: tl.constexpr = 16,
):
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

        if USE_IEEE:
            dot_chunk = tl.dot(q_block, tl.trans(d_chunk), input_precision="ieee")
        else:
            dot_chunk = tl.dot(q_block, tl.trans(d_chunk), allow_tf32=ALLOW_TF32)
        dist_chunk = q_norm[:, None] + d_norm_chunk[None, :] - 2.0 * dot_chunk
        phi_chunk = tl.exp(-0.5 * dist_chunk * inv_h2)
        phi_chunk = tl.where(data_mask_chunk[None, :], phi_chunk, 0.0)
        phi_chunk = tl.where(query_mask[:, None], phi_chunk, 0.0)

        phi_sum += tl.sum(phi_chunk, axis=1)
        if USE_IEEE:
            weighted += tl.dot(phi_chunk, d_chunk, input_precision="ieee")
        else:
            weighted += tl.dot(phi_chunk, d_chunk, allow_tf32=ALLOW_TF32)

    pdf_ptrs = pdf_workspace_ptr + pid_n * stride_pdf_tile + offs_m * stride_pdf_query
    tl.store(pdf_ptrs, phi_sum, mask=query_mask)

    weighted_ptrs = (
        weighted_workspace_ptr
        + pid_n * stride_weighted_tile
        + offs_m[:, None] * stride_weighted_query
        + offs_k[None, :] * stride_weighted_k
    )
    tl.store(weighted_ptrs, weighted, mask=query_mask[:, None])


@triton.jit
def _gaussian_kde_workspace_kernel_nd(
    data_ptr,
    query_ptr,
    kde_workspace_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_workspace_tile,
    stride_workspace_query,
    inv_h2,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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

    if USE_IEEE:
        dot = tl.dot(q_block, tl.trans(d_block), input_precision="ieee")
    else:
        dot = tl.dot(q_block, tl.trans(d_block), allow_tf32=ALLOW_TF32)
    dist = tl.maximum(q_norm[:, None] + d_norm[None, :] - 2.0 * dot, 0.0)
    contrib = tl.exp(-0.5 * dist * inv_h2)
    contrib = tl.where(data_mask[None, :], contrib, 0.0)
    block_sum = tl.sum(contrib, axis=1)

    workspace_ptrs = kde_workspace_ptr + pid_n * stride_workspace_tile + offs_m * stride_workspace_query
    tl.store(workspace_ptrs, block_sum, mask=query_mask)


def _workspace_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors)


def emp_score_16d_flash_sd_kde_defused(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    block_m: int = 64,
    block_n: int = 2048,
    num_warps: int = 2,
    num_stages: int = 2,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
    use_tensorcores: bool = True,
    return_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("emp_score_16d_flash_sd_kde_defused requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for SD-KDE.")

    train = _to_matrix_tensor(data, device, dim=_ND_FEATURES)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must contain at least one element.")

    pdf_acc = torch.empty(n_data, device=device, dtype=torch.float32)
    weighted_acc = torch.empty((n_data, _ND_FEATURES), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)
    use_ieee = not use_tensorcores
    allow_tf32 = use_tensorcores
    score_workspace_peak_bytes = 0

    for q_start in range(0, n_data, max_queries_per_launch):
        q_end = min(n_data, q_start + max_queries_per_launch)
        query_chunk = train[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="emp_score_16d_flash_sd_kde_defused",
        )
        grid = (grid_m, grid_n)

        pdf_workspace = torch.empty((grid_n, chunk_n_query), device=device, dtype=torch.float32)
        weighted_workspace = torch.empty(
            (grid_n, chunk_n_query, _ND_FEATURES),
            device=device,
            dtype=torch.float32,
        )
        score_workspace_peak_bytes = max(
            score_workspace_peak_bytes,
            _workspace_bytes(pdf_workspace, weighted_workspace),
        )

        _empirical_sd_kde_workspace_kernel_nd[grid](
            train,
            query_chunk,
            pdf_workspace,
            weighted_workspace,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            pdf_workspace.stride(0),
            pdf_workspace.stride(1),
            weighted_workspace.stride(0),
            weighted_workspace.stride(1),
            weighted_workspace.stride(2),
            inv_h2,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=_ND_FEATURES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        pdf_acc[q_start:q_end] = pdf_workspace.sum(dim=0)
        weighted_acc[q_start:q_end] = weighted_workspace.sum(dim=0)
        del pdf_workspace, weighted_workspace

    if synchronize:
        torch.cuda.synchronize(device)

    if return_debug:
        return pdf_acc, weighted_acc, {"score_workspace_bytes": float(score_workspace_peak_bytes)}
    return pdf_acc, weighted_acc


def gaussian_kde_triton_nd_defused(
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
    use_tensorcores: bool = True,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("gaussian_kde_triton_nd_defused requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested for Triton KDE.")

    train = _to_matrix_tensor(data, device, dim=_ND_FEATURES)
    query = _to_matrix_tensor(queries, device, dim=_ND_FEATURES)

    n_data = train.shape[0]
    n_query = query.shape[0]
    if n_data == 0 or n_query == 0:
        raise ValueError("data and queries must contain at least one sample.")

    output = torch.empty(n_query, device=device, dtype=torch.float32)
    inv_bandwidth = 1.0 / bandwidth
    inv_h2 = inv_bandwidth * inv_bandwidth

    max_queries_per_launch = max(block_m, block_m * _CUDA_MAX_GRID_DIM_X)
    stride_data = train.stride(0)
    use_ieee = not use_tensorcores
    allow_tf32 = use_tensorcores
    kde_workspace_peak_bytes = 0
    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        query_chunk = query[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="gaussian_kde_triton_nd_defused",
        )
        grid = (grid_m, grid_n)

        kde_workspace = torch.empty((grid_n, chunk_n_query), device=device, dtype=torch.float32)
        kde_workspace_peak_bytes = max(kde_workspace_peak_bytes, _workspace_bytes(kde_workspace))

        _gaussian_kde_workspace_kernel_nd[grid](
            train,
            query_chunk,
            kde_workspace,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            kde_workspace.stride(0),
            kde_workspace.stride(1),
            inv_h2,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=_ND_FEATURES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        output[q_start:q_end] = kde_workspace.sum(dim=0)
        del kde_workspace

    if synchronize:
        torch.cuda.synchronize(device)

    norm = (inv_bandwidth ** _ND_FEATURES) / (
        ((2.0 * math.pi) ** (_ND_FEATURES / 2.0)) * n_data
    )
    output.mul_(norm)

    if return_debug:
        return output, {"kde_workspace_bytes": float(kde_workspace_peak_bytes)}
    return output


def flash_sd_kde_triton_nd_defused(
    data: Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    score_block_m: int = 64,
    score_block_n: int = 2048,
    kde_block_m: int = 64,
    kde_block_n: int = 64,
    device: str | torch.device = "cuda",
    synchronize: bool = True,
    use_tensorcores: bool = True,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    pdf_sum, weighted_sum, score_debug = emp_score_16d_flash_sd_kde_defused(
        data,
        bandwidth,
        block_m=score_block_m,
        block_n=score_block_n,
        device=device,
        synchronize=False,
        use_tensorcores=use_tensorcores,
        return_debug=True,
    )
    train = _to_matrix_tensor(data, torch.device(device), dim=_ND_FEATURES)
    score = (weighted_sum / (pdf_sum[:, None] + 1e-12) - train) * (1.0 / (bandwidth * bandwidth))
    debiased = train + 0.5 * (bandwidth ** 2) * score

    densities, kde_debug = gaussian_kde_triton_nd_defused(
        debiased,
        queries,
        bandwidth,
        block_m=kde_block_m,
        block_n=kde_block_n,
        device=device,
        synchronize=synchronize,
        use_tensorcores=use_tensorcores,
        return_debug=True,
    )

    if return_debug:
        debug = {
            "score_workspace_bytes": score_debug["score_workspace_bytes"],
            "kde_workspace_bytes": kde_debug["kde_workspace_bytes"],
            "workspace_peak_bytes": max(
                score_debug["score_workspace_bytes"],
                kde_debug["kde_workspace_bytes"],
            ),
        }
        return densities, debug
    return densities

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from globals import ND_FEATURES, PRECISION_FAST_TF32, PRECISION_FP32_IEEE
from kernels.reduce_partials import reduce_pdf_partials

_CUDA_MAX_GRID_DIM = 65_535


@triton.jit
def _kde_eval_16d_splitk_pass_a(
    data_ptr,
    query_ptr,
    partial_ptr,
    data_norms_ptr,
    query_norms_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_partial_split,
    stride_partial_query,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_ITER: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_query
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = query_ptr + offs_m[:, None] * stride_query + offs_k[None, :]
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if USE_PRECOMPUTED_NORMS:
        q_norm = tl.load(query_norms_ptr + offs_m, mask=mask_m, other=0.0)
    else:
        q_norm = tl.sum(q_block * q_block, axis=1)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    split_start = pid_s * (BLOCK_N_ITER * ITERS_PER_SPLIT)
    for iter_id in range(0, ITERS_PER_SPLIT):
        offs_n = split_start + iter_id * BLOCK_N_ITER + tl.arange(0, BLOCK_N_ITER)
        mask_n = offs_n < n_data
        d_ptrs = data_ptr + offs_n[:, None] * stride_data + offs_k[None, :]
        d_block = tl.load(d_ptrs, mask=mask_n[:, None], other=0.0)

        if USE_PRECOMPUTED_NORMS:
            d_norm = tl.load(data_norms_ptr + offs_n, mask=mask_n, other=0.0)
        else:
            d_norm = tl.sum(d_block * d_block, axis=1)

        if USE_IEEE:
            dot = tl.dot(q_block, tl.trans(d_block), input_precision="ieee")
        else:
            dot = tl.dot(q_block, tl.trans(d_block), allow_tf32=ALLOW_TF32)

        dist = q_norm[:, None] + d_norm[None, :] - 2.0 * dot
        dist = tl.maximum(dist, 0.0)
        phi = tl.exp(-0.5 * dist * inv_h2)
        phi = tl.where(mask_n[None, :], phi, 0.0)
        acc += tl.sum(phi, axis=1)

    out_ptrs = partial_ptr + pid_s * stride_partial_split + offs_m * stride_partial_query
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 16}, num_warps=4, num_stages=4),
    ],
    key=["n_data", "n_query"],
)
@triton.jit
def _kde_eval_16d_splitk_pass_a_autotune(
    data_ptr,
    query_ptr,
    partial_ptr,
    data_norms_ptr,
    query_norms_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    stride_partial_split,
    stride_partial_query,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_ITER: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _kde_eval_16d_splitk_pass_a(
        data_ptr,
        query_ptr,
        partial_ptr,
        data_norms_ptr,
        query_norms_ptr,
        n_data,
        n_query,
        stride_data,
        stride_query,
        stride_partial_split,
        stride_partial_query,
        inv_h2,
        USE_PRECOMPUTED_NORMS=USE_PRECOMPUTED_NORMS,
        USE_IEEE=USE_IEEE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_M=BLOCK_M,
        BLOCK_N_ITER=BLOCK_N_ITER,
        ITERS_PER_SPLIT=ITERS_PER_SPLIT,
        BLOCK_K=BLOCK_K,
    )


@triton.jit
def _kde_eval_16d_atomic_kernel(
    data_ptr,
    query_ptr,
    out_ptr,
    data_norms_ptr,
    query_norms_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
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

    mask_m = offs_m < n_query
    mask_n = offs_n < n_data

    q_ptrs = query_ptr + offs_m[:, None] * stride_query + offs_k[None, :]
    d_ptrs = data_ptr + offs_n[:, None] * stride_data + offs_k[None, :]
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    d_block = tl.load(d_ptrs, mask=mask_n[:, None], other=0.0)

    if USE_PRECOMPUTED_NORMS:
        q_norm = tl.load(query_norms_ptr + offs_m, mask=mask_m, other=0.0)
        d_norm = tl.load(data_norms_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        q_norm = tl.sum(q_block * q_block, axis=1)
        d_norm = tl.sum(d_block * d_block, axis=1)

    if USE_IEEE:
        dot = tl.dot(q_block, tl.trans(d_block), input_precision="ieee")
    else:
        dot = tl.dot(q_block, tl.trans(d_block), allow_tf32=ALLOW_TF32)

    dist = q_norm[:, None] + d_norm[None, :] - 2.0 * dot
    dist = tl.maximum(dist, 0.0)
    phi = tl.exp(-0.5 * dist * inv_h2)
    phi = tl.where(mask_n[None, :], phi, 0.0)
    acc = tl.sum(phi, axis=1)
    tl.atomic_add(out_ptr + offs_m, acc, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["n_data", "n_query"],
)
@triton.jit
def _kde_eval_16d_atomic_kernel_autotune(
    data_ptr,
    query_ptr,
    out_ptr,
    data_norms_ptr,
    query_norms_ptr,
    n_data,
    n_query,
    stride_data,
    stride_query,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _kde_eval_16d_atomic_kernel(
        data_ptr,
        query_ptr,
        out_ptr,
        data_norms_ptr,
        query_norms_ptr,
        n_data,
        n_query,
        stride_data,
        stride_query,
        inv_h2,
        USE_PRECOMPUTED_NORMS=USE_PRECOMPUTED_NORMS,
        USE_IEEE=USE_IEEE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def _to_matrix_tensor(
    array_like: Sequence[Sequence[float]] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != ND_FEATURES:
        raise ValueError(f"expected tensor with shape (n, {ND_FEATURES}).")
    return tensor.contiguous()


def kde_eval_16d_splitk(
    data: Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    precision_mode: str,
    use_precomputed_norms: bool,
    autotune: bool,
) -> torch.Tensor:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_matrix_tensor(data, device)
    query = _to_matrix_tensor(queries, device)
    n_data = train.shape[0]
    n_query = query.shape[0]
    if n_data == 0 or n_query == 0:
        raise ValueError("data and queries must be non-empty.")

    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    configs = [
        (64, 128, 8),
        (128, 128, 8),
        (64, 256, 8),
        (128, 256, 8),
        (64, 256, 16),
    ]
    default_config = (128, 256, 8)
    block_n_total_min = min(bn * iters for _, bn, iters in configs)
    n_splits = int(math.ceil(n_data / block_n_total_min))

    partials = torch.empty((n_splits, n_query), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32

    if use_precomputed_norms:
        data_norms = (train * train).sum(dim=1)
        query_norms = (query * query).sum(dim=1)
    else:
        data_norms = train
        query_norms = query

    min_block_m = min(cfg[0] for cfg in configs)
    max_queries_per_launch = _CUDA_MAX_GRID_DIM * min_block_m

    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        q_chunk = query[q_start:q_end]
        chunk_n_query = q_chunk.shape[0]
        grid = (triton.cdiv(chunk_n_query, min_block_m), n_splits)
        partials_chunk = partials[:, q_start:q_end]

        kernel = _kde_eval_16d_splitk_pass_a_autotune if autotune else _kde_eval_16d_splitk_pass_a
        if autotune:
            kernel[grid](
                train,
                q_chunk,
                partials_chunk,
                data_norms,
                query_norms[q_start:q_end],
                n_data,
                chunk_n_query,
                train.stride(0),
                q_chunk.stride(0),
                partials_chunk.stride(0),
                partials_chunk.stride(1),
                inv_h2,
                USE_PRECOMPUTED_NORMS=use_precomputed_norms,
                USE_IEEE=use_ieee,
                ALLOW_TF32=allow_tf32,
                BLOCK_K=ND_FEATURES,
            )
        else:
            block_m, block_n_iter, iters_per_split = default_config
            kernel[grid](
                train,
                q_chunk,
                partials_chunk,
                data_norms,
                query_norms[q_start:q_end],
                n_data,
                chunk_n_query,
                train.stride(0),
                q_chunk.stride(0),
                partials_chunk.stride(0),
                partials_chunk.stride(1),
                inv_h2,
                USE_PRECOMPUTED_NORMS=use_precomputed_norms,
                USE_IEEE=use_ieee,
                ALLOW_TF32=allow_tf32,
                BLOCK_M=block_m,
                BLOCK_N_ITER=block_n_iter,
                ITERS_PER_SPLIT=iters_per_split,
                BLOCK_K=ND_FEATURES,
                num_warps=4,
                num_stages=3,
            )

    pdf_sum = reduce_pdf_partials(partials)
    return pdf_sum


def kde_eval_16d_atomic(
    data: Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    precision_mode: str,
    use_precomputed_norms: bool,
    block_m: int = 64,
    block_n: int = 64,
    autotune: bool = False,
) -> torch.Tensor:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_matrix_tensor(data, device)
    query = _to_matrix_tensor(queries, device)
    n_data = train.shape[0]
    n_query = query.shape[0]

    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    if use_precomputed_norms:
        data_norms = (train * train).sum(dim=1)
        query_norms = (query * query).sum(dim=1)
    else:
        data_norms = train
        query_norms = query

    output = torch.zeros((n_query,), device=device, dtype=torch.float32)
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32

    if autotune:
        grid = lambda meta: (
            triton.cdiv(n_query, meta["BLOCK_M"]),
            triton.cdiv(n_data, meta["BLOCK_N"]),
        )
        _kde_eval_16d_atomic_kernel_autotune[grid](
            train,
            query,
            output,
            data_norms,
            query_norms,
            n_data,
            n_query,
            train.stride(0),
            query.stride(0),
            inv_h2,
            USE_PRECOMPUTED_NORMS=use_precomputed_norms,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            BLOCK_K=ND_FEATURES,
        )
    else:
        grid = (triton.cdiv(n_query, block_m), triton.cdiv(n_data, block_n))
        _kde_eval_16d_atomic_kernel[grid](
            train,
            query,
            output,
            data_norms,
            query_norms,
            n_data,
            n_query,
            train.stride(0),
            query.stride(0),
            inv_h2,
            USE_PRECOMPUTED_NORMS=use_precomputed_norms,
            USE_IEEE=use_ieee,
            ALLOW_TF32=allow_tf32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=ND_FEATURES,
        )
    return output

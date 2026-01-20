from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from typing import Sequence, Tuple

import torch
import triton
import triton.language as tl

from globals import ND_FEATURES, PRECISION_FAST_TF32, PRECISION_FP32_IEEE
from kernels.reduce_partials import reduce_pdf_partials, reduce_weighted_partials

_CUDA_MAX_GRID_DIM = 65_535


@triton.jit
def _emp_score_16d_splitk_pass_a(
    data_ptr,
    partial_pdf_ptr,
    partial_weighted_ptr,
    data_norms_ptr,
    n_data,
    stride_data,
    stride_partial_pdf_split,
    stride_partial_pdf_query,
    stride_partial_weighted_split,
    stride_partial_weighted_query,
    stride_partial_weighted_k,
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
    mask_m = offs_m < n_data
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = data_ptr + offs_m[:, None] * stride_data + offs_k[None, :]
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if USE_PRECOMPUTED_NORMS:
        q_norm = tl.load(data_norms_ptr + offs_m, mask=mask_m, other=0.0)
    else:
        q_norm = tl.sum(q_block * q_block, axis=1)

    acc_pdf = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_weighted = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

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
        phi = tl.where(mask_m[:, None], phi, 0.0)

        acc_pdf += tl.sum(phi, axis=1)
        if USE_IEEE:
            acc_weighted += tl.dot(phi, d_block, input_precision="ieee")
        else:
            acc_weighted += tl.dot(phi, d_block, allow_tf32=ALLOW_TF32)

    pdf_ptrs = partial_pdf_ptr + pid_s * stride_partial_pdf_split + offs_m * stride_partial_pdf_query
    tl.store(pdf_ptrs, acc_pdf, mask=mask_m)

    weighted_ptrs = (
        partial_weighted_ptr
        + pid_s * stride_partial_weighted_split
        + offs_m[:, None] * stride_partial_weighted_query
        + offs_k[None, :] * stride_partial_weighted_k
    )
    tl.store(weighted_ptrs, acc_weighted, mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 16}, num_warps=4, num_stages=2),
    ],
    key=["n_data"],
)
@triton.jit
def _emp_score_16d_splitk_pass_a_autotune(
    data_ptr,
    partial_pdf_ptr,
    partial_weighted_ptr,
    data_norms_ptr,
    n_data,
    stride_data,
    stride_partial_pdf_split,
    stride_partial_pdf_query,
    stride_partial_weighted_split,
    stride_partial_weighted_query,
    stride_partial_weighted_k,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_ITER: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _emp_score_16d_splitk_pass_a(
        data_ptr,
        partial_pdf_ptr,
        partial_weighted_ptr,
        data_norms_ptr,
        n_data,
        stride_data,
        stride_partial_pdf_split,
        stride_partial_pdf_query,
        stride_partial_weighted_split,
        stride_partial_weighted_query,
        stride_partial_weighted_k,
        inv_h2,
        USE_PRECOMPUTED_NORMS=USE_PRECOMPUTED_NORMS,
        USE_IEEE=USE_IEEE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_M=BLOCK_M,
        BLOCK_N_ITER=BLOCK_N_ITER,
        ITERS_PER_SPLIT=ITERS_PER_SPLIT,
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


def emp_score_16d_ordered_splitk(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    precision_mode: str,
    use_precomputed_norms: bool,
    autotune: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_matrix_tensor(data, device)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must be non-empty.")

    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    configs = [
        (64, 128, 8),
        (128, 128, 8),
        (64, 256, 8),
        (128, 256, 8),
        (64, 256, 16),
    ]
    default_config = (64, 128, 8)
    block_n_total_min = min(bn * iters for _, bn, iters in configs)
    n_splits = int(math.ceil(n_data / block_n_total_min))

    partial_pdf = torch.empty((n_splits, n_data), device=device, dtype=torch.float32)
    partial_weighted = torch.empty((n_splits, n_data, ND_FEATURES), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32

    if use_precomputed_norms:
        data_norms = (train * train).sum(dim=1)
    else:
        data_norms = train

    min_block_m = min(cfg[0] for cfg in configs)
    max_queries_per_launch = _CUDA_MAX_GRID_DIM * min_block_m

    for q_start in range(0, n_data, max_queries_per_launch):
        q_end = min(n_data, q_start + max_queries_per_launch)
        chunk_n_query = q_end - q_start
        grid = (triton.cdiv(chunk_n_query, min_block_m), n_splits)
        partial_pdf_chunk = partial_pdf[:, q_start:q_end]
        partial_weighted_chunk = partial_weighted[:, q_start:q_end, :]

        kernel = _emp_score_16d_splitk_pass_a_autotune if autotune else _emp_score_16d_splitk_pass_a
        if autotune:
            kernel[grid](
                train,
                partial_pdf_chunk,
                partial_weighted_chunk,
                data_norms,
                n_data,
                train.stride(0),
                partial_pdf_chunk.stride(0),
                partial_pdf_chunk.stride(1),
                partial_weighted_chunk.stride(0),
                partial_weighted_chunk.stride(1),
                partial_weighted_chunk.stride(2),
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
                partial_pdf_chunk,
                partial_weighted_chunk,
                data_norms,
                n_data,
                train.stride(0),
                partial_pdf_chunk.stride(0),
                partial_pdf_chunk.stride(1),
                partial_weighted_chunk.stride(0),
                partial_weighted_chunk.stride(1),
                partial_weighted_chunk.stride(2),
                inv_h2,
                USE_PRECOMPUTED_NORMS=use_precomputed_norms,
                USE_IEEE=use_ieee,
                ALLOW_TF32=allow_tf32,
                BLOCK_M=block_m,
                BLOCK_N_ITER=block_n_iter,
                ITERS_PER_SPLIT=iters_per_split,
                BLOCK_K=ND_FEATURES,
                num_warps=4,
                num_stages=2,
            )

    pdf_sum = reduce_pdf_partials(partial_pdf)
    weighted_sum = reduce_weighted_partials(partial_weighted)
    return pdf_sum, weighted_sum

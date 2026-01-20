from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from kernels.reduce_partials import reduce_pdf_partials

_CUDA_MAX_GRID_DIM = 65_535


@triton.jit
def _kde_eval_1d_splitk_pass_a(
    data_ptr,
    query_ptr,
    partial_ptr,
    n_data,
    n_query,
    inv_bandwidth,
    stride_partial_split,
    stride_partial_query,
    BLOCK_M: tl.constexpr,
    BLOCK_N_ITER: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_query

    q = tl.load(query_ptr + offs_m, mask=mask_m, other=0.0)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    split_start = pid_s * (BLOCK_N_ITER * ITERS_PER_SPLIT)
    for iter_id in range(0, ITERS_PER_SPLIT):
        offs_n = split_start + iter_id * BLOCK_N_ITER + tl.arange(0, BLOCK_N_ITER)
        mask_n = offs_n < n_data
        d = tl.load(data_ptr + offs_n, mask=mask_n, other=0.0)
        diff = (q[:, None] - d[None, :]) * inv_bandwidth
        phi = tl.exp(-0.5 * diff * diff)
        phi = tl.where(mask_n[None, :], phi, 0.0)
        acc += tl.sum(phi, axis=1)

    out_ptrs = partial_ptr + pid_s * stride_partial_split + offs_m * stride_partial_query
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 128, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N_ITER": 256, "ITERS_PER_SPLIT": 16}, num_warps=4, num_stages=3),
    ],
    key=["n_data", "n_query"],
)
@triton.jit
def _kde_eval_1d_splitk_pass_a_autotune(
    data_ptr,
    query_ptr,
    partial_ptr,
    n_data,
    n_query,
    inv_bandwidth,
    stride_partial_split,
    stride_partial_query,
    BLOCK_M: tl.constexpr,
    BLOCK_N_ITER: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
):
    _kde_eval_1d_splitk_pass_a(
        data_ptr,
        query_ptr,
        partial_ptr,
        n_data,
        n_query,
        inv_bandwidth,
        stride_partial_split,
        stride_partial_query,
        BLOCK_M=BLOCK_M,
        BLOCK_N_ITER=BLOCK_N_ITER,
        ITERS_PER_SPLIT=ITERS_PER_SPLIT,
    )


@triton.jit
def _kde_eval_1d_atomic_kernel(
    data_ptr,
    query_ptr,
    out_ptr,
    n_data,
    n_query,
    inv_bandwidth,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_query
    mask_n = offs_n < n_data

    q = tl.load(query_ptr + offs_m, mask=mask_m, other=0.0)
    d = tl.load(data_ptr + offs_n, mask=mask_n, other=0.0)
    diff = (q[:, None] - d[None, :]) * inv_bandwidth
    phi = tl.exp(-0.5 * diff * diff)
    phi = tl.where(mask_n[None, :], phi, 0.0)
    acc = tl.sum(phi, axis=1)
    tl.atomic_add(out_ptr + offs_m, acc, mask=mask_m)


def _to_torch_tensor_1d(array_like: Sequence[float] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    if tensor.ndim != 1:
        raise ValueError("expected 1D tensor for KDE eval.")
    return tensor.contiguous()


def kde_eval_1d_splitk(
    data: Sequence[float] | torch.Tensor,
    queries: Sequence[float] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    autotune: bool,
) -> torch.Tensor:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_torch_tensor_1d(data, device)
    query = _to_torch_tensor_1d(queries, device)

    n_data = train.numel()
    n_query = query.numel()
    if n_data == 0 or n_query == 0:
        raise ValueError("data and queries must be non-empty.")

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
    inv_bandwidth = 1.0 / bandwidth

    min_block_m = min(cfg[0] for cfg in configs)
    max_queries_per_launch = _CUDA_MAX_GRID_DIM * min_block_m
    for q_start in range(0, n_query, max_queries_per_launch):
        q_end = min(n_query, q_start + max_queries_per_launch)
        q_chunk = query[q_start:q_end]
        chunk_n_query = q_chunk.numel()
        grid = (triton.cdiv(chunk_n_query, min_block_m), n_splits)
        partials_chunk = partials[:, q_start:q_end]

        kernel = _kde_eval_1d_splitk_pass_a_autotune if autotune else _kde_eval_1d_splitk_pass_a
        if autotune:
            kernel[grid](
                train,
                q_chunk,
                partials_chunk,
                n_data,
                chunk_n_query,
                inv_bandwidth,
                partials_chunk.stride(0),
                partials_chunk.stride(1),
            )
        else:
            block_m, block_n_iter, iters_per_split = default_config
            kernel[grid](
                train,
                q_chunk,
                partials_chunk,
                n_data,
                chunk_n_query,
                inv_bandwidth,
                partials_chunk.stride(0),
                partials_chunk.stride(1),
                BLOCK_M=block_m,
                BLOCK_N_ITER=block_n_iter,
                ITERS_PER_SPLIT=iters_per_split,
                num_warps=4,
                num_stages=2,
            )

    pdf_sum = reduce_pdf_partials(partials)
    return pdf_sum


def kde_eval_1d_atomic(
    data: Sequence[float] | torch.Tensor,
    queries: Sequence[float] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    block_m: int = 128,
    block_n: int = 128,
) -> torch.Tensor:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_torch_tensor_1d(data, device)
    query = _to_torch_tensor_1d(queries, device)
    n_data = train.numel()
    n_query = query.numel()

    output = torch.zeros((n_query,), device=device, dtype=torch.float32)
    inv_bandwidth = 1.0 / bandwidth

    grid = (triton.cdiv(n_query, block_m), triton.cdiv(n_data, block_n))
    _kde_eval_1d_atomic_kernel[grid](
        train,
        query,
        output,
        n_data,
        n_query,
        inv_bandwidth,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return output

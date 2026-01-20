from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_pdf_partials_kernel(
    partial_ptr,
    out_ptr,
    n_splits,
    n_query,
    stride_split,
    stride_query,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_query

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for split_id in range(0, n_splits):
        ptrs = partial_ptr + split_id * stride_split + offs_m * stride_query
        acc += tl.load(ptrs, mask=mask_m, other=0.0)

    tl.store(out_ptr + offs_m, acc, mask=mask_m)


@triton.jit
def _reduce_weighted_partials_kernel(
    partial_ptr,
    out_ptr,
    n_splits,
    n_query,
    stride_split,
    stride_query,
    stride_k,
    out_stride_query,
    out_stride_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < n_query

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for split_id in range(0, n_splits):
        ptrs = (
            partial_ptr
            + split_id * stride_split
            + offs_m[:, None] * stride_query
            + offs_k[None, :] * stride_k
        )
        acc += tl.load(ptrs, mask=mask_m[:, None], other=0.0)

    out_ptrs = out_ptr + offs_m[:, None] * out_stride_query + offs_k[None, :] * out_stride_k
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


def reduce_pdf_partials(partials: torch.Tensor, *, block_m: int = 256) -> torch.Tensor:
    if partials.ndim != 2:
        raise ValueError("partials must be 2D (n_splits, n_query).")
    if partials.device.type != "cuda":
        return partials.sum(dim=0)

    n_splits, n_query = partials.shape
    out = torch.empty((n_query,), device=partials.device, dtype=partials.dtype)
    grid = (triton.cdiv(n_query, block_m),)
    _reduce_pdf_partials_kernel[grid](
        partials,
        out,
        n_splits,
        n_query,
        partials.stride(0),
        partials.stride(1),
        BLOCK_M=block_m,
    )
    return out


def reduce_weighted_partials(
    partials: torch.Tensor,
    *,
    block_m: int = 128,
) -> torch.Tensor:
    if partials.ndim != 3:
        raise ValueError("partials must be 3D (n_splits, n_query, features).")
    if partials.device.type != "cuda":
        return partials.sum(dim=0)

    n_splits, n_query, features = partials.shape
    out = torch.empty((n_query, features), device=partials.device, dtype=partials.dtype)
    grid = (triton.cdiv(n_query, block_m),)
    _reduce_weighted_partials_kernel[grid](
        partials,
        out,
        n_splits,
        n_query,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=features,
    )
    return out

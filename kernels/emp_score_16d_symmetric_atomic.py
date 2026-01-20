from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Sequence, Tuple

import torch
import triton
import triton.language as tl

from globals import ND_FEATURES, PRECISION_FAST_TF32, PRECISION_FP32_IEEE


@triton.jit
def _emp_score_16d_symmetric_atomic_kernel(
    data_ptr,
    pdf_ptr,
    weighted_ptr,
    data_norms_ptr,
    n_data,
    stride_data,
    stride_weighted_query,
    stride_weighted_k,
    inv_h2,
    USE_PRECOMPUTED_NORMS: tl.constexpr,
    USE_IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)

    if pid_i > pid_j:
        return

    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_j * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_i = offs_i < n_data
    mask_j = offs_j < n_data
    offs_k = tl.arange(0, BLOCK_K)

    xi_ptrs = data_ptr + offs_i[:, None] * stride_data + offs_k[None, :]
    xj_ptrs = data_ptr + offs_j[:, None] * stride_data + offs_k[None, :]
    xi = tl.load(xi_ptrs, mask=mask_i[:, None], other=0.0)
    xj = tl.load(xj_ptrs, mask=mask_j[:, None], other=0.0)

    if USE_PRECOMPUTED_NORMS:
        xi_norm = tl.load(data_norms_ptr + offs_i, mask=mask_i, other=0.0)
        xj_norm = tl.load(data_norms_ptr + offs_j, mask=mask_j, other=0.0)
    else:
        xi_norm = tl.sum(xi * xi, axis=1)
        xj_norm = tl.sum(xj * xj, axis=1)

    if USE_IEEE:
        dot = tl.dot(xi, tl.trans(xj), input_precision="ieee")
    else:
        dot = tl.dot(xi, tl.trans(xj), allow_tf32=ALLOW_TF32)

    dist = xi_norm[:, None] + xj_norm[None, :] - 2.0 * dot
    dist = tl.maximum(dist, 0.0)
    phi = tl.exp(-0.5 * dist * inv_h2)
    phi = tl.where(mask_i[:, None] & mask_j[None, :], phi, 0.0)

    if pid_i < pid_j:
        pdf_i = tl.sum(phi, axis=1)
        pdf_j = tl.sum(phi, axis=0)

        if USE_IEEE:
            w_i = tl.dot(phi, xj, input_precision="ieee")
            w_j = tl.dot(tl.trans(phi), xi, input_precision="ieee")
        else:
            w_i = tl.dot(phi, xj, allow_tf32=ALLOW_TF32)
            w_j = tl.dot(tl.trans(phi), xi, allow_tf32=ALLOW_TF32)

        tl.atomic_add(pdf_ptr + offs_i, pdf_i, mask=mask_i)
        tl.atomic_add(pdf_ptr + offs_j, pdf_j, mask=mask_j)

        w_i_ptrs = weighted_ptr + offs_i[:, None] * stride_weighted_query + offs_k[None, :] * stride_weighted_k
        w_j_ptrs = weighted_ptr + offs_j[:, None] * stride_weighted_query + offs_k[None, :] * stride_weighted_k
        tl.atomic_add(w_i_ptrs, w_i, mask=mask_i[:, None])
        tl.atomic_add(w_j_ptrs, w_j, mask=mask_j[:, None])
    else:
        idx = tl.arange(0, BLOCK_M)
        upper_mask = idx[:, None] <= idx[None, :]
        phi_upper = tl.where(upper_mask, phi, 0.0)

        row_sum = tl.sum(phi_upper, axis=1)
        col_sum = tl.sum(phi_upper, axis=0)

        diag_mask = idx[:, None] == idx[None, :]
        diag_vals = tl.sum(tl.where(diag_mask, phi_upper, 0.0), axis=1)

        pdf_vals = row_sum + col_sum - diag_vals

        if USE_IEEE:
            w_row = tl.dot(phi_upper, xi, input_precision="ieee")
            w_col = tl.dot(tl.trans(phi_upper), xi, input_precision="ieee")
        else:
            w_row = tl.dot(phi_upper, xi, allow_tf32=ALLOW_TF32)
            w_col = tl.dot(tl.trans(phi_upper), xi, allow_tf32=ALLOW_TF32)

        w_vals = w_row + w_col - diag_vals[:, None] * xi

        tl.atomic_add(pdf_ptr + offs_i, pdf_vals, mask=mask_i)
        w_ptrs = weighted_ptr + offs_i[:, None] * stride_weighted_query + offs_k[None, :] * stride_weighted_k
        tl.atomic_add(w_ptrs, w_vals, mask=mask_i[:, None])


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


def emp_score_16d_symmetric_atomic(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    precision_mode: str,
    use_precomputed_norms: bool,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    train = _to_matrix_tensor(data, device)
    n_data = train.shape[0]
    if n_data == 0:
        raise ValueError("data must be non-empty.")

    if precision_mode not in {PRECISION_FAST_TF32, PRECISION_FP32_IEEE}:
        raise ValueError(f"invalid precision_mode {precision_mode}.")

    if use_precomputed_norms:
        data_norms = (train * train).sum(dim=1)
    else:
        data_norms = train

    pdf_sum = torch.zeros((n_data,), device=device, dtype=torch.float32)
    weighted_sum = torch.zeros((n_data, ND_FEATURES), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    use_ieee = precision_mode == PRECISION_FP32_IEEE
    allow_tf32 = precision_mode == PRECISION_FAST_TF32

    grid = (triton.cdiv(n_data, block_size), triton.cdiv(n_data, block_size))
    _emp_score_16d_symmetric_atomic_kernel[grid](
        train,
        pdf_sum,
        weighted_sum,
        data_norms,
        n_data,
        train.stride(0),
        weighted_sum.stride(0),
        weighted_sum.stride(1),
        inv_h2,
        USE_PRECOMPUTED_NORMS=use_precomputed_norms,
        USE_IEEE=use_ieee,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=block_size,
        BLOCK_K=ND_FEATURES,
    )
    return pdf_sum, weighted_sum

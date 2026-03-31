import math
from typing import Sequence, Tuple

import numpy as np
import torch

from flash_sd_kde.config import (
    DEFAULT_HEURISTICS,
    select_emp_score_backend,
    validate_emp_score_backend,
    validate_kde_backend,
    validate_precision_mode,
)
from globals import (
    DEFAULT_EPS,
    DEFAULT_EMP_SCORE_BACKEND,
    DEFAULT_KDE_BACKEND,
    DEFAULT_PRECISION_MODE,
    EMP_SCORE_BACKEND_FLASH_SD_KDE,
    EMP_SCORE_BACKEND_ORDERED_SPLITK,
    EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC,
    KDE_BACKEND_ATOMIC,
    KDE_BACKEND_SPLITK_STREAM,
    ND_FEATURES,
)
from kernels.emp_score_16d_ordered_splitk import emp_score_16d_ordered_splitk
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic
from kernels.flash_sd_kde import emp_score_16d_flash_sd_kde
from kernels.flash_sd_kde.padded_nd import (
    emp_score_padded_nd_flash_sd_kde,
    gaussian_kde_triton_padded_nd,
)
from kernels.kde_eval_16d_stream_splitk import (
    kde_eval_16d_atomic,
    kde_eval_16d_splitk,
)
from kernels.kde_eval_1d_splitk import kde_eval_1d_atomic, kde_eval_1d_splitk


def _infer_ndim(data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor) -> int:
    if isinstance(data, torch.Tensor):
        return data.ndim
    if isinstance(data, np.ndarray):
        return data.ndim
    if len(data) == 0:
        return 1
    first = data[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return 2
    return 1


def _infer_feature_dim(data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor) -> int:
    ndim = _infer_ndim(data)
    if ndim == 1:
        return 1
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return int(data.shape[1])
    if len(data) == 0:
        return 1
    first = data[0]
    return len(first)


def kde_eval(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = DEFAULT_PRECISION_MODE,
    kde_backend: str = DEFAULT_KDE_BACKEND,
    use_precomputed_norms: bool = True,
    autotune: bool = True,
    apply_laplacian_correction: bool = False,
    prefer_specialized_dims: bool = True,
) -> torch.Tensor:
    """Evaluate a Gaussian KDE or its Laplacian-corrected linearized variant."""
    validate_precision_mode(precision_mode)
    validate_kde_backend(kde_backend)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("kde_eval currently expects a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested.")

    ndim = _infer_ndim(data)
    query_ndim = _infer_ndim(queries)
    feature_dim = _infer_feature_dim(data)
    query_feature_dim = _infer_feature_dim(queries)
    if feature_dim != query_feature_dim:
        raise ValueError(
            f"data and queries must share feature dimension, got {feature_dim} and {query_feature_dim}."
        )

    use_legacy_1d = prefer_specialized_dims and ndim == 1 and query_ndim == 1
    use_legacy_16d = (
        prefer_specialized_dims
        and ndim == 2
        and query_ndim == 2
        and feature_dim == ND_FEATURES
    )

    if use_legacy_1d:
        n_data = data.shape[0] if isinstance(data, torch.Tensor) else len(data)
        if kde_backend == KDE_BACKEND_SPLITK_STREAM:
            pdf_sum = kde_eval_1d_splitk(
                data,
                queries,
                bandwidth,
                device=device,
                autotune=autotune,
                apply_laplacian_correction=apply_laplacian_correction,
            )
        else:
            pdf_sum = kde_eval_1d_atomic(
                data,
                queries,
                bandwidth,
                device=device,
                apply_laplacian_correction=apply_laplacian_correction,
            )
        norm = 1.0 / (math.sqrt(2.0 * math.pi) * bandwidth)
        return pdf_sum * (norm / n_data)

    if use_legacy_16d:
        if kde_backend == KDE_BACKEND_SPLITK_STREAM:
            pdf_sum = kde_eval_16d_splitk(
                data,
                queries,
                bandwidth,
                device=device,
                precision_mode=precision_mode,
                use_precomputed_norms=use_precomputed_norms,
                autotune=autotune,
                apply_laplacian_correction=apply_laplacian_correction,
            )
        else:
            pdf_sum = kde_eval_16d_atomic(
                data,
                queries,
                bandwidth,
                device=device,
                precision_mode=precision_mode,
                use_precomputed_norms=use_precomputed_norms,
                autotune=autotune,
                apply_laplacian_correction=apply_laplacian_correction,
            )
        n_data = data.shape[0] if isinstance(data, torch.Tensor) else len(data)
        dim = ND_FEATURES
        norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * n_data)
        return pdf_sum * norm

    if ndim not in {1, 2} or query_ndim not in {1, 2}:
        raise ValueError("data and queries must be 1D or 2D.")

    return gaussian_kde_triton_padded_nd(
        data,
        queries,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        apply_laplacian_correction=apply_laplacian_correction,
    )


def kde_eval_linearized(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = DEFAULT_PRECISION_MODE,
    kde_backend: str = DEFAULT_KDE_BACKEND,
    use_precomputed_norms: bool = True,
    autotune: bool = True,
    prefer_specialized_dims: bool = True,
) -> torch.Tensor:
    """Evaluate the linearized Emp-SD-KDE approximation using K - (h^2/2) ΔK."""
    return kde_eval(
        data,
        queries,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
        apply_laplacian_correction=True,
        prefer_specialized_dims=prefer_specialized_dims,
    )


def kde_eval_linearized_nonfused(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    queries: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = DEFAULT_PRECISION_MODE,
    kde_backend: str = DEFAULT_KDE_BACKEND,
    use_precomputed_norms: bool = True,
    autotune: bool = True,
    chunk_size: int | None = None,
    prefer_specialized_dims: bool = True,
) -> torch.Tensor:
    """Evaluate the linearized KDE using a non-fused correction pass.

    This computes the base KDE via the chosen backend, then applies the
    Laplacian correction using explicit Torch ops in a separate pass.
    Intended for benchmarking the benefit of fused kernels.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    device = torch.device(device)

    base = kde_eval(
        data,
        queries,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
        apply_laplacian_correction=False,
        prefer_specialized_dims=prefer_specialized_dims,
    )

    def _to_1d_tensor(array_like) -> torch.Tensor:
        if isinstance(array_like, torch.Tensor):
            tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
        else:
            tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
        if tensor.ndim != 1:
            raise ValueError("expected 1D data for KDE eval.")
        return tensor.contiguous()

    def _to_2d_tensor(array_like) -> torch.Tensor:
        if isinstance(array_like, torch.Tensor):
            tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
        else:
            tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
        if tensor.ndim != 2:
            raise ValueError("expected 2D data for KDE eval.")
        return tensor.contiguous()

    ndim = _infer_ndim(data)
    if ndim == 1:
        train = _to_1d_tensor(data)
        query = _to_1d_tensor(queries)
        n_data = train.numel()
        if n_data == 0:
            raise ValueError("data must be non-empty.")

        inv_h = 1.0 / bandwidth
        chunk = n_data if chunk_size is None else max(int(chunk_size), 1)
        sum_phi_scaled = torch.zeros((query.numel(),), device=device, dtype=torch.float32)

        for start in range(0, n_data, chunk):
            end = min(n_data, start + chunk)
            data_chunk = train[start:end]
            diff = (query[:, None] - data_chunk[None, :]) * inv_h
            scaled = diff * diff
            phi = torch.exp(-0.5 * scaled)
            sum_phi_scaled += (phi * scaled).sum(dim=1)

        norm = 1.0 / (math.sqrt(2.0 * math.pi) * bandwidth * n_data)
        dim = 1.0
        return base * (1.0 + 0.5 * dim) - 0.5 * norm * sum_phi_scaled

    if ndim == 2:
        train = _to_2d_tensor(data)
        query = _to_2d_tensor(queries)
        if train.shape[1] != query.shape[1]:
            raise ValueError(
                f"data and queries must share feature dimension, got {train.shape[1]} and {query.shape[1]}."
            )
        n_data = train.shape[0]
        if n_data == 0:
            raise ValueError("data must be non-empty.")

        inv_h2 = 1.0 / (bandwidth * bandwidth)
        q_norm = (query * query).sum(dim=1, keepdim=True)
        chunk = n_data if chunk_size is None else max(int(chunk_size), 1)
        sum_phi_scaled = torch.zeros((query.shape[0],), device=device, dtype=torch.float32)

        for start in range(0, n_data, chunk):
            end = min(n_data, start + chunk)
            data_chunk = train[start:end]
            d_norm = (data_chunk * data_chunk).sum(dim=1, keepdim=True).T
            dist = q_norm + d_norm - 2.0 * (query @ data_chunk.T)
            dist = torch.clamp(dist, min=0.0)
            scaled = dist * inv_h2
            phi = torch.exp(-0.5 * scaled)
            sum_phi_scaled += (phi * scaled).sum(dim=1)

        dim = float(train.shape[1])
        norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * n_data)
        return base * (1.0 + 0.5 * dim) - 0.5 * norm * sum_phi_scaled

    raise ValueError("data must be 1D or 2D.")


def emp_sd_kde_fit_transform(
    data: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = DEFAULT_PRECISION_MODE,
    emp_score_backend: str | None = None,
    use_precomputed_norms: bool = True,
    autotune: bool = True,
    eps: float = DEFAULT_EPS,
    return_debug: bool = False,
    prefer_specialized_dims: bool = True,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    validate_precision_mode(precision_mode)
    if emp_score_backend is None:
        emp_score_backend = select_emp_score_backend(
            len(data) if not isinstance(data, torch.Tensor) else data.shape[0],
            heuristics=DEFAULT_HEURISTICS,
        )
    validate_emp_score_backend(emp_score_backend)

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("emp_sd_kde_fit_transform currently expects a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested.")

    if isinstance(data, torch.Tensor):
        train = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    else:
        train = torch.as_tensor(data, dtype=torch.float32, device=device).contiguous()
    squeeze_result = False
    if train.ndim == 1:
        train = train.unsqueeze(1)
        squeeze_result = True
    if train.ndim != 2:
        raise ValueError("data must be 1D or 2D.")

    use_legacy_16d = prefer_specialized_dims and train.shape[1] == ND_FEATURES
    if use_legacy_16d and emp_score_backend == EMP_SCORE_BACKEND_FLASH_SD_KDE:
        pdf_sum, weighted_sum = emp_score_16d_flash_sd_kde(
            train,
            bandwidth,
            device=device,
        )
    elif use_legacy_16d and emp_score_backend == EMP_SCORE_BACKEND_ORDERED_SPLITK:
        pdf_sum, weighted_sum = emp_score_16d_ordered_splitk(
            train,
            bandwidth,
            device=device,
            precision_mode=precision_mode,
            use_precomputed_norms=use_precomputed_norms,
            autotune=autotune,
        )
    elif use_legacy_16d and emp_score_backend == EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC:
        pdf_sum, weighted_sum = emp_score_16d_symmetric_atomic(
            train,
            bandwidth,
            device=device,
            precision_mode=precision_mode,
            use_precomputed_norms=use_precomputed_norms,
            autotune=autotune,
        )
    else:
        pdf_sum, weighted_sum = emp_score_padded_nd_flash_sd_kde(
            train,
            bandwidth,
            device=device,
            precision_mode=precision_mode,
        )

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    score = (weighted_sum / (pdf_sum[:, None] + eps) - train) * inv_h2
    delta = 0.5 * (bandwidth ** 2)
    debiased = train + delta * score

    if squeeze_result:
        debiased_out = debiased[:, 0]
        weighted_out = weighted_sum[:, 0]
        score_out = score[:, 0]
    else:
        debiased_out = debiased
        weighted_out = weighted_sum
        score_out = score

    if return_debug:
        return debiased_out, pdf_sum, weighted_out, score_out
    return debiased_out

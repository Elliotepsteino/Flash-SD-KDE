from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

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
    EMP_SCORE_BACKEND_ORDERED_SPLITK,
    EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC,
    KDE_BACKEND_ATOMIC,
    KDE_BACKEND_SPLITK_STREAM,
    ND_FEATURES,
)
from kernels.emp_score_16d_ordered_splitk import emp_score_16d_ordered_splitk
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic
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
) -> torch.Tensor:
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
    if ndim == 1:
        n_data = data.shape[0] if isinstance(data, torch.Tensor) else len(data)
        if kde_backend == KDE_BACKEND_SPLITK_STREAM:
            pdf_sum = kde_eval_1d_splitk(
                data,
                queries,
                bandwidth,
                device=device,
                autotune=autotune,
            )
        else:
            pdf_sum = kde_eval_1d_atomic(
                data,
                queries,
                bandwidth,
                device=device,
            )
        norm = 1.0 / (math.sqrt(2.0 * math.pi) * bandwidth)
        return pdf_sum * (norm / n_data)

    if ndim == 2:
        if kde_backend == KDE_BACKEND_SPLITK_STREAM:
            pdf_sum = kde_eval_16d_splitk(
                data,
                queries,
                bandwidth,
                device=device,
                precision_mode=precision_mode,
                use_precomputed_norms=use_precomputed_norms,
                autotune=autotune,
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
            )
        n_data = data.shape[0] if isinstance(data, torch.Tensor) else len(data)
        dim = ND_FEATURES
        norm = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * (bandwidth ** dim) * n_data)
        return pdf_sum * norm

    raise ValueError("data must be 1D or 2D.")


def emp_sd_kde_fit_transform(
    data: Sequence[Sequence[float]] | torch.Tensor,
    bandwidth: float,
    *,
    device: str | torch.device = "cuda",
    precision_mode: str = DEFAULT_PRECISION_MODE,
    emp_score_backend: str | None = None,
    use_precomputed_norms: bool = True,
    autotune: bool = True,
    eps: float = DEFAULT_EPS,
    return_debug: bool = False,
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

    if emp_score_backend == EMP_SCORE_BACKEND_ORDERED_SPLITK:
        pdf_sum, weighted_sum = emp_score_16d_ordered_splitk(
            data,
            bandwidth,
            device=device,
            precision_mode=precision_mode,
            use_precomputed_norms=use_precomputed_norms,
            autotune=autotune,
        )
    else:
        pdf_sum, weighted_sum = emp_score_16d_symmetric_atomic(
            data,
            bandwidth,
            device=device,
            precision_mode=precision_mode,
            use_precomputed_norms=use_precomputed_norms,
            autotune=autotune,
        )

    if isinstance(data, torch.Tensor):
        train = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    else:
        train = torch.as_tensor(data, dtype=torch.float32, device=device).contiguous()
    if train.ndim != 2 or train.shape[1] != ND_FEATURES:
        raise ValueError(f"expected data shape (n, {ND_FEATURES}).")

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    score = (weighted_sum / (pdf_sum[:, None] + eps) - train) * inv_h2
    delta = 0.5 * (bandwidth ** 2)
    debiased = train + delta * score

    if return_debug:
        return debiased, pdf_sum, weighted_sum, score
    return debiased

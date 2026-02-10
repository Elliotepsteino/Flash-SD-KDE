from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Any

import torch

from flash_sd_kde.kde import emp_sd_kde_fit_transform, kde_eval, kde_eval_linearized
from globals import ND_FEATURES


def _dtype_from_str(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    name = name.lower()
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _validate_shape(x: torch.Tensor, *, name: str) -> None:
    if x.ndim != 2 or x.shape[1] != ND_FEATURES:
        raise ValueError(f"{name} must have shape (n, {ND_FEATURES})")


def flash_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def sd_kde_available() -> bool:
    return flash_available() and emp_sd_kde_fit_transform is not None


def run_flash_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    kernel: str,
    device: torch.device,
    precision_mode: str,
    kde_backend: str,
    use_precomputed_norms: bool,
    autotune: bool,
    compute_dtype: str | None = None,
) -> torch.Tensor:
    if kernel != "gaussian":
        raise ValueError(f"unsupported kernel: {kernel}")

    samples_t = samples
    queries_t = queries
    dtype = _dtype_from_str(compute_dtype)
    if dtype is not None:
        samples_t = samples_t.to(dtype=dtype)
        queries_t = queries_t.to(dtype=dtype)

    _validate_shape(samples_t, name="samples")
    _validate_shape(queries_t, name="queries")

    return kde_eval(
        samples_t,
        queries_t,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
    )


def run_flash_log_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    kernel: str,
    device: torch.device,
    precision_mode: str,
    kde_backend: str,
    use_precomputed_norms: bool,
    autotune: bool,
    compute_dtype: str | None = None,
    eps: float = 1e-30,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    density = run_flash_density(
        samples,
        queries,
        bandwidth=bandwidth,
        kernel=kernel,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
        compute_dtype=compute_dtype,
    )
    nonpos = (density <= 0).float().mean().item()
    logp = torch.log(torch.clamp(density, min=eps))
    if return_aux:
        return logp, {"clamped_fraction": float(nonpos)}
    return logp


def run_flash_linearized_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    device: torch.device,
    precision_mode: str,
    kde_backend: str,
    use_precomputed_norms: bool,
    autotune: bool,
    compute_dtype: str | None = None,
) -> torch.Tensor:
    samples_t = samples
    queries_t = queries
    dtype = _dtype_from_str(compute_dtype)
    if dtype is not None:
        samples_t = samples_t.to(dtype=dtype)
        queries_t = queries_t.to(dtype=dtype)

    _validate_shape(samples_t, name="samples")
    _validate_shape(queries_t, name="queries")

    return kde_eval_linearized(
        samples_t,
        queries_t,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
    )


def run_flash_linearized_log_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    device: torch.device,
    precision_mode: str,
    kde_backend: str,
    use_precomputed_norms: bool,
    autotune: bool,
    compute_dtype: str | None = None,
    eps: float = 1e-30,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    density = run_flash_linearized_density(
        samples,
        queries,
        bandwidth=bandwidth,
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
        compute_dtype=compute_dtype,
    )
    nonpos = (density <= 0).float().mean().item()
    logp = torch.log(torch.clamp(density, min=eps))
    if return_aux:
        return logp, {"clamped_fraction": float(nonpos)}
    return logp


def run_sd_kde_log_density(
    samples: torch.Tensor,
    queries: torch.Tensor,
    *,
    bandwidth: float,
    device: torch.device,
    precision_mode: str,
    emp_score_backend: str | None,
    use_precomputed_norms: bool,
    autotune: bool,
    kde_backend: str,
) -> torch.Tensor:
    if not sd_kde_available():
        raise RuntimeError("SD-KDE not available")
    debiased = emp_sd_kde_fit_transform(
        samples,
        bandwidth,
        device=device,
        precision_mode=precision_mode,
        emp_score_backend=emp_score_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
    )
    return run_flash_log_density(
        debiased,
        queries,
        bandwidth=bandwidth,
        kernel="gaussian",
        device=device,
        precision_mode=precision_mode,
        kde_backend=kde_backend,
        use_precomputed_norms=use_precomputed_norms,
        autotune=autotune,
    )

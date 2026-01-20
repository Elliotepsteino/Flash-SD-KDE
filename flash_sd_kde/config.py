from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass
from typing import Iterable

from globals import (
    DEFAULT_EPS,
    DEFAULT_EMP_SCORE_BACKEND,
    DEFAULT_KDE_BACKEND,
    DEFAULT_PRECISION_MODE,
    EMP_SCORE_BACKEND_ORDERED_SPLITK,
    EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC,
    KDE_BACKEND_ATOMIC,
    KDE_BACKEND_SPLITK_STREAM,
)


@dataclass(frozen=True)
class SplitKParams:
    block_m: int
    block_n_iter: int
    iters_per_split: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class SymmetricParams:
    block_size: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class KernelHeuristics:
    n_sym_max: int = 50_000
    max_partial_weighted_bytes: int = 256 * 1024 * 1024
    eps: float = DEFAULT_EPS
    default_precision_mode: str = DEFAULT_PRECISION_MODE
    default_kde_backend: str = DEFAULT_KDE_BACKEND
    default_emp_score_backend: str = DEFAULT_EMP_SCORE_BACKEND


DEFAULT_HEURISTICS = KernelHeuristics()


def validate_precision_mode(precision_mode: str) -> None:
    valid = {"fast_tf32", "fp32_ieee"}
    if precision_mode not in valid:
        raise ValueError(f"precision_mode must be one of {sorted(valid)}, got {precision_mode}.")


def validate_kde_backend(kde_backend: str) -> None:
    valid = {KDE_BACKEND_SPLITK_STREAM, KDE_BACKEND_ATOMIC}
    if kde_backend not in valid:
        raise ValueError(f"kde_backend must be one of {sorted(valid)}, got {kde_backend}.")


def validate_emp_score_backend(emp_score_backend: str) -> None:
    valid = {EMP_SCORE_BACKEND_ORDERED_SPLITK, EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC}
    if emp_score_backend not in valid:
        raise ValueError(
            f"emp_score_backend must be one of {sorted(valid)}, got {emp_score_backend}."
        )


def select_emp_score_backend(n: int, *, heuristics: KernelHeuristics = DEFAULT_HEURISTICS) -> str:
    if n <= heuristics.n_sym_max:
        return EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC
    return EMP_SCORE_BACKEND_ORDERED_SPLITK


def compute_splitk_splits(
    n_data: int,
    *,
    block_n_iter: int,
    iters_per_split: int,
) -> int:
    block_n_total = block_n_iter * iters_per_split
    if block_n_total <= 0:
        raise ValueError("block_n_total must be positive.")
    return int((n_data + block_n_total - 1) // block_n_total)


def cap_splits_for_memory(
    n_query: int,
    n_splits: int,
    *,
    features: int,
    heuristics: KernelHeuristics = DEFAULT_HEURISTICS,
) -> int:
    if n_splits <= 1:
        return n_splits
    bytes_per_split = n_query * features * 4
    if bytes_per_split <= 0:
        return n_splits
    max_splits = max(1, heuristics.max_partial_weighted_bytes // bytes_per_split)
    return int(min(n_splits, max_splits))


def iter_autotune_candidates(candidates: Iterable[SplitKParams]) -> list[SplitKParams]:
    return list(candidates)

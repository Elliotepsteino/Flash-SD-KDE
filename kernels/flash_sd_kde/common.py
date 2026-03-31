# Copyright (c) 2026, Elliot Epstein.
from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch
import triton

_CUDA_MAX_GRID_DIM_X = 65_535
_CUDA_MAX_GRID_DIM_Y = 65_535
_MAX_BLOCK_TILE = 1024
_ND_FEATURES = 16
_GENERAL_TILE_K = 16


def _to_torch_tensor(
    array_like: Sequence[float] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert array-like data to a contiguous float32 torch tensor on device."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    return tensor.contiguous()


def _to_matrix_tensor(
    array_like: Sequence[Sequence[float]] | torch.Tensor,
    device: torch.device,
    *,
    dim: int,
) -> torch.Tensor:
    """Convert array-like data to (n, dim) float32 tensor on device."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != dim:
        raise ValueError(f"expected tensor with shape (n, {dim}), got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _to_feature_matrix_tensor(
    array_like: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert 1D or 2D feature arrays to a contiguous `(n, d)` float32 tensor."""
    if isinstance(array_like, torch.Tensor):
        tensor = array_like.to(device=device, dtype=torch.float32, copy=False)
    else:
        tensor = torch.as_tensor(array_like, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.contiguous().unsqueeze(1)
    if tensor.ndim == 2:
        return tensor.contiguous()
    raise ValueError(f"expected 1D or 2D tensor-like input, got shape {tuple(tensor.shape)}")


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    return ((value + multiple - 1) // multiple) * multiple


def _pad_feature_matrix(
    tensor: torch.Tensor,
    *,
    tile_multiple: int = _GENERAL_TILE_K,
) -> tuple[torch.Tensor, int, int]:
    """Pad a `(n, d)` tensor with zeros to the next multiple of `tile_multiple`."""
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(tensor.shape)}")
    dim_true = int(tensor.shape[1])
    dim_pad = max(tile_multiple, _round_up_to_multiple(dim_true, tile_multiple))
    if dim_pad == dim_true:
        return tensor.contiguous(), dim_true, dim_pad
    padded = torch.zeros((tensor.shape[0], dim_pad), device=tensor.device, dtype=tensor.dtype)
    padded[:, :dim_true] = tensor
    return padded.contiguous(), dim_true, dim_pad


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length())


def _resolve_launch_shape(
    *,
    n_query: int,
    n_data: int,
    block_m: int,
    block_n: int,
    kernel_name: str,
) -> tuple[int, int, int, int]:
    """Ensure Triton grid dimensions stay within CUDA limits."""
    bm = int(block_m)
    bn = int(block_n)

    if bm <= 0 or bn <= 0:
        raise ValueError(f"{kernel_name}: block sizes must be positive, got ({bm}, {bn}).")

    grid_m = triton.cdiv(n_query, bm)
    if grid_m > _CUDA_MAX_GRID_DIM_X:
        required_bm = math.ceil(n_query / _CUDA_MAX_GRID_DIM_X)
        promoted_bm = max(bm, _next_power_of_two(required_bm))
        promoted_bm = min(_MAX_BLOCK_TILE, promoted_bm)
        promoted_bm = min(promoted_bm, n_query)
        grid_m = triton.cdiv(n_query, promoted_bm)
        if grid_m > _CUDA_MAX_GRID_DIM_X:
            raise ValueError(
                f"{kernel_name}: even block_m={promoted_bm} yields grid_m={grid_m} "
                f"> {_CUDA_MAX_GRID_DIM_X}; split the queries before launching."
            )
        if promoted_bm != bm:
            warnings.warn(
                f"{kernel_name}: promoting block_m from {bm} to {promoted_bm} so that "
                f"grid_m stays within CUDA limit {_CUDA_MAX_GRID_DIM_X}.",
                RuntimeWarning,
                stacklevel=3,
            )
            bm = promoted_bm

    grid_n = triton.cdiv(n_data, bn)
    if grid_n > _CUDA_MAX_GRID_DIM_Y:
        required_bn = math.ceil(n_data / _CUDA_MAX_GRID_DIM_Y)
        promoted_bn = max(bn, _next_power_of_two(required_bn))
        promoted_bn = min(_MAX_BLOCK_TILE, promoted_bn)
        promoted_bn = min(promoted_bn, n_data)
        grid_n = triton.cdiv(n_data, promoted_bn)
        if grid_n > _CUDA_MAX_GRID_DIM_Y:
            raise ValueError(
                f"{kernel_name}: even block_n={promoted_bn} yields grid_n={grid_n} "
                f"> {_CUDA_MAX_GRID_DIM_Y}; split the training data before launching."
            )
        if promoted_bn != bn:
            warnings.warn(
                f"{kernel_name}: promoting block_n from {bn} to {promoted_bn} so that "
                f"grid_n stays within CUDA limit {_CUDA_MAX_GRID_DIM_Y}.",
                RuntimeWarning,
                stacklevel=3,
            )
            bn = promoted_bn

    return bm, bn, grid_m, grid_n

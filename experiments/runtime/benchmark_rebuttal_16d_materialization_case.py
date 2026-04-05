"""Single-case 16-D streaming vs full-materialization benchmark for the rebuttal."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import (
    empirical_sd_kde_triton_nd,
    empirical_sd_kde_triton_nd_no_tensorcore,
    gaussian_kde_triton_nd,
    gaussian_kde_triton_nd_no_tensorcore,
)

_METHODS = (
    ("streamed_tc", "Streamed Flash + Tensor Cores"),
    ("streamed_no_tc", "Streamed Flash + No Tensor Cores"),
    ("materialized_tc", "Full Materialization + Tensor Cores"),
    ("materialized_no_tc", "Full Materialization + No Tensor Cores"),
)


def _configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


@contextmanager
def _torch_mm_mode(*, use_tensorcores: bool):
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_precision = None
    if hasattr(torch, "get_float32_matmul_precision"):
        prev_precision = torch.get_float32_matmul_precision()

    torch.backends.cuda.matmul.allow_tf32 = bool(use_tensorcores)
    torch.backends.cudnn.allow_tf32 = bool(use_tensorcores)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if use_tensorcores else "highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        if prev_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(prev_precision)


def _fmt_float(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def _to_mb(value_bytes: float) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def _measure_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, float, float, np.ndarray]:
    for _ in range(max(warmup, 0)):
        _ = fn()
        torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    base_alloc = float(torch.cuda.memory_allocated(device))
    base_reserved = float(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    values = None
    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        values = fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    assert values is not None
    peak_alloc = float(torch.cuda.max_memory_allocated(device))
    peak_reserved = float(torch.cuda.max_memory_reserved(device))
    arr = np.asarray(times_ms, dtype=float)
    return (
        float(arr.mean()),
        float(arr.std(ddof=0)),
        _to_mb(max(0.0, peak_alloc - base_alloc)),
        _to_mb(max(0.0, peak_reserved - base_reserved)),
        float(arr.min()),
        values.detach().cpu().numpy().ravel(),
    )


def _streamed_flash(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    use_tensorcores: bool,
) -> torch.Tensor:
    if use_tensorcores:
        debiased, _ = empirical_sd_kde_triton_nd(
            train,
            bandwidth,
            device=device,
            return_tensor=True,
            synchronize=False,
        )
        return gaussian_kde_triton_nd(
            debiased,
            queries,
            bandwidth,
            device=device,
            synchronize=False,
        )

    debiased, _ = empirical_sd_kde_triton_nd_no_tensorcore(
        train,
        bandwidth,
        device=device,
        return_tensor=True,
        synchronize=False,
    )
    return gaussian_kde_triton_nd_no_tensorcore(
        debiased,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
    )


def _materialized_flash(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    use_tensorcores: bool,
) -> torch.Tensor:
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    delta = 0.5 * (bandwidth ** 2)

    with _torch_mm_mode(use_tensorcores=use_tensorcores):
        x_norm = (train * train).sum(dim=1)
        gram = train @ train.T
        gram.mul_(-2.0)
        gram += x_norm[:, None]
        gram += x_norm[None, :]
        gram.clamp_(min=0.0)
        gram.mul_(-0.5 * inv_h2)
        gram.exp_()

        pdf_sum = gram.sum(dim=1, keepdim=True)
        weighted = gram @ train
        score = (weighted / (pdf_sum + 1e-12) - train) * inv_h2
        debiased = train + delta * score

        del gram, pdf_sum, weighted, score

        q_norm = (queries * queries).sum(dim=1, keepdim=True)
        d_norm = (debiased * debiased).sum(dim=1, keepdim=True).T
        qgram = queries @ debiased.T
        qgram.mul_(-2.0)
        qgram += q_norm
        qgram += d_norm
        qgram.clamp_(min=0.0)
        qgram.mul_(-0.5 * inv_h2)
        qgram.exp_()

        out = qgram.sum(dim=1)
        dim = train.shape[1]
        norm = 1.0 / (((2.0 * math.pi) ** (dim / 2.0)) * (bandwidth ** dim) * train.shape[0])
        out.mul_(norm)
        return out


def benchmark_case(
    *,
    n_train: int,
    n_test: int,
    device: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for materialization benchmark.")
    _configure_torch_precision()

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()
    queries_t = torch.as_tensor(queries_np, device=torch_device, dtype=torch.float32).contiguous()

    train_train_kernel_peak_mb = _to_mb(float(n_train) * float(n_train) * 4.0)
    query_train_kernel_peak_mb = _to_mb(float(n_test) * float(n_train) * 4.0)
    materialized_kernel_peak_mb = max(train_train_kernel_peak_mb, query_train_kernel_peak_mb)

    method_fns: dict[str, Callable[[], torch.Tensor]] = {
        "streamed_tc": lambda: _streamed_flash(
            train_t, queries_t, bandwidth, device=torch_device, use_tensorcores=True
        ),
        "streamed_no_tc": lambda: _streamed_flash(
            train_t, queries_t, bandwidth, device=torch_device, use_tensorcores=False
        ),
        "materialized_tc": lambda: _materialized_flash(
            train_t, queries_t, bandwidth, use_tensorcores=True
        ),
        "materialized_no_tc": lambda: _materialized_flash(
            train_t, queries_t, bandwidth, use_tensorcores=False
        ),
    }

    row: dict[str, float] = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "bandwidth": bandwidth,
        "train_train_kernel_peak_mb": train_train_kernel_peak_mb,
        "query_train_kernel_peak_mb": query_train_kernel_peak_mb,
        "materialized_kernel_peak_mb": materialized_kernel_peak_mb,
    }
    outputs: dict[str, np.ndarray] = {}

    for key, _ in _METHODS:
        mean_ms, std_ms, peak_alloc_mb, peak_reserved_mb, min_ms, values = _measure_cuda(
            method_fns[key],
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
        )
        outputs[key] = values
        row[f"{key}_ms"] = mean_ms
        row[f"{key}_ms_std"] = std_ms
        row[f"{key}_ms_min"] = min_ms
        row[f"{key}_peak_extra_allocated_mb"] = peak_alloc_mb
        row[f"{key}_peak_extra_reserved_mb"] = peak_reserved_mb
        row[f"{key}_explicit_materialized_kernel_mb"] = materialized_kernel_peak_mb if key.startswith("materialized") else 0.0

    row["speedup_streamed_vs_materialized_tc"] = (
        row["materialized_tc_ms"] / row["streamed_tc_ms"] if row["streamed_tc_ms"] > 0 else float("inf")
    )
    row["speedup_streamed_vs_materialized_no_tc"] = (
        row["materialized_no_tc_ms"] / row["streamed_no_tc_ms"] if row["streamed_no_tc_ms"] > 0 else float("inf")
    )
    row["memory_reduction_streamed_vs_materialized_tc"] = (
        row["materialized_tc_peak_extra_allocated_mb"] / row["streamed_tc_peak_extra_allocated_mb"]
        if row["streamed_tc_peak_extra_allocated_mb"] > 0
        else float("inf")
    )
    row["memory_reduction_streamed_vs_materialized_no_tc"] = (
        row["materialized_no_tc_peak_extra_allocated_mb"] / row["streamed_no_tc_peak_extra_allocated_mb"]
        if row["streamed_no_tc_peak_extra_allocated_mb"] > 0
        else float("inf")
    )
    row["speedup_tensorcore_within_streamed"] = (
        row["streamed_no_tc_ms"] / row["streamed_tc_ms"] if row["streamed_tc_ms"] > 0 else float("inf")
    )
    row["speedup_tensorcore_within_materialized"] = (
        row["materialized_no_tc_ms"] / row["materialized_tc_ms"] if row["materialized_tc_ms"] > 0 else float("inf")
    )
    row["delta_max_materialized_tc_vs_streamed_tc"] = float(
        np.max(np.abs(outputs["materialized_tc"] - outputs["streamed_tc"]))
    )
    row["delta_max_materialized_no_tc_vs_streamed_no_tc"] = float(
        np.max(np.abs(outputs["materialized_no_tc"] - outputs["streamed_no_tc"]))
    )
    row["rel_l2_materialized_tc_vs_streamed_tc"] = float(
        np.linalg.norm(outputs["materialized_tc"] - outputs["streamed_tc"])
        / (np.linalg.norm(outputs["streamed_tc"]) + 1e-12)
    )
    row["rel_l2_materialized_no_tc_vs_streamed_no_tc"] = float(
        np.linalg.norm(outputs["materialized_no_tc"] - outputs["streamed_no_tc"])
        / (np.linalg.norm(outputs["streamed_no_tc"]) + 1e-12)
    )

    print(
        f"[Materialization Case 16D] n_train={n_train}, n_test={n_test} | "
        f"streamed+TC={row['streamed_tc_ms']:.2f} ms, {row['streamed_tc_peak_extra_allocated_mb']:.2f} MB | "
        f"streamed+noTC={row['streamed_no_tc_ms']:.2f} ms, {row['streamed_no_tc_peak_extra_allocated_mb']:.2f} MB | "
        f"materialized+TC={row['materialized_tc_ms']:.2f} ms, {row['materialized_tc_peak_extra_allocated_mb']:.2f} MB | "
        f"materialized+noTC={row['materialized_no_tc_ms']:.2f} ms, {row['materialized_no_tc_peak_extra_allocated_mb']:.2f} MB"
    )
    print(
        "  Derived:"
        f" streamed/materialized speedup TC={row['speedup_streamed_vs_materialized_tc']:.2f}x,"
        f" noTC={row['speedup_streamed_vs_materialized_no_tc']:.2f}x |"
        f" memory reduction TC={row['memory_reduction_streamed_vs_materialized_tc']:.2f}x,"
        f" noTC={row['memory_reduction_streamed_vs_materialized_no_tc']:.2f}x"
    )
    return row


def format_markdown(payload: dict[str, object]) -> str:
    row = payload["row"]
    lines = [
        "## Full Metrics",
        "",
        "| Method | Runtime (ms) | Peak Extra Alloc (MB) | Peak Extra Reserved (MB) | Explicit Materialized Kernel Peak (MB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in _METHODS:
        lines.append(
            f"| {label} | {_fmt_float(row[f'{key}_ms'])} | "
            f"{_fmt_float(row[f'{key}_peak_extra_allocated_mb'])} | "
            f"{_fmt_float(row[f'{key}_peak_extra_reserved_mb'])} | "
            f"{_fmt_float(row[f'{key}_explicit_materialized_kernel_mb'])} |"
        )

    lines.extend(
        [
            "",
            "## Derived Comparisons",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Streamed / Materialized speedup (Tensor Cores) | {_fmt_float(row['speedup_streamed_vs_materialized_tc'])}x |",
            f"| Streamed / Materialized speedup (No Tensor Cores) | {_fmt_float(row['speedup_streamed_vs_materialized_no_tc'])}x |",
            f"| Streamed / Materialized memory reduction (Tensor Cores) | {_fmt_float(row['memory_reduction_streamed_vs_materialized_tc'])}x |",
            f"| Streamed / Materialized memory reduction (No Tensor Cores) | {_fmt_float(row['memory_reduction_streamed_vs_materialized_no_tc'])}x |",
            f"| Tensor Core speedup within streamed path | {_fmt_float(row['speedup_tensorcore_within_streamed'])}x |",
            f"| Tensor Core speedup within materialized path | {_fmt_float(row['speedup_tensorcore_within_materialized'])}x |",
            "",
            "## Correctness Checks",
            "",
            "| Comparison | max abs delta | rel-L2 |",
            "| --- | ---: | ---: |",
            f"| Materialized vs streamed (Tensor Cores) | {row['delta_max_materialized_tc_vs_streamed_tc']:.3e} | {row['rel_l2_materialized_tc_vs_streamed_tc']:.3e} |",
            f"| Materialized vs streamed (No Tensor Cores) | {row['delta_max_materialized_no_tc_vs_streamed_no_tc']:.3e} | {row['rel_l2_materialized_no_tc_vs_streamed_no_tc']:.3e} |",
            "",
            "Notes:",
            f"- This experiment uses the single large case `n_train={int(row['n_train'])}`, `n_test={int(row['n_test'])}`.",
            "- `Peak Extra Alloc` and `Peak Extra Reserved` are measured above the resident input-tensor footprint.",
            "- The materialized variants explicitly build the full `n_train x n_train` train kernel matrix and the `n_test x n_train` query kernel matrix.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-case 16-D streaming vs materialization benchmark for the rebuttal."
    )
    parser.add_argument("--n-train", type=int, default=65536)
    parser.add_argument("--n-test", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    row = benchmark_case(
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = {
        "figure": "rebuttal_16d_materialization_case",
        "device": args.device,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "row": row,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote materialization benchmark JSON to {args.output}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown(payload), encoding="utf-8")
        print(f"Wrote materialization benchmark markdown to {args.markdown_output}")


if __name__ == "__main__":
    main()

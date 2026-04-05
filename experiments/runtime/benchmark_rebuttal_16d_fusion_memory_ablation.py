"""Benchmark 16-D Flash-SD-KDE fusion/memory ablation with and without Tensor Cores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmarks.exact_kde_baselines import time_cuda_ms
from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import (
    empirical_sd_kde_triton_nd,
    empirical_sd_kde_triton_nd_no_tensorcore,
    gaussian_kde_triton_nd,
    gaussian_kde_triton_nd_no_tensorcore,
)
from kernels.flash_sd_kde.defused_workspace import flash_sd_kde_triton_nd_defused

_METHODS = (
    ("flash_fused_tc", "Fused + Tensor Cores"),
    ("flash_fused_no_tc", "Fused + No Tensor Cores"),
    ("flash_defused_tc", "De-fused + Tensor Cores"),
    ("flash_defused_no_tc", "De-fused + No Tensor Cores"),
)


def _configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _fmt_float(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def _to_mb(value_bytes: float) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def _capture_cuda_memory(
    fn: Callable[[], tuple[torch.Tensor, dict[str, float]]],
    *,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    base_alloc = float(torch.cuda.memory_allocated(device))
    base_reserved = float(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    values, debug = fn()
    torch.cuda.synchronize(device)

    peak_alloc = float(torch.cuda.max_memory_allocated(device))
    peak_reserved = float(torch.cuda.max_memory_reserved(device))
    stats = {
        "memory_base_allocated_mb": _to_mb(base_alloc),
        "memory_base_reserved_mb": _to_mb(base_reserved),
        "memory_peak_allocated_mb": _to_mb(peak_alloc),
        "memory_peak_reserved_mb": _to_mb(peak_reserved),
        "memory_peak_extra_allocated_mb": _to_mb(max(0.0, peak_alloc - base_alloc)),
        "memory_peak_extra_reserved_mb": _to_mb(max(0.0, peak_reserved - base_reserved)),
        "workspace_peak_mb": _to_mb(float(debug.get("workspace_peak_bytes", 0.0))),
        "score_workspace_peak_mb": _to_mb(float(debug.get("score_workspace_bytes", 0.0))),
        "kde_workspace_peak_mb": _to_mb(float(debug.get("kde_workspace_bytes", 0.0))),
    }
    values_np = values.detach().cpu().numpy().ravel()
    del values
    return stats, values_np


def _fused_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    return_debug: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    debiased, _ = empirical_sd_kde_triton_nd(
        train,
        bandwidth,
        device=device,
        return_tensor=True,
        synchronize=False,
    )
    out = gaussian_kde_triton_nd(
        debiased,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
    )
    if return_debug:
        return out, {"workspace_peak_bytes": 0.0}
    return out


def _fused_no_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    return_debug: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    debiased, _ = empirical_sd_kde_triton_nd_no_tensorcore(
        train,
        bandwidth,
        device=device,
        return_tensor=True,
        synchronize=False,
    )
    out = gaussian_kde_triton_nd_no_tensorcore(
        debiased,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
    )
    if return_debug:
        return out, {"workspace_peak_bytes": 0.0}
    return out


def _defused_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    return_debug: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    return flash_sd_kde_triton_nd_defused(
        train,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
        use_tensorcores=True,
        return_debug=return_debug,
    )


def _defused_no_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
    return_debug: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    return flash_sd_kde_triton_nd_defused(
        train,
        queries,
        bandwidth,
        device=device,
        synchronize=False,
        use_tensorcores=False,
        return_debug=return_debug,
    )


def benchmark_point(
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
        raise RuntimeError("CUDA is required for fusion/memory ablation benchmark.")
    _configure_torch_precision()

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()
    queries_t = torch.as_tensor(queries_np, device=torch_device, dtype=torch.float32).contiguous()

    method_impls = {
        "flash_fused_tc": _fused_tensorcore,
        "flash_fused_no_tc": _fused_no_tensorcore,
        "flash_defused_tc": _defused_tensorcore,
        "flash_defused_no_tc": _defused_no_tensorcore,
    }

    row: dict[str, float] = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "bandwidth": bandwidth,
    }
    outputs: dict[str, np.ndarray] = {}

    for method_key, _ in _METHODS:
        impl = method_impls[method_key]
        runtime_mean, runtime_std, runtime_min, _ = time_cuda_ms(
            lambda: impl(train_t, queries_t, bandwidth, device=torch_device, return_debug=False),
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
        )
        mem_stats, values_np = _capture_cuda_memory(
            lambda: impl(train_t, queries_t, bandwidth, device=torch_device, return_debug=True),
            device=torch_device,
        )
        outputs[method_key] = values_np

        row[f"{method_key}_ms"] = runtime_mean
        row[f"{method_key}_ms_std"] = runtime_std
        row[f"{method_key}_ms_min"] = runtime_min
        for stat_key, stat_val in mem_stats.items():
            row[f"{method_key}_{stat_key}"] = stat_val

    row["speedup_fused_vs_defused_tc"] = (
        row["flash_defused_tc_ms"] / row["flash_fused_tc_ms"] if row["flash_fused_tc_ms"] > 0 else float("inf")
    )
    row["speedup_fused_vs_defused_no_tc"] = (
        row["flash_defused_no_tc_ms"] / row["flash_fused_no_tc_ms"]
        if row["flash_fused_no_tc_ms"] > 0
        else float("inf")
    )
    row["memory_reduction_fused_vs_defused_tc"] = (
        row["flash_defused_tc_memory_peak_extra_allocated_mb"]
        / row["flash_fused_tc_memory_peak_extra_allocated_mb"]
        if row["flash_fused_tc_memory_peak_extra_allocated_mb"] > 0
        else float("inf")
    )
    row["memory_reduction_fused_vs_defused_no_tc"] = (
        row["flash_defused_no_tc_memory_peak_extra_allocated_mb"]
        / row["flash_fused_no_tc_memory_peak_extra_allocated_mb"]
        if row["flash_fused_no_tc_memory_peak_extra_allocated_mb"] > 0
        else float("inf")
    )
    row["speedup_tensorcore_within_fused"] = (
        row["flash_fused_no_tc_ms"] / row["flash_fused_tc_ms"] if row["flash_fused_tc_ms"] > 0 else float("inf")
    )
    row["speedup_tensorcore_within_defused"] = (
        row["flash_defused_no_tc_ms"] / row["flash_defused_tc_ms"]
        if row["flash_defused_tc_ms"] > 0
        else float("inf")
    )
    row["delta_max_defused_tc_vs_fused_tc"] = float(
        np.max(np.abs(outputs["flash_defused_tc"] - outputs["flash_fused_tc"]))
    )
    row["delta_max_defused_no_tc_vs_fused_no_tc"] = float(
        np.max(np.abs(outputs["flash_defused_no_tc"] - outputs["flash_fused_no_tc"]))
    )
    row["rel_l2_defused_tc_vs_fused_tc"] = float(
        np.linalg.norm(outputs["flash_defused_tc"] - outputs["flash_fused_tc"])
        / (np.linalg.norm(outputs["flash_fused_tc"]) + 1e-12)
    )
    row["rel_l2_defused_no_tc_vs_fused_no_tc"] = float(
        np.linalg.norm(outputs["flash_defused_no_tc"] - outputs["flash_fused_no_tc"])
        / (np.linalg.norm(outputs["flash_fused_no_tc"]) + 1e-12)
    )

    print(
        f"[Fusion Memory Ablation 16D] n_train={n_train}, n_test={n_test} | "
        f"fused+TC={row['flash_fused_tc_ms']:.2f} ms, {row['flash_fused_tc_memory_peak_extra_allocated_mb']:.2f} MB | "
        f"fused+noTC={row['flash_fused_no_tc_ms']:.2f} ms, {row['flash_fused_no_tc_memory_peak_extra_allocated_mb']:.2f} MB | "
        f"defused+TC={row['flash_defused_tc_ms']:.2f} ms, {row['flash_defused_tc_memory_peak_extra_allocated_mb']:.2f} MB | "
        f"defused+noTC={row['flash_defused_no_tc_ms']:.2f} ms, {row['flash_defused_no_tc_memory_peak_extra_allocated_mb']:.2f} MB"
    )
    print(
        "  Derived:"
        f" fused/defused speedup TC={row['speedup_fused_vs_defused_tc']:.2f}x,"
        f" noTC={row['speedup_fused_vs_defused_no_tc']:.2f}x |"
        f" memory reduction TC={row['memory_reduction_fused_vs_defused_tc']:.2f}x,"
        f" noTC={row['memory_reduction_fused_vs_defused_no_tc']:.2f}x"
    )
    return row


def benchmark_sweep(
    *,
    start_power: int,
    end_power: int,
    device: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    rows = []
    for power in range(start_power, end_power + 1):
        n_train = 1 << power
        n_test = max(1, n_train // 8)
        rows.append(
            benchmark_point(
                n_train=n_train,
                n_test=n_test,
                device=device,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
            )
        )
    return {
        "figure": "rebuttal_16d_fusion_memory_ablation",
        "device": device,
        "seed": seed,
        "start_power": start_power,
        "end_power": end_power,
        "warmup": warmup,
        "repeats": repeats,
        "rows": rows,
    }


def format_markdown_tables(payload: dict[str, object]) -> str:
    rows = list(payload["rows"])
    lines = [
        "## Full Metrics",
        "",
        "| n_train | n_test | Method | Runtime (ms) | Peak Extra Alloc (MB) | Peak Extra Reserved (MB) | Explicit Workspace Peak (MB) |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for method_key, label in _METHODS:
            lines.append(
                f"| {int(row['n_train'])} | {int(row['n_test'])} | {label} | "
                f"{_fmt_float(row[f'{method_key}_ms'])} | "
                f"{_fmt_float(row[f'{method_key}_memory_peak_extra_allocated_mb'])} | "
                f"{_fmt_float(row[f'{method_key}_memory_peak_extra_reserved_mb'])} | "
                f"{_fmt_float(row[f'{method_key}_workspace_peak_mb'])} |"
            )

    lines.extend(
        [
            "",
            "## Derived Comparisons",
            "",
            "| n_train | n_test | Fused/De-fused Speedup (TC) | Fused/De-fused Speedup (No TC) | Fused/De-fused Memory Reduction (TC) | Fused/De-fused Memory Reduction (No TC) | TC Speedup Within Fused | TC Speedup Within De-fused |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{_fmt_float(row['speedup_fused_vs_defused_tc'])}x | "
            f"{_fmt_float(row['speedup_fused_vs_defused_no_tc'])}x | "
            f"{_fmt_float(row['memory_reduction_fused_vs_defused_tc'])}x | "
            f"{_fmt_float(row['memory_reduction_fused_vs_defused_no_tc'])}x | "
            f"{_fmt_float(row['speedup_tensorcore_within_fused'])}x | "
            f"{_fmt_float(row['speedup_tensorcore_within_defused'])}x |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- `Peak Extra Alloc` and `Peak Extra Reserved` are measured above the resident input-tensor footprint.",
            "- `Explicit Workspace Peak` counts only the intentionally materialized global-memory workspaces in the de-fused implementations; fused variants report `0.00` by construction.",
            "- Fused/De-fused ratios are `de_fused / fused`, so values above `1.00x` mean the fused path is better.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 16-D Flash-SD-KDE fusion/memory ablation with and without Tensor Cores."
    )
    parser.add_argument("--start-power", type=int, default=11)
    parser.add_argument("--end-power", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    args = parser.parse_args()

    payload = benchmark_sweep(
        start_power=args.start_power,
        end_power=args.end_power,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote fusion/memory ablation benchmark JSON to {args.output}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown_tables(payload), encoding="utf-8")
        print(f"Wrote fusion/memory ablation markdown tables to {args.markdown_output}")


if __name__ == "__main__":
    main()

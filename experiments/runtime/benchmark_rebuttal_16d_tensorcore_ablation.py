"""Benchmark Flash-SD-KDE with and without Tensor Core use in 16-D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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


def _flash_sd_kde_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
) -> torch.Tensor:
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


def _flash_sd_kde_no_tensorcore(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
) -> torch.Tensor:
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


def _fmt_float(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


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
        raise RuntimeError("CUDA is required for tensorcore ablation benchmark.")

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()
    queries_t = torch.as_tensor(queries_np, device=torch_device, dtype=torch.float32).contiguous()

    tc_mean, tc_std, tc_min, tc_vals = time_cuda_ms(
        lambda: _flash_sd_kde_tensorcore(train_t, queries_t, bandwidth, device=torch_device),
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )
    no_tc_mean, no_tc_std, no_tc_min, no_tc_vals = time_cuda_ms(
        lambda: _flash_sd_kde_no_tensorcore(train_t, queries_t, bandwidth, device=torch_device),
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )

    tc_vals = tc_vals.ravel()
    no_tc_vals = no_tc_vals.ravel()
    row = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "bandwidth": bandwidth,
        "flash_tensorcore_ms": tc_mean,
        "flash_tensorcore_ms_std": tc_std,
        "flash_tensorcore_ms_min": tc_min,
        "flash_no_tensorcore_ms": no_tc_mean,
        "flash_no_tensorcore_ms_std": no_tc_std,
        "flash_no_tensorcore_ms_min": no_tc_min,
        "speedup_tensorcore_vs_no_tensorcore": no_tc_mean / tc_mean if tc_mean > 0 else float("inf"),
        "delta_max_no_tensorcore_vs_tensorcore": float(np.max(np.abs(no_tc_vals - tc_vals))),
        "rel_l2_no_tensorcore_vs_tensorcore": float(
            np.linalg.norm(no_tc_vals - tc_vals) / (np.linalg.norm(tc_vals) + 1e-12)
        ),
    }

    print(
        f"[Tensorcore Ablation 16D] n_train={n_train}, n_test={n_test}, h={bandwidth:.6e} | "
        f"Flash(TC)={tc_mean:.2f} ms | Flash(no-TC)={no_tc_mean:.2f} ms | "
        f"speedup={row['speedup_tensorcore_vs_no_tensorcore']:.2f}x"
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
        "figure": "rebuttal_16d_tensorcore_ablation",
        "device": device,
        "seed": seed,
        "start_power": start_power,
        "end_power": end_power,
        "warmup": warmup,
        "repeats": repeats,
        "rows": rows,
    }


def format_markdown_table(payload: dict[str, object]) -> str:
    rows = list(payload["rows"])
    lines = [
        "| n_train | n_test | Flash-SD-KDE Tensor Core (ms) | Flash-SD-KDE no Tensor Core (ms) | no-TC / TC speedup | max abs delta | rel-L2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{_fmt_float(row['flash_tensorcore_ms'])} | {_fmt_float(row['flash_no_tensorcore_ms'])} | "
            f"{_fmt_float(row['speedup_tensorcore_vs_no_tensorcore'])}x | "
            f"{row['delta_max_no_tensorcore_vs_tensorcore']:.3e} | "
            f"{row['rel_l2_no_tensorcore_vs_tensorcore']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Lower is better for runtime. The `no-TC / TC speedup` column is",
            "`runtime_without_tensorcores / runtime_with_tensorcores`, so values above `1.00x`",
            "mean the Tensor-Core path is faster.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Flash-SD-KDE with and without Tensor Core use in 16-D."
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
    print(f"Wrote tensorcore ablation benchmark JSON to {args.output}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown_table(payload), encoding="utf-8")
        print(f"Wrote tensorcore ablation markdown table to {args.markdown_output}")


if __name__ == "__main__":
    main()

"""Operator-level ablation and exact torch.compile baseline for 16-D SD-KDE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.exact_kde_baselines import time_cuda_ms, torch_exact_sd_kde_nd
from flash_sd_kde.kde import kde_eval_linearized
from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd


def _flash_sd_kde_nd(
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


def _flash_laplace_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return kde_eval_linearized(
        train,
        queries,
        bandwidth,
        device=device,
        precision_mode="fast_tf32",
        kde_backend="splitk_stream",
        use_precomputed_norms=True,
        autotune=True,
    )


def _compile_available() -> bool:
    return hasattr(torch, "compile")


def _operator_rows() -> list[dict[str, object]]:
    return [
        {
            "method": "Flash-Laplace-KDE",
            "gemm_passes": 1,
            "exp_passes": 1,
            "reduction_passes": 1,
            "atomic_passes": 1,
            "notes": "single fused linearized KDE pass",
        },
        {
            "method": "Flash-SD-KDE",
            "gemm_passes": 2,
            "exp_passes": 2,
            "reduction_passes": 2,
            "atomic_passes": 2,
            "notes": "fused score pass plus fused KDE pass",
        },
        {
            "method": "Exact SD-KDE (Torch eager)",
            "gemm_passes": 3,
            "exp_passes": 2,
            "reduction_passes": 2,
            "atomic_passes": 0,
            "notes": "Gram matrix, weighted sum GEMM, query GEMM",
        },
        {
            "method": "Exact SD-KDE (torch.compile)",
            "gemm_passes": 3,
            "exp_passes": 2,
            "reduction_passes": 2,
            "atomic_passes": 0,
            "notes": "same operator family as eager; better scheduling/codegen",
        },
    ]


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
        raise RuntimeError("CUDA is required for operator ablation benchmark.")

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()
    queries_t = torch.as_tensor(queries_np, device=torch_device, dtype=torch.float32).contiguous()

    eager_mean, eager_std, eager_min, eager_vals = time_cuda_ms(
        lambda: torch_exact_sd_kde_nd(train_t, queries_t, bandwidth),
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )

    if _compile_available():
        compiled_fn = torch.compile(
            lambda x, q: torch_exact_sd_kde_nd(x, q, bandwidth),
            mode="reduce-overhead",
            fullgraph=False,
        )
        compiled_mean, compiled_std, compiled_min, compiled_vals = time_cuda_ms(
            lambda: compiled_fn(train_t, queries_t),
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
        )
    else:
        compiled_mean = float("nan")
        compiled_std = float("nan")
        compiled_min = float("nan")
        compiled_vals = eager_vals

    flash_mean, flash_std, flash_min, flash_vals = time_cuda_ms(
        lambda: _flash_sd_kde_nd(train_t, queries_t, bandwidth, device=torch_device),
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )
    laplace_mean, laplace_std, laplace_min, laplace_vals = time_cuda_ms(
        lambda: _flash_laplace_nd(train_t, queries_t, bandwidth, device=torch_device),
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )

    eager_vals = eager_vals.ravel()
    compiled_vals = compiled_vals.ravel()
    flash_vals = flash_vals.ravel()
    laplace_vals = laplace_vals.ravel()

    row = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "bandwidth": bandwidth,
        "exact_sd_torch_ms": eager_mean,
        "exact_sd_torch_ms_std": eager_std,
        "exact_sd_torch_ms_min": eager_min,
        "exact_sd_compile_ms": compiled_mean,
        "exact_sd_compile_ms_std": compiled_std,
        "exact_sd_compile_ms_min": compiled_min,
        "flash_sd_kde_ms": flash_mean,
        "flash_sd_kde_ms_std": flash_std,
        "flash_sd_kde_ms_min": flash_min,
        "flash_laplace_ms": laplace_mean,
        "flash_laplace_ms_std": laplace_std,
        "flash_laplace_ms_min": laplace_min,
        "compile_speedup_vs_eager": eager_mean / compiled_mean if compiled_mean > 0 else float("nan"),
        "flash_speedup_vs_eager": eager_mean / flash_mean if flash_mean > 0 else float("inf"),
        "laplace_speedup_vs_eager": eager_mean / laplace_mean if laplace_mean > 0 else float("inf"),
        "delta_max_compile_vs_eager": float(np.max(np.abs(compiled_vals - eager_vals))),
        "delta_max_flash_vs_eager": float(np.max(np.abs(flash_vals - eager_vals))),
        "delta_max_laplace_vs_eager": float(np.max(np.abs(laplace_vals - eager_vals))),
        "rel_l2_compile_vs_eager": float(np.linalg.norm(compiled_vals - eager_vals) / (np.linalg.norm(eager_vals) + 1e-12)),
        "rel_l2_flash_vs_eager": float(np.linalg.norm(flash_vals - eager_vals) / (np.linalg.norm(eager_vals) + 1e-12)),
        "rel_l2_laplace_vs_eager": float(np.linalg.norm(laplace_vals - eager_vals) / (np.linalg.norm(eager_vals) + 1e-12)),
    }

    print(
        f"[Operator Ablation] n_train={n_train}, n_test={n_test} | "
        f"eager={eager_mean:.2f} ms | compile={compiled_mean:.2f} ms | "
        f"flash={flash_mean:.2f} ms | laplace={laplace_mean:.2f} ms"
    )
    return row


def render_markdown(payload: dict[str, object]) -> str:
    op_rows = payload["operator_rows"]
    runtime_rows = payload["runtime_rows"]

    lines = [
        "# Operator-Level Ablation",
        "",
        "This report counts major operator-family passes rather than exact low-level instruction counts.",
        "",
        "## Operator Families",
        "",
        "| Method | GEMM passes | Exponential passes | Reduction passes | Atomic passes | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in op_rows:
        lines.append(
            f"| {row['method']} | {row['gemm_passes']} | {row['exp_passes']} | "
            f"{row['reduction_passes']} | {row['atomic_passes']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| n_train | n_test | Exact SD-KDE Torch (ms) | Exact SD-KDE torch.compile (ms) | Flash-SD-KDE (ms) | Flash-Laplace-KDE (ms) | compile speedup | Flash speedup | Laplace speedup |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in runtime_rows:
        lines.append(
            f"| {int(row['n_train'])} | {int(row['n_test'])} | {row['exact_sd_torch_ms']:.2f} | "
            f"{row['exact_sd_compile_ms']:.2f} | {row['flash_sd_kde_ms']:.2f} | {row['flash_laplace_ms']:.2f} | "
            f"{row['compile_speedup_vs_eager']:.2f}x | {row['flash_speedup_vs_eager']:.2f}x | {row['laplace_speedup_vs_eager']:.2f}x |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark operator-level SD-KDE ablation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-power", type=int, default=11)
    parser.add_argument("--end-power", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    runtime_rows = []
    for power in range(args.start_power, args.end_power + 1):
        n_train = 1 << power
        n_test = max(1, n_train // 8)
        runtime_rows.append(
            benchmark_point(
                n_train=n_train,
                n_test=n_test,
                device=args.device,
                seed=args.seed,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    payload = {
        "device": args.device,
        "seed": args.seed,
        "start_power": args.start_power,
        "end_power": args.end_power,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "operator_rows": _operator_rows(),
        "runtime_rows": runtime_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote operator ablation JSON to {args.output}")
    print(f"Wrote operator ablation report to {args.markdown_output}")


if __name__ == "__main__":
    main()

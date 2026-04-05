"""Benchmark 16-D query-level batching at fixed n_train for the rebuttal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.exact_kde_baselines import time_cuda_ms, torch_exact_sd_kde_nd
from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd, gaussian_kde_triton_nd


def _require_pykeops():
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pykeops is not installed or importable. Install `pykeops` to run this benchmark."
        ) from exc
    return LazyTensor


def _compile_available() -> bool:
    return hasattr(torch, "compile")


def _configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _pykeops_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    LazyTensor = _require_pykeops()
    x_i = LazyTensor(queries[:, None, :])
    y_j = LazyTensor(train[None, :, :])
    d_ij = ((x_i - y_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()
    n_train = float(train.shape[0])
    dim = float(train.shape[1])
    norm = (1.0 / bandwidth) ** dim
    norm /= ((2.0 * math.pi) ** (dim / 2.0)) * n_train
    return (norm * k_ij.sum(dim=1)).view(-1)


def _pykeops_sd_kde_nd(
    train: torch.Tensor,
    queries: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    LazyTensor = _require_pykeops()
    x_i = LazyTensor(train[:, None, :])
    x_j = LazyTensor(train[None, :, :])
    d_ij = ((x_i - x_j) ** 2).sum(dim=2)
    k_ij = (-0.5 * d_ij / (bandwidth * bandwidth)).exp()
    pdf_sum = k_ij.sum(dim=1)
    weighted_sum = (k_ij * x_j).sum(dim=1)
    score = (weighted_sum / (pdf_sum + 1e-12) - train) * (1.0 / (bandwidth * bandwidth))
    debiased = train + 0.5 * (bandwidth ** 2) * score
    return _pykeops_kde_nd(debiased, queries, bandwidth)


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


def _fmt_float(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def benchmark_point(
    *,
    train_t: torch.Tensor,
    queries_t: torch.Tensor,
    bandwidth: float,
    n_train: int,
    n_test: int,
    device: torch.device,
    warmup: int,
    flash_repeats: int,
    baseline_repeats: int,
) -> dict[str, float]:
    eager_mean, eager_std, eager_min, eager_vals = time_cuda_ms(
        lambda: torch_exact_sd_kde_nd(train_t, queries_t, bandwidth),
        device=device,
        warmup=warmup,
        repeats=baseline_repeats,
    )

    if _compile_available():
        compiled_fn = torch.compile(
            lambda x, q: torch_exact_sd_kde_nd(x, q, bandwidth),
            mode="reduce-overhead",
            fullgraph=False,
        )
        compile_mean, compile_std, compile_min, compile_vals = time_cuda_ms(
            lambda: compiled_fn(train_t, queries_t),
            device=device,
            warmup=warmup,
            repeats=baseline_repeats,
        )
    else:
        compile_mean = float("nan")
        compile_std = float("nan")
        compile_min = float("nan")
        compile_vals = eager_vals

    pykeops_mean, pykeops_std, pykeops_min, pykeops_vals = time_cuda_ms(
        lambda: _pykeops_sd_kde_nd(train_t, queries_t, bandwidth),
        device=device,
        warmup=warmup,
        repeats=baseline_repeats,
    )
    flash_mean, flash_std, flash_min, flash_vals = time_cuda_ms(
        lambda: _flash_sd_kde_nd(train_t, queries_t, bandwidth, device=device),
        device=device,
        warmup=warmup,
        repeats=flash_repeats,
    )

    eager_vals = eager_vals.ravel()
    compile_vals = compile_vals.ravel()
    pykeops_vals = pykeops_vals.ravel()
    flash_vals = flash_vals.ravel()

    row = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "bandwidth": bandwidth,
        "sd_torch_ms": eager_mean,
        "sd_torch_ms_std": eager_std,
        "sd_torch_ms_min": eager_min,
        "sd_torch_compile_ms": compile_mean,
        "sd_torch_compile_ms_std": compile_std,
        "sd_torch_compile_ms_min": compile_min,
        "sd_pykeops_ms": pykeops_mean,
        "sd_pykeops_ms_std": pykeops_std,
        "sd_pykeops_ms_min": pykeops_min,
        "flash_sd_kde_ms": flash_mean,
        "flash_sd_kde_ms_std": flash_std,
        "flash_sd_kde_ms_min": flash_min,
        "speedup_flash_vs_torch": eager_mean / flash_mean if flash_mean > 0 else float("inf"),
        "speedup_flash_vs_torch_compile": compile_mean / flash_mean if flash_mean > 0 else float("nan"),
        "speedup_flash_vs_pykeops": pykeops_mean / flash_mean if flash_mean > 0 else float("inf"),
        "delta_max_flash_vs_torch": float(np.max(np.abs(flash_vals - eager_vals))),
        "delta_max_flash_vs_torch_compile": float(np.max(np.abs(flash_vals - compile_vals))),
        "delta_max_flash_vs_pykeops": float(np.max(np.abs(flash_vals - pykeops_vals))),
        "rel_l2_flash_vs_torch": float(
            np.linalg.norm(flash_vals - eager_vals) / (np.linalg.norm(eager_vals) + 1e-12)
        ),
        "rel_l2_flash_vs_torch_compile": float(
            np.linalg.norm(flash_vals - compile_vals) / (np.linalg.norm(compile_vals) + 1e-12)
        ),
        "rel_l2_flash_vs_pykeops": float(
            np.linalg.norm(flash_vals - pykeops_vals) / (np.linalg.norm(pykeops_vals) + 1e-12)
        ),
    }

    print(
        f"[Rebuttal Query Sweep 16D] n_train={n_train}, n_test={n_test}, h={bandwidth:.6e} | "
        f"Torch={eager_mean:.2f} ms | compile={compile_mean:.2f} ms | "
        f"PyKeOps={pykeops_mean:.2f} ms | Flash={flash_mean:.2f} ms"
    )
    print(
        "  Flash speedups:"
        f" Torch={row['speedup_flash_vs_torch']:.2f}x |"
        f" compile={row['speedup_flash_vs_torch_compile']:.2f}x |"
        f" PyKeOps={row['speedup_flash_vs_pykeops']:.2f}x"
    )
    return row


def benchmark_sweep(
    *,
    n_train: int,
    n_test_list: list[int],
    device: str,
    seed: int,
    warmup: int,
    flash_repeats: int,
    baseline_repeats: int,
) -> dict[str, object]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    _configure_torch_precision()
    _require_pykeops()

    np.random.seed(seed)
    train_np = sample_gaussian_mixture_16d(n_train)
    max_n_test = max(n_test_list)
    np.random.seed(seed + 10_000)
    queries_np = sample_gaussian_mixture_16d(max_n_test)
    bandwidth = float(silverman_bandwidth_nd(train_np))

    train_t = torch.as_tensor(train_np, device=torch_device, dtype=torch.float32).contiguous()

    rows = []
    for n_test in n_test_list:
        queries_t = torch.as_tensor(
            queries_np[:n_test],
            device=torch_device,
            dtype=torch.float32,
        ).contiguous()
        rows.append(
            benchmark_point(
                train_t=train_t,
                queries_t=queries_t,
                bandwidth=bandwidth,
                n_train=n_train,
                n_test=n_test,
                device=torch_device,
                warmup=warmup,
                flash_repeats=flash_repeats,
                baseline_repeats=baseline_repeats,
            )
        )

    return {
        "figure": "rebuttal_16d_query_batching_sweep",
        "device": device,
        "seed": seed,
        "n_train": n_train,
        "n_test_list": n_test_list,
        "bandwidth": bandwidth,
        "warmup": warmup,
        "flash_repeats": flash_repeats,
        "baseline_repeats": baseline_repeats,
        "rows": rows,
    }


def format_markdown_table(payload: dict[str, object]) -> str:
    rows = list(payload["rows"])
    lines = [
        "| n_train | n_test | Torch (ms) | torch.compile (ms) | PyKeOps (ms) | Flash-SD-KDE (ms) | Torch/Flash | Compile/Flash | PyKeOps/Flash |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{_fmt_float(row['sd_torch_ms'])} | {_fmt_float(row['sd_torch_compile_ms'])} | "
            f"{_fmt_float(row['sd_pykeops_ms'])} | {_fmt_float(row['flash_sd_kde_ms'])} | {_fmt_float(row['speedup_flash_vs_torch'])}x | "
            f"{_fmt_float(row['speedup_flash_vs_torch_compile'])}x | "
            f"{_fmt_float(row['speedup_flash_vs_pykeops'])}x |"
        )
    lines.extend(
        [
            "",
            "Lower is better for runtime. Speedup columns are `baseline_runtime / flash_runtime`,",
            "so values above `1.00x` mean Flash-SD-KDE is faster.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_n_test_list(raw: str) -> list[int]:
    values = [int(tok) for tok in raw.split(",") if tok.strip()]
    if not values:
        raise ValueError("n_test_list must contain at least one positive integer.")
    if any(v <= 0 for v in values):
        raise ValueError("n_test_list values must be positive.")
    return sorted(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-n_train 16-D query-level batching for the rebuttal."
    )
    parser.add_argument("--n-train", type=int, default=32768)
    parser.add_argument("--n-test-list", type=str, default="4,16,64,256,1024,4096,16384")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--flash-repeats", type=int, default=10)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    args = parser.parse_args()

    payload = benchmark_sweep(
        n_train=args.n_train,
        n_test_list=_parse_n_test_list(args.n_test_list),
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        flash_repeats=args.flash_repeats,
        baseline_repeats=args.baseline_repeats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote rebuttal query-sweep benchmark JSON to {args.output}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown_table(payload), encoding="utf-8")
        print(f"Wrote rebuttal query-sweep markdown table to {args.markdown_output}")


if __name__ == "__main__":
    main()

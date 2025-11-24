"""Microbenchmark for the 16-D empirical SD-KDE Triton kernel.

This script isolates `_empirical_sd_kde_kernel_nd` from `triton_kde.py` and
benchmarks it across problem sizes and block configurations using Triton's
testing utilities. It is intended for use with either:

  python nd_score_proton.py

or, if available in your environment:

  proton nd_score_proton.py
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
import triton.testing as ttesting

from triton_kde import (
    _empirical_sd_kde_kernel_nd,
    _ND_FEATURES,
    _resolve_launch_shape,
)


def _launch_empirical_sd_kde_nd(
    x: torch.Tensor,
    *,
    bandwidth: float,
    block_m: int,
    block_n: int,
) -> None:
    """Launch the 16-D empirical SD-KDE kernel once (no host-side postprocessing)."""
    assert x.is_cuda and x.dtype == torch.float32 and x.ndim == 2
    n_data = x.shape[0]
    device = x.device

    pdf_acc = torch.zeros(n_data, device=device, dtype=torch.float32)
    weighted_acc = torch.zeros((n_data, _ND_FEATURES), device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    max_queries_per_launch = max(block_m, block_m * 65535)
    stride_data = x.stride(0)

    for q_start in range(0, n_data, max_queries_per_launch):
        q_end = min(n_data, q_start + max_queries_per_launch)
        query_chunk = x[q_start:q_end]
        pdf_chunk = pdf_acc[q_start:q_end]
        weighted_chunk = weighted_acc[q_start:q_end]
        chunk_n_query = query_chunk.shape[0]

        chunk_block_m, chunk_block_n, grid_m, grid_n = _resolve_launch_shape(
            n_query=chunk_n_query,
            n_data=n_data,
            block_m=block_m,
            block_n=block_n,
            kernel_name="empirical_sd_kde_triton_nd",
        )
        grid = (grid_m, grid_n)

        _empirical_sd_kde_kernel_nd[grid](
            x,
            query_chunk,
            pdf_chunk,
            weighted_chunk,
            n_data,
            chunk_n_query,
            stride_data,
            query_chunk.stride(0),
            weighted_chunk.stride(0),
            inv_h2,
            BLOCK_M=chunk_block_m,
            BLOCK_N=chunk_block_n,
            BLOCK_K=_ND_FEATURES,
            num_warps=4,
            num_stages=2,
        )


@ttesting.perf_report(
    ttesting.Benchmark(
        x_names=["n_train"],
        x_vals=[2**k for k in range(12, 18)],  # 4k → 131k
        line_arg="config",
        line_vals=[
            (64, 64),
            (64, 128),
            (128, 128),
        ],
        line_names=[
            "BLOCK_M=64,BLOCK_N=64",
            "BLOCK_M=64,BLOCK_N=128",
            "BLOCK_M=128,BLOCK_N=128",
        ],
        ylabel="Runtime (ms)",
        plot_name="nd-score-kernel",
        args={},
    )
)
def bench_empirical_sd_kde_nd(n_train: int, config):
    """Benchmark wrapper used by Triton's perf_report / Proton tooling."""
    block_m, block_n = config
    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn((n_train, _ND_FEATURES), device=device, dtype=torch.float32)

    # Simple Silverman-style bandwidth heuristic in 16-D: use per-dim std.
    with torch.no_grad():
        std_per_dim = x.std(dim=0)
        sigma = torch.mean(std_per_dim)
        h = float(0.9 * sigma * n_train ** (-1.0 / (16 + 4)))

    # Warm-up
    _launch_empirical_sd_kde_nd(x, bandwidth=h, block_m=block_m, block_n=block_n)
    torch.cuda.synchronize(device)

    ms = ttesting.do_bench(
        lambda: _launch_empirical_sd_kde_nd(
            x, bandwidth=h, block_m=block_m, block_n=block_n
        )
    )
    return ms


def main():
    # Print a small runtime table to stdout and save plots to the current dir.
    bench_empirical_sd_kde_nd.run(print_data=True, show_plots=False, save_path=".")


if __name__ == "__main__":
    main()

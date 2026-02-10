from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import time

import numpy as np
import torch

from globals import DEFAULT_EPS, DEFAULT_PRECISION_MODE, PRECISION_FAST_TF32, PRECISION_FP32_IEEE
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic
from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
from kernels.flash_sd_kde import empirical_sd_kde_triton_nd


def _time_kernel(fn, *, device: torch.device, warmup: int, repeats: int) -> np.ndarray:
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return np.asarray(times, dtype=float)


def _summarize(label: str, times: np.ndarray) -> None:
    mean_ms = times.mean() * 1e3
    std_ms = times.std() * 1e3
    med_ms = np.median(times) * 1e3
    min_ms = times.min() * 1e3
    max_ms = times.max() * 1e3
    print(
        f"{label}: mean={mean_ms:.3f} ms, std={std_ms:.3f} ms, "
        f"median={med_ms:.3f} ms, min={min_ms:.3f} ms, max={max_ms:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 16D empirical score kernels at a fixed n_train."
    )
    parser.add_argument("--n-train", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision-mode",
        default=DEFAULT_PRECISION_MODE,
        choices=[PRECISION_FAST_TF32, PRECISION_FP32_IEEE],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--no-precomputed-norms", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    np.random.seed(args.seed)
    train_np = sample_gaussian_mixture_16d(args.n_train)
    bandwidth = silverman_bandwidth_nd(train_np)
    train = torch.as_tensor(train_np, device=device, dtype=torch.float32)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    delta = 0.5 * (bandwidth ** 2)
    eps = DEFAULT_EPS

    def run_symmetric() -> torch.Tensor:
        pdf_sum, weighted_sum = emp_score_16d_symmetric_atomic(
            train,
            bandwidth,
            device=device,
            precision_mode=args.precision_mode,
            use_precomputed_norms=not args.no_precomputed_norms,
            block_size=args.block_size,
            autotune=args.autotune,
        )
        score = (weighted_sum / (pdf_sum[:, None] + eps) - train) * inv_h2
        return train + delta * score

    def run_flash_sd_kde() -> torch.Tensor:
        debiased, _ = empirical_sd_kde_triton_nd(
            train,
            bandwidth,
            device=device,
            return_tensor=True,
            synchronize=False,
        )
        return debiased

    print(
        "Empirical score kernel speed test",
        f"(n_train={args.n_train}, device={device}, precision={args.precision_mode})",
    )
    if args.precision_mode != PRECISION_FAST_TF32:
        print("Note: flash_sd_kde uses TF32 matmul; precision_mode only affects symmetric_atomic.")
    print(f"bandwidth={bandwidth:.6f}, repeats={args.repeats}, warmup={args.warmup}")

    times_sym = _time_kernel(run_symmetric, device=device, warmup=args.warmup, repeats=args.repeats)
    times_flash = _time_kernel(run_flash_sd_kde, device=device, warmup=args.warmup, repeats=args.repeats)

    _summarize("symmetric_atomic (full debias)", times_sym)
    _summarize("flash_sd_kde (full debias)", times_flash)
    speedup = times_flash.mean() / times_sym.mean() if times_sym.mean() > 0 else float("inf")
    print(f"speedup (flash_sd_kde / symmetric_atomic): {speedup:.2f}x")


if __name__ == "__main__":
    main()

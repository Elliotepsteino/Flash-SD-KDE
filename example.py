import numpy as np
import time
import torch
from sklearn.neighbors import KernelDensity

import flash_sd_kde


def _time_call_ms(fn, *, repeats: int, cuda_sync: bool) -> float:
    times_ms = []
    for _ in range(repeats):
        if cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda_sync:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1e3)
    return float(np.mean(times_ms))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Flash-SD-KDE example.")

    rng = np.random.default_rng(0)

    # 1D: sklearn KDE vs Flash KDE (same sklearn-style API).
    n_train = 4096
    n_query = 1024
    x_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, 1)).astype(np.float32)
    x_query = rng.normal(loc=0.2, scale=1.2, size=(n_query, 1)).astype(np.float32)

    flash_kde = flash_sd_kde.FlashSDKDE(mode="kde", bandwidth="silverman", device="cuda")
    flash_kde.fit(x_train)

    sk = KernelDensity(kernel="gaussian", bandwidth=flash_kde.bandwidth_)
    sk.fit(x_train)

    sk_log = sk.score_samples(x_query)
    flash_log = flash_kde.score_samples(x_query)
    sk_density = np.exp(sk_log).astype(np.float32)
    flash_density = np.exp(flash_log).astype(np.float32)

    diff = np.abs(sk_density - flash_density)

    print("=== 1D KDE (sklearn vs FlashSDKDE mode='kde') ===")
    print(f"n_train={n_train}, n_query={n_query}, bandwidth={flash_kde.bandwidth_:.6f}")
    print(f"sklearn mean density:   {sk_density.mean():.6e}")
    print(f"flash_sd_kde mean dens: {flash_density.mean():.6e}")
    print(f"max abs diff:           {diff.max():.6e}")
    print(f"mean abs diff:          {diff.mean():.6e}")
    print()

    # 16D: timing comparison for KDE mode (sklearn vs Flash-SD-KDE wrapper),
    # including fit time and fit+score end-to-end time.
    n_train_16d = 8192
    n_query_16d = 1024
    x_train_16d = rng.normal(size=(n_train_16d, 16)).astype(np.float32)
    x_query_16d = rng.normal(size=(n_query_16d, 16)).astype(np.float32)
    total_repeats = 3

    flash_kde_16d = flash_sd_kde.FlashSDKDE(mode="kde", bandwidth="silverman", device="cuda")
    flash_kde_16d.fit(x_train_16d)
    bandwidth_16d = float(flash_kde_16d.bandwidth_)
    sk_16d = KernelDensity(kernel="gaussian", bandwidth=bandwidth_16d)
    sk_16d.fit(x_train_16d)

    # Warmup once to avoid one-time setup effects in timing.
    _ = flash_kde_16d.score_samples(x_query_16d)
    _ = sk_16d.score_samples(x_query_16d)

    def flash_kde_fit_and_score():
        est = flash_sd_kde.FlashSDKDE(mode="kde", bandwidth=bandwidth_16d, device="cuda")
        est.fit(x_train_16d)
        est.score_samples(x_query_16d)

    def sk_kde_fit_and_score():
        est = KernelDensity(kernel="gaussian", bandwidth=bandwidth_16d)
        est.fit(x_train_16d)
        est.score_samples(x_query_16d)

    flash_kde_total_ms = _time_call_ms(
        flash_kde_fit_and_score,
        repeats=total_repeats,
        cuda_sync=True,
    )
    sk_kde_total_ms = _time_call_ms(
        sk_kde_fit_and_score,
        repeats=total_repeats,
        cuda_sync=False,
    )

    flash_kde_log_16d = flash_kde_16d.score_samples(x_query_16d)
    sk_kde_log_16d = sk_16d.score_samples(x_query_16d)
    kde_diff_16d = np.abs(np.exp(sk_kde_log_16d) - np.exp(flash_kde_log_16d))

    print("=== 16D KDE Timing (sklearn vs FlashSDKDE mode='kde') ===")
    print(f"n_train={n_train_16d}, n_query={n_query_16d}, bandwidth={bandwidth_16d:.6f}")
    print(f"sklearn avg fit+score:       {sk_kde_total_ms:.3f} ms")
    print(f"flash_sd_kde avg fit+score:  {flash_kde_total_ms:.3f} ms")
    print(f"total speedup (sklearn/flash): {sk_kde_total_ms / flash_kde_total_ms:.2f}x")
    print(f"max abs density diff:        {kde_diff_16d.max():.6e}")
    print(f"mean abs density diff:       {kde_diff_16d.mean():.6e}")
    print()

    # 16D: Flash SD-KDE mode, including fit timing.
    flash_sd = flash_sd_kde.FlashSDKDE(mode="sd_kde", bandwidth=bandwidth_16d, device="cuda")
    flash_sd.fit(x_train_16d)
    log_dens_16d = flash_sd.score_samples(x_query_16d)

    def flash_sd_fit_and_score():
        est = flash_sd_kde.FlashSDKDE(mode="sd_kde", bandwidth=bandwidth_16d, device="cuda")
        est.fit(x_train_16d)
        est.score_samples(x_query_16d)

    flash_sd_total_ms = _time_call_ms(
        flash_sd_fit_and_score,
        repeats=total_repeats,
        cuda_sync=True,
    )

    print("=== 16D Flash-SD-KDE (mode='sd_kde') ===")
    print(f"n_train={n_train_16d}, n_query={n_query_16d}, bandwidth={bandwidth_16d:.6f}")
    print(f"avg fit+score:          {flash_sd_total_ms:.3f} ms")
    print(f"mean log density:       {log_dens_16d.mean():.6e}")
    print(f"std log density:        {log_dens_16d.std():.6e}")


if __name__ == "__main__":
    main()

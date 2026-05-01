from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets

from benchmarks.exact_kde_baselines import (
    torch_exact_log_kde_nd,
    torch_exact_log_sd_kde_nd,
)
from flash_sd_kde.estimator import FlashSDKDE
from flash_sd_kde.reference import silverman_bandwidth_nd
from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json
from globals import PRECISION_FP32_IEEE


def _load_dataset(root: Path, *, train: bool, fashion: bool) -> torch.Tensor:
    dataset = datasets.FashionMNIST(root=str(root), train=train, download=True) if fashion else datasets.MNIST(
        root=str(root), train=train, download=True
    )
    return dataset.data.float() / 255.0


def _flatten(x: torch.Tensor) -> np.ndarray:
    return x.reshape(x.shape[0], -1).numpy().astype(np.float32, copy=False)


def _compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "mean_score_id": float(np.mean(id_scores)),
        "mean_score_ood": float(np.mean(ood_scores)),
        "score_gap": float(np.mean(id_scores) - np.mean(ood_scores)),
    }


def _time_cuda_ms(
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, Any]:
    for _ in range(max(warmup, 0)):
        _ = fn()
        torch.cuda.synchronize(device)

    values = None
    times_ms: list[float] = []
    for _ in range(max(repeats, 1)):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        values = fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1e3)
    assert values is not None
    arr = np.asarray(times_ms, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), values


def _bucketize_hash_features(
    x: np.ndarray,
    *,
    projections: torch.Tensor,
    n_tables: int,
    n_bits: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    bit_weights = (1 << torch.arange(n_bits, device=device, dtype=torch.int32)).view(1, 1, n_bits)
    out = np.empty((x.shape[0], n_tables), dtype=np.uint16)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        batch = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(device)
        logits = batch @ projections
        signs = (logits >= 0).to(torch.int32).view(-1, n_tables, n_bits)
        buckets = torch.sum(signs * bit_weights, dim=-1).cpu().numpy().astype(np.uint16, copy=False)
        out[start:stop] = buckets
    return out


def _fit_hash_density(
    train: np.ndarray,
    *,
    n_tables: int,
    n_bits: int,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    if n_bits <= 0 or n_bits > 15:
        raise ValueError("--hash-bits must be in [1, 15].")
    if n_tables <= 0:
        raise ValueError("--hash-tables must be positive.")

    _, dim = train.shape
    n_buckets = 1 << n_bits
    bucket_counts = np.zeros((n_tables, n_buckets), dtype=np.int64)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projections = torch.randn(
        dim,
        n_tables * n_bits,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    train_buckets = _bucketize_hash_features(
        train,
        projections=projections,
        n_tables=n_tables,
        n_bits=n_bits,
        device=device,
        batch_size=batch_size,
    )
    for table_idx in range(n_tables):
        bucket_counts[table_idx] += np.bincount(
            train_buckets[:, table_idx].astype(np.int64, copy=False),
            minlength=n_buckets,
        )
    meta = {
        "hash_tables": int(n_tables),
        "hash_bits": int(n_bits),
        "hash_buckets": int(n_buckets),
    }
    model = {
        "projections": projections,
        "bucket_counts": bucket_counts,
        "n_tables": n_tables,
        "n_bits": n_bits,
        "device": device,
        "batch_size": batch_size,
    }
    return model, meta


def _score_hash_density(
    model: dict[str, Any],
    queries: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    projections = model["projections"]
    bucket_counts = model["bucket_counts"]
    n_tables = int(model["n_tables"])
    n_bits = int(model["n_bits"])
    device = str(model["device"])
    batch_size = int(model["batch_size"])

    query_buckets = _bucketize_hash_features(
        queries,
        projections=projections,
        n_tables=n_tables,
        n_bits=n_bits,
        device=device,
        batch_size=batch_size,
    )
    table_axis = np.arange(n_tables, dtype=np.int64)
    counts = np.empty((queries.shape[0], n_tables), dtype=np.float64)
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        buckets = np.asarray(query_buckets[start:stop], dtype=np.int64)
        counts[start:stop] = bucket_counts[table_axis[:, None], buckets.T].T

    density_scores = np.maximum(counts.mean(axis=1), 1.0)
    log_density = np.log(density_scores).astype(np.float32, copy=False)
    meta = {
        "density_min": float(density_scores.min()),
        "density_mean": float(density_scores.mean()),
        "density_max": float(density_scores.max()),
    }
    return log_density, meta


def _render_report(results: dict[str, Any]) -> str:
    methods = results["methods"]
    n_train_list = results["n_train_list"]
    largest_key = str(max(int(n) for n in n_train_list))
    best_quality = max(methods, key=lambda method: results["metrics"][method][largest_key]["roc_auc"])
    fastest = min(methods, key=lambda method: results["runtime_ms"][method][largest_key]["total_ms"])

    lines = [
        "# MNIST/Fashion Approximate Comparison",
        "",
        "## Setup",
        "",
        f"- Feature source: `{results['config']['feature_source']}`",
        f"- Feature dimension: `{results['config']['feature_dim']}`",
        f"- PCA components: `{results['config']['pca_components']}`",
        f"- `n_train` sweep: {', '.join(str(n) for n in n_train_list)}",
        f"- MNIST eval queries: `{results['config']['n_id_eval']}`",
        f"- Fashion-MNIST eval queries: `{results['config']['n_ood_eval']}`",
        f"- Approximate baseline: hash-density with `{results['config']['hash_tables']}` tables and `{results['config']['hash_bits']}` bits",
        "",
        "## Headline",
        "",
        f"- Best ROC AUC at the largest setting: `{best_quality}` with `{results['metrics'][best_quality][largest_key]['roc_auc']:.4f}`.",
        f"- Fastest method at the largest setting: `{fastest}` with `{results['runtime_ms'][fastest][largest_key]['total_ms']:.2f} ms` total runtime.",
        "",
        "## Results Table",
        "",
        "| n_train | Method | Fit (ms) | Score ID (ms) | Score OOD (ms) | Total (ms) | ROC AUC | PR AUC | mean score (ID) | mean score (OOD) | score gap |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n_train in n_train_list:
        key = str(n_train)
        for method in methods:
            runtime = results["runtime_ms"][method][key]
            metrics = results["metrics"][method][key]
            lines.append(
                f"| {n_train} | {method} | {runtime['fit_ms']:.2f} | {runtime['score_id_ms']:.2f} | "
                f"{runtime['score_ood_ms']:.2f} | {runtime['total_ms']:.2f} | {metrics['roc_auc']:.4f} | "
                f"{metrics['pr_auc']:.4f} | {metrics['mean_score_id']:.4f} | {metrics['mean_score_ood']:.4f} | "
                f"{metrics['score_gap']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flash-SD-KDE and a hash-density approximate baseline on MNIST/Fashion features.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-list", default="1000,2000,4000")
    parser.add_argument("--n-id-eval", type=int, default=4000)
    parser.add_argument("--n-ood-eval", type=int, default=4000)
    parser.add_argument("--feature-source", choices=("raw", "pca"), default="raw")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--hash-tables", type=int, default=16)
    parser.add_argument("--hash-bits", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-tag", default="benchmarks/mnist_fashion_raw784_approx_compare")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    run_dir = make_run_dir(tag=args.output_tag)
    data_root = run_dir / "datasets"

    mnist_train = _load_dataset(data_root, train=True, fashion=False)
    mnist_test = _load_dataset(data_root, train=False, fashion=False)
    fashion_test = _load_dataset(data_root, train=False, fashion=True)

    x_train = _flatten(mnist_train)
    x_id = _flatten(mnist_test[: args.n_id_eval])
    x_ood = _flatten(fashion_test[: args.n_ood_eval])

    if args.feature_source == "pca":
        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        x_train = pca.fit_transform(x_train).astype(np.float32, copy=False)
        x_id = pca.transform(x_id).astype(np.float32, copy=False)
        x_ood = pca.transform(x_ood).astype(np.float32, copy=False)
    else:
        pca = None

    n_train_list = [int(tok) for tok in args.n_train_list.split(",") if tok.strip()]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(x_train.shape[0])
    x_train = x_train[perm]

    methods = ["hash_density", "flash_sd_kde", "exact_kde_torch", "exact_sd_torch"]
    results = {
        "config": {
            "device": args.device,
            "seed": args.seed,
            "feature_source": args.feature_source,
            "feature_dim": int(x_train.shape[1]),
            "pca_components": int(args.pca_components) if pca is not None else None,
            "n_train_list": n_train_list,
            "n_id_eval": args.n_id_eval,
            "n_ood_eval": args.n_ood_eval,
            "hash_tables": args.hash_tables,
            "hash_bits": args.hash_bits,
            "batch_size": args.batch_size,
        },
        "meta": get_repo_state(),
        "methods": methods,
        "n_train_list": n_train_list,
        "runtime_ms": {method: {} for method in methods},
        "metrics": {method: {} for method in methods},
    }

    for n_train in n_train_list:
        key = str(n_train)
        train_np = np.asarray(x_train[:n_train], dtype=np.float32, copy=False)

        def _hash_fit_only():
            model, meta = _fit_hash_density(
                train_np,
                n_tables=args.hash_tables,
                n_bits=args.hash_bits,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
            )
            return model, meta

        hash_fit_ms, _, _, (hash_model, hash_meta_fit) = _time_cuda_ms(_hash_fit_only, device=device, warmup=0, repeats=1)
        hash_id_ms, _, _, (hash_id_scores, hash_meta_id) = _time_cuda_ms(
            lambda: _score_hash_density(hash_model, x_id),
            device=device,
            warmup=0,
            repeats=1,
        )
        hash_ood_ms, _, _, (hash_ood_scores, hash_meta_ood) = _time_cuda_ms(
            lambda: _score_hash_density(hash_model, x_ood),
            device=device,
            warmup=0,
            repeats=1,
        )
        results["runtime_ms"]["hash_density"][key] = {
            "fit_ms": float(hash_fit_ms),
            "score_id_ms": float(hash_id_ms),
            "score_ood_ms": float(hash_ood_ms),
            "total_ms": float(hash_fit_ms + hash_id_ms + hash_ood_ms),
            **hash_meta_fit,
            "id_density_mean": float(hash_meta_id["density_mean"]),
            "ood_density_mean": float(hash_meta_ood["density_mean"]),
        }
        results["metrics"]["hash_density"][key] = _compute_metrics(hash_id_scores, hash_ood_scores)

        estimator = FlashSDKDE(
            bandwidth="silverman",
            mode="sd_kde",
            device=args.device,
            precision_mode=PRECISION_FP32_IEEE,
            prefer_specialized_dims=True,
        )

        def _flash_fit():
            estimator.fit(train_np)
            return 0

        flash_fit_ms, _, _, _ = _time_cuda_ms(_flash_fit, device=device, warmup=0, repeats=1)

        estimator.fit(train_np)
        flash_id_ms, _, _, flash_id_scores = _time_cuda_ms(
            lambda: estimator.score_samples(x_id),
            device=device,
            warmup=1,
            repeats=1,
        )
        flash_ood_ms, _, _, flash_ood_scores = _time_cuda_ms(
            lambda: estimator.score_samples(x_ood),
            device=device,
            warmup=0,
            repeats=1,
        )
        flash_id_scores = np.asarray(flash_id_scores, dtype=np.float32).ravel()
        flash_ood_scores = np.asarray(flash_ood_scores, dtype=np.float32).ravel()
        results["runtime_ms"]["flash_sd_kde"][key] = {
            "fit_ms": float(flash_fit_ms),
            "score_id_ms": float(flash_id_ms),
            "score_ood_ms": float(flash_ood_ms),
            "total_ms": float(flash_fit_ms + flash_id_ms + flash_ood_ms),
        }
        results["metrics"]["flash_sd_kde"][key] = _compute_metrics(flash_id_scores, flash_ood_scores)

        bandwidth = float(silverman_bandwidth_nd(train_np))
        train_t = torch.as_tensor(train_np, device=device, dtype=torch.float32).contiguous()
        id_t = torch.as_tensor(x_id, device=device, dtype=torch.float32).contiguous()
        ood_t = torch.as_tensor(x_ood, device=device, dtype=torch.float32).contiguous()

        exact_kde_id_ms, _, _, exact_kde_id_scores = _time_cuda_ms(
            lambda: torch_exact_log_kde_nd(train_t, id_t, bandwidth),
            device=device,
            warmup=1,
            repeats=1,
        )
        exact_kde_ood_ms, _, _, exact_kde_ood_scores = _time_cuda_ms(
            lambda: torch_exact_log_kde_nd(train_t, ood_t, bandwidth),
            device=device,
            warmup=0,
            repeats=1,
        )
        exact_kde_id_scores = exact_kde_id_scores.detach().cpu().numpy().ravel()
        exact_kde_ood_scores = exact_kde_ood_scores.detach().cpu().numpy().ravel()
        results["runtime_ms"]["exact_kde_torch"][key] = {
            "fit_ms": 0.0,
            "score_id_ms": float(exact_kde_id_ms),
            "score_ood_ms": float(exact_kde_ood_ms),
            "total_ms": float(exact_kde_id_ms + exact_kde_ood_ms),
        }
        results["metrics"]["exact_kde_torch"][key] = _compute_metrics(exact_kde_id_scores, exact_kde_ood_scores)

        exact_sd_id_ms, _, _, exact_sd_id_scores = _time_cuda_ms(
            lambda: torch_exact_log_sd_kde_nd(train_t, id_t, bandwidth),
            device=device,
            warmup=1,
            repeats=1,
        )
        exact_sd_ood_ms, _, _, exact_sd_ood_scores = _time_cuda_ms(
            lambda: torch_exact_log_sd_kde_nd(train_t, ood_t, bandwidth),
            device=device,
            warmup=0,
            repeats=1,
        )
        exact_sd_id_scores = exact_sd_id_scores.detach().cpu().numpy().ravel()
        exact_sd_ood_scores = exact_sd_ood_scores.detach().cpu().numpy().ravel()
        results["runtime_ms"]["exact_sd_torch"][key] = {
            "fit_ms": 0.0,
            "score_id_ms": float(exact_sd_id_ms),
            "score_ood_ms": float(exact_sd_ood_ms),
            "total_ms": float(exact_sd_id_ms + exact_sd_ood_ms),
        }
        results["metrics"]["exact_sd_torch"][key] = _compute_metrics(exact_sd_id_scores, exact_sd_ood_scores)

    results_path = run_dir / "results.json"
    report_path = run_dir / "report.md"
    write_json(results_path, results)
    report_path.write_text(_render_report(results), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "results_json": str(results_path), "report_md": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

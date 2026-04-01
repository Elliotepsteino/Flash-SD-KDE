from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from numpy.lib.format import open_memmap
from sklearn.decomposition import IncrementalPCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_sd_kde.estimator import FlashSDKDE
from flash_sd_kde.utils import get_repo_state, make_run_dir, read_json, write_json


def _select_indices(
    *,
    total_rows: int,
    validation_size: int,
    shuffle_seed: int,
    subset: str,
) -> np.ndarray:
    if validation_size <= 0 or validation_size >= total_rows:
        raise ValueError(
            f"--validation-size must be in [1, {total_rows - 1}], got {validation_size}."
        )
    permutation = np.random.default_rng(shuffle_seed).permutation(total_rows)
    if subset == "validation":
        return permutation[:validation_size].astype(np.int64, copy=False)
    if subset == "train_pool":
        return permutation[validation_size:].astype(np.int64, copy=False)
    raise ValueError(f"Unsupported subset {subset!r}.")


def _render_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").strip().upper()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    rendered = "\n\n".join(parts).strip()
    if not rendered:
        raise ValueError("Encountered empty conversation text.")
    return rendered


def _has_valid_supervised_target(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    last = messages[-1]
    prev = messages[-2]
    if not isinstance(last, dict):
        return False
    if not isinstance(prev, dict):
        return False
    if str(last.get("role") or "").strip() != "assistant":
        return False
    if str(prev.get("role") or "").strip() == "assistant":
        return False
    return bool(str(last.get("content") or "").strip())


def _filter_candidate_pool(
    *,
    dataset,
    embeddings: np.ndarray,
    candidate_dataset_indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    valid_mask = np.zeros(candidate_dataset_indices.shape[0], dtype=bool)
    valid_rows = 0
    invalid_rows = 0
    for start in range(0, candidate_dataset_indices.shape[0], batch_size):
        stop = min(start + batch_size, candidate_dataset_indices.shape[0])
        batch_indices = np.asarray(candidate_dataset_indices[start:stop], dtype=np.int64)
        batch = dataset[batch_indices.tolist()]
        messages_batch = batch["messages"]
        for local_idx, messages in enumerate(messages_batch):
            ok = _has_valid_supervised_target(messages)
            valid_mask[start + local_idx] = ok
            valid_rows += int(ok)
            invalid_rows += int(not ok)
    filtered_embeddings = embeddings[valid_mask]
    filtered_dataset_indices = np.asarray(candidate_dataset_indices[valid_mask], dtype=np.int64)
    return (
        filtered_embeddings,
        filtered_dataset_indices,
        {
            "candidate_valid_rows": int(valid_rows),
            "candidate_invalid_rows": int(invalid_rows),
        },
    )


def _select_valid_eval_indices(
    *,
    dataset,
    total_rows: int,
    validation_size: int,
    shuffle_seed: int,
    eval_size: int,
    batch_size: int,
) -> np.ndarray:
    validation_indices = _select_indices(
        total_rows=total_rows,
        validation_size=validation_size,
        shuffle_seed=shuffle_seed,
        subset="validation",
    )
    selected: list[int] = []
    for start in range(0, validation_indices.shape[0], batch_size):
        if len(selected) >= eval_size:
            break
        stop = min(start + batch_size, validation_indices.shape[0])
        batch_indices = np.asarray(validation_indices[start:stop], dtype=np.int64)
        batch = dataset[batch_indices.tolist()]
        for local_idx, messages in enumerate(batch["messages"]):
            if _has_valid_supervised_target(messages):
                selected.append(int(batch_indices[local_idx]))
                if len(selected) >= eval_size:
                    break
    if len(selected) < eval_size:
        raise RuntimeError(
            f"Could only find {len(selected)} valid validation examples, expected {eval_size}."
        )
    return np.asarray(selected, dtype=np.int64)


def _fit_incremental_pca(
    embeddings: np.ndarray,
    *,
    n_components: int,
    batch_size: int,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start in range(0, embeddings.shape[0], batch_size):
        stop = min(start + batch_size, embeddings.shape[0])
        pca.partial_fit(np.asarray(embeddings[start:stop], dtype=np.float32))
    return pca


def _transform_with_pca(
    embeddings: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    batch_size: int,
    output_path: Path,
) -> np.memmap:
    n_samples = embeddings.shape[0]
    n_components = components.shape[0]
    transformed = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_components),
    )
    components_t = components.T.astype(np.float32, copy=False)
    mean = mean.astype(np.float32, copy=False)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = np.asarray(embeddings[start:stop], dtype=np.float32)
        transformed[start:stop] = (batch - mean) @ components_t
    transformed.flush()
    return transformed


def _compute_hash_log_weights(
    features: np.ndarray,
    *,
    n_tables: int,
    n_bits: int,
    seed: int,
    batch_size: int,
    device: str,
    buckets_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if n_bits <= 0 or n_bits > 15:
        raise ValueError("--hash-bits must be in [1, 15].")
    if n_tables <= 0:
        raise ValueError("--hash-tables must be positive.")

    n_samples, n_features = features.shape
    n_buckets = 1 << n_bits
    bucket_ids = open_memmap(
        buckets_path,
        mode="w+",
        dtype=np.uint16,
        shape=(n_samples, n_tables),
    )
    bucket_counts = np.zeros((n_tables, n_buckets), dtype=np.int64)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projections = torch.randn(
        n_features,
        n_tables * n_bits,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    bit_weights = (1 << torch.arange(n_bits, device=device, dtype=torch.int32)).view(1, 1, n_bits)

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
        logits = batch @ projections
        signs = (logits >= 0).to(torch.int32).view(-1, n_tables, n_bits)
        buckets = torch.sum(signs * bit_weights, dim=-1).cpu().numpy().astype(np.uint16, copy=False)
        bucket_ids[start:stop] = buckets
        for table_idx in range(n_tables):
            bucket_counts[table_idx] += np.bincount(
                buckets[:, table_idx].astype(np.int64, copy=False),
                minlength=n_buckets,
            )

    bucket_ids.flush()

    density_scores = np.empty(n_samples, dtype=np.float64)
    table_axis = np.arange(n_tables, dtype=np.int64)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        buckets = np.asarray(bucket_ids[start:stop], dtype=np.int64)
        counts = bucket_counts[table_axis[:, None], buckets.T].T
        density_scores[start:stop] = np.maximum(counts.mean(axis=1) - 1.0, 1.0)

    log_weights = -np.log(density_scores)
    metadata = {
        "hash_tables": n_tables,
        "hash_bits": n_bits,
        "hash_buckets": n_buckets,
        "density_min": float(density_scores.min()),
        "density_mean": float(density_scores.mean()),
        "density_max": float(density_scores.max()),
    }
    return log_weights, metadata


def _compute_flash_log_weights(
    features: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    estimator = FlashSDKDE(
        bandwidth="silverman",
        mode=mode,
        device=device,
        prefer_specialized_dims=True,
    )
    estimator.fit(np.asarray(features, dtype=np.float32))

    log_weights = np.empty(features.shape[0], dtype=np.float64)
    non_finite = 0
    log_density_min = math.inf
    log_density_max = -math.inf
    log_density_sum = 0.0
    log_density_count = 0
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        log_density = estimator.score_samples(np.asarray(features[start:stop], dtype=np.float32))
        bad_mask = ~np.isfinite(log_density)
        if np.any(bad_mask):
            non_finite += int(bad_mask.sum())
            log_density[bad_mask] = math.log(1e-12)
        log_weights[start:stop] = -log_density.astype(np.float64, copy=False)
        log_density_min = min(log_density_min, float(log_density.min()))
        log_density_max = max(log_density_max, float(log_density.max()))
        log_density_sum += float(log_density.sum())
        log_density_count += int(log_density.shape[0])
    metadata = {
        "flash_mode": mode,
        "non_finite_log_density": non_finite,
        "log_density_min": log_density_min,
        "log_density_mean": log_density_sum / max(log_density_count, 1),
        "log_density_max": log_density_max,
    }
    return log_weights, metadata


def _sample_without_replacement(
    log_weights: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if sample_size <= 0 or sample_size > log_weights.shape[0]:
        raise ValueError(
            f"Requested sample_size={sample_size}, but candidate pool has {log_weights.shape[0]} rows."
        )
    rng = np.random.default_rng(seed)
    gumbels = -np.log(-np.log(np.clip(rng.random(log_weights.shape[0]), 1e-12, 1 - 1e-12)))
    priorities = log_weights + gumbels
    selected = np.argpartition(priorities, -sample_size)[-sample_size:]
    selected = selected[np.argsort(priorities[selected])[::-1]]
    return selected.astype(np.int64, copy=False)


def _write_messages_jsonl(
    *,
    dataset,
    dataset_indices: np.ndarray,
    path: Path,
    batch_size: int,
) -> dict[str, Any]:
    source_counter: Counter[str] = Counter()
    token_lengths: list[int] = []
    with path.open("w", encoding="utf-8") as handle:
        for start in range(0, dataset_indices.shape[0], batch_size):
            stop = min(start + batch_size, dataset_indices.shape[0])
            batch_indices = dataset_indices[start:stop]
            batch = dataset[batch_indices.tolist()]
            for local_offset, dataset_index in enumerate(batch_indices):
                messages = batch["messages"][local_offset]
                rendered = _render_messages(messages)
                source = str(batch["source"][local_offset])
                source_counter[source] += 1
                token_lengths.append(len(rendered.split()))
                payload = {
                    "selection_rank": start + local_offset,
                    "dataset_index": int(dataset_index),
                    "id": batch["id"][local_offset],
                    "source": source,
                    "messages": messages,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return {
        "num_rows": int(dataset_indices.shape[0]),
        "mean_rendered_words": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "top_sources": source_counter.most_common(10),
    }


def _render_report(results: dict[str, Any]) -> str:
    feature_source = results["config"]["feature_source"]
    feature_dim = results["summary"]["feature_dim"]
    pca_components = results["summary"].get("pca_components")
    feature_label = f"{feature_source} ({feature_dim}D)"
    if feature_source == "pca" and pca_components is not None:
        feature_label = f"PCA-{pca_components} ({feature_dim}D)"
    lines = [
        "# Tulu Real-Application Sampling",
        "",
        "## Goal",
        "",
        "Create train subsets for Qwen3 SFT from the Tulu-3 mixture using three sampling rules:",
        "`random`, `hash_density`, and `flash_sd_kde`.",
        "",
        "## Candidate Pool",
        "",
        f"- Dataset: `{results['config']['dataset_name']}`",
        f"- Embedding run: `{results['config']['embedding_run_dir']}`",
        f"- Candidate examples: `{results['summary']['candidate_examples']}`",
        f"- Invalid candidate rows removed: `{results['summary']['candidate_invalid_rows']}`",
        f"- Eval examples exported: `{results['summary']['eval_examples']}`",
        f"- Feature source: `{feature_label}`",
        f"- Flash mode: `{results['config']['flash_mode']}`",
        "",
        "## Score Summary",
        "",
        f"- Hash density mean bucket count: `{results['hash_density']['density_mean']:.3f}`",
        f"- Flash mean log density: `{results['flash_sd_kde']['log_density_mean']:.6f}`",
        f"- Flash non-finite log densities replaced: `{results['flash_sd_kde']['non_finite_log_density']}`",
        "",
        "## Timings",
        "",
        f"- Feature preparation: `{results['timings']['feature_prepare_seconds']:.2f}` s",
        f"- Hash density scoring: `{results['timings']['hash_seconds']:.2f}` s",
        f"- Flash-SD-KDE scoring: `{results['timings']['flash_seconds']:.2f}` s",
        f"- Total pipeline runtime: `{results['timings']['total_seconds']:.2f}` s",
        "",
        "## Exported Train Sets",
        "",
        "| method | sample_size | path | mean rendered words |",
        "| --- | ---: | --- | ---: |",
    ]
    for export in results["exports"]:
        lines.append(
            f"| {export['method']} | {export['sample_size']} | `{export['path']}` | "
            f"{export['summary']['mean_rendered_words']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Eval Set",
            "",
            f"- Path: `{results['eval_export']['path']}`",
            f"- Mean rendered words: `{results['eval_export']['summary']['mean_rendered_words']:.1f}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Tulu subset exports for the real-application benchmark.")
    parser.add_argument("--dataset-name", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--split", default="train")
    parser.add_argument("--validation-size", type=int, default=50_000)
    parser.add_argument("--shuffle-seed", type=int, default=20260330)
    parser.add_argument("--embedding-run-dir", required=True)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--feature-source", choices=("pca", "raw"), default="pca")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--flash-batch-size", type=int, default=4096)
    parser.add_argument("--hash-tables", type=int, default=16)
    parser.add_argument("--hash-bits", type=int, default=8)
    parser.add_argument("--flash-mode", choices=("sd_kde", "kde"), default="sd_kde")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-sizes", default="100")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--output-tag", default="benchmarks/tulu_real_application_sampling")
    args = parser.parse_args()

    sample_sizes = [int(item) for item in args.sample_sizes.split(",") if item.strip()]
    if not sample_sizes:
        raise ValueError("--sample-sizes must contain at least one integer.")

    overall_start = time.perf_counter()
    run_dir = make_run_dir(tag=args.output_tag)
    embedding_run_dir = Path(args.embedding_run_dir).expanduser().resolve()
    if not embedding_run_dir.exists():
        raise FileNotFoundError(f"Embedding run directory not found: {embedding_run_dir}")

    embeddings = np.load(embedding_run_dir / "embeddings.npy", mmap_mode="r")
    candidate_dataset_indices = np.load(embedding_run_dir / "dataset_indices.npy", mmap_mode="r")
    if embeddings.shape[0] != candidate_dataset_indices.shape[0]:
        raise RuntimeError("embeddings.npy and dataset_indices.npy disagree on number of rows.")
    if args.candidate_limit > 0:
        candidate_limit = min(args.candidate_limit, embeddings.shape[0])
        embeddings = embeddings[:candidate_limit]
        candidate_dataset_indices = candidate_dataset_indices[:candidate_limit]
    metadata_path = embedding_run_dir / "results.json"
    embedding_meta = read_json(metadata_path) if metadata_path.exists() else None
    dataset = load_dataset(args.dataset_name, split=args.split)

    embeddings, candidate_dataset_indices, candidate_filter_meta = _filter_candidate_pool(
        dataset=dataset,
        embeddings=embeddings,
        candidate_dataset_indices=candidate_dataset_indices,
        batch_size=2048,
    )
    if max(sample_sizes) > embeddings.shape[0]:
        raise ValueError(
            f"Requested sample size {max(sample_sizes)}, but valid embedding pool only has {embeddings.shape[0]} rows."
        )

    feature_prepare_start = time.perf_counter()
    pca = None
    if args.feature_source == "pca":
        pca = _fit_incremental_pca(
            embeddings,
            n_components=args.pca_components,
            batch_size=args.feature_batch_size,
        )
        feature_prepare_mid = time.perf_counter()
        pca_features_path = run_dir / f"pca_{args.pca_components}d.npy"
        features = _transform_with_pca(
            embeddings,
            mean=np.asarray(pca.mean_, dtype=np.float32),
            components=np.asarray(pca.components_, dtype=np.float32),
            batch_size=args.feature_batch_size,
            output_path=pca_features_path,
        )
        pca_fit_seconds = feature_prepare_mid - feature_prepare_start
        pca_transform_seconds = time.perf_counter() - feature_prepare_mid
        np.savez(
            run_dir / "pca_model.npz",
            mean=np.asarray(pca.mean_, dtype=np.float32),
            components=np.asarray(pca.components_, dtype=np.float32),
            explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        )
    else:
        features = np.asarray(embeddings, dtype=np.float32)
        pca_fit_seconds = 0.0
        pca_transform_seconds = 0.0
    feature_prepare_seconds = time.perf_counter() - feature_prepare_start

    hash_start = time.perf_counter()
    hash_log_weights, hash_meta = _compute_hash_log_weights(
        features,
        n_tables=args.hash_tables,
        n_bits=args.hash_bits,
        seed=args.shuffle_seed,
        batch_size=args.feature_batch_size,
        device=args.device,
        buckets_path=run_dir / "hash_bucket_ids.npy",
    )
    hash_seconds = time.perf_counter() - hash_start
    np.save(run_dir / "hash_log_weights.npy", hash_log_weights.astype(np.float32))

    flash_start = time.perf_counter()
    flash_log_weights, flash_meta = _compute_flash_log_weights(
        features,
        mode=args.flash_mode,
        batch_size=args.flash_batch_size,
        device=args.device,
    )
    flash_seconds = time.perf_counter() - flash_start
    np.save(run_dir / "flash_log_weights.npy", flash_log_weights.astype(np.float32))

    eval_indices = _select_valid_eval_indices(
        dataset=dataset,
        total_rows=len(dataset),
        validation_size=args.validation_size,
        shuffle_seed=args.shuffle_seed,
        eval_size=args.eval_size,
        batch_size=2048,
    )
    eval_path = run_dir / "eval_messages.jsonl"
    eval_summary = _write_messages_jsonl(
        dataset=dataset,
        dataset_indices=eval_indices,
        path=eval_path,
        batch_size=2048,
    )

    exports: list[dict[str, Any]] = []
    method_to_log_weights = {
        "random": np.zeros(embeddings.shape[0], dtype=np.float64),
        "hash_density": hash_log_weights,
        "flash_sd_kde": flash_log_weights,
    }
    for sample_size in sample_sizes:
        for method, log_weights in method_to_log_weights.items():
            local_indices = _sample_without_replacement(
                log_weights,
                sample_size=sample_size,
                seed=args.shuffle_seed + sample_size + {"random": 11, "hash_density": 23, "flash_sd_kde": 37}[method],
            )
            dataset_indices = np.asarray(candidate_dataset_indices[local_indices], dtype=np.int64)
            np.save(run_dir / f"{method}_local_indices_n{sample_size}.npy", local_indices)
            np.save(run_dir / f"{method}_dataset_indices_n{sample_size}.npy", dataset_indices)
            export_path = run_dir / f"{method}_train_n{sample_size}.jsonl"
            export_summary = _write_messages_jsonl(
                dataset=dataset,
                dataset_indices=dataset_indices,
                path=export_path,
                batch_size=2048,
            )
            exports.append(
                {
                    "method": method,
                    "sample_size": sample_size,
                    "path": export_path.name,
                    "summary": export_summary,
                }
            )

    total_seconds = time.perf_counter() - overall_start
    results = {
        "config": {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "validation_size": args.validation_size,
            "shuffle_seed": args.shuffle_seed,
            "embedding_run_dir": str(embedding_run_dir),
            "candidate_limit": args.candidate_limit,
            "feature_source": args.feature_source,
            "pca_components": args.pca_components,
            "feature_batch_size": args.feature_batch_size,
            "flash_batch_size": args.flash_batch_size,
            "hash_tables": args.hash_tables,
            "hash_bits": args.hash_bits,
            "flash_mode": args.flash_mode,
            "device": args.device,
            "sample_sizes": sample_sizes,
            "eval_size": args.eval_size,
            "output_tag": args.output_tag,
        },
        "meta": get_repo_state(),
        "embedding_meta": embedding_meta,
        "summary": {
            "candidate_examples": int(embeddings.shape[0]),
            "candidate_embedding_dim": int(embeddings.shape[1]),
            **candidate_filter_meta,
            "feature_dim": int(features.shape[1]),
            "eval_examples": int(eval_indices.shape[0]),
        },
        "timings": {
            "feature_prepare_seconds": feature_prepare_seconds,
            "pca_fit_seconds": pca_fit_seconds,
            "pca_transform_seconds": pca_transform_seconds,
            "hash_seconds": hash_seconds,
            "flash_seconds": flash_seconds,
            "total_seconds": total_seconds,
        },
        "hash_density": hash_meta,
        "flash_sd_kde": flash_meta,
        "exports": exports,
        "eval_export": {
            "path": eval_path.name,
            "summary": eval_summary,
        },
    }
    if pca is not None:
        results["summary"]["pca_components"] = args.pca_components
        results["summary"]["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))

    results_path = run_dir / "results.json"
    report_path = run_dir / "report.md"
    write_json(results_path, results)
    report_path.write_text(_render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

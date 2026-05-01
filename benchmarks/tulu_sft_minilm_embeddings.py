from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _render_messages(messages: list[dict[str, Any]]) -> str:
    rendered_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").strip().upper()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        rendered_parts.append(f"{role}: {content}")
    rendered_text = "\n\n".join(rendered_parts).strip()
    if not rendered_text:
        raise ValueError("Encountered a Tulu sample with no non-empty message content.")
    return rendered_text


def _select_indices(
    *,
    total_rows: int,
    validation_size: int,
    shuffle_seed: int,
    subset: str,
    n_samples: int,
) -> tuple[np.ndarray, int]:
    if total_rows <= 0:
        raise ValueError("Dataset is empty.")
    if validation_size <= 0 or validation_size >= total_rows:
        raise ValueError(
            f"--validation-size must be in [1, {total_rows - 1}], got {validation_size}."
        )

    permutation = np.random.default_rng(shuffle_seed).permutation(total_rows)
    if subset == "validation":
        available_indices = permutation[:validation_size]
    elif subset == "train_pool":
        available_indices = permutation[validation_size:]
    elif subset == "full":
        available_indices = permutation
    else:
        raise ValueError(f"Unsupported subset {subset!r}.")

    available = int(available_indices.shape[0])
    if n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if n_samples > available:
        raise ValueError(f"Requested {n_samples} samples, but subset {subset!r} only has {available}.")
    return available_indices[:n_samples].astype(np.int64, copy=False), available


def _device_name(device: str) -> str | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def _truncate_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[str], list[int]]:
    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    truncated_texts: list[str] = []
    used_token_counts: list[int] = []
    for token_ids in tokenized:
        truncated_text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if not truncated_text:
            raise RuntimeError("Tokenizer truncation produced an empty rendered text.")
        truncated_texts.append(truncated_text)
        used_token_counts.append(len(token_ids))
    return truncated_texts, used_token_counts


def _render_report(results: dict[str, Any]) -> str:
    config = results["config"]
    summary = results["summary"]
    timings = results["timings"]
    artifacts = results["artifacts"]
    preview = results["sample_preview"]
    lines = [
        "# Tulu-3 SFT Embeddings",
        "",
        "## What This Run Does",
        "",
        "Embeds a deterministic subset of `allenai/tulu-3-sft-mixture` for the",
        "Flash-SD-KDE real-application pipeline. Each record is rendered as one",
        "conversation string with role prefixes, truncated to the embedding-model token limit,",
        f"and embedded with `{config['embedding_model']}`.",
        "",
        "## Split",
        "",
        f"- Total rows in dataset: `{summary['dataset_total_rows']}`",
        f"- Fixed validation size: `{config['validation_size']}`",
        f"- Train-pool size after split: `{summary['train_pool_size']}`",
        f"- Subset used for this run: `{config['subset']}`",
        f"- Samples embedded in this run: `{summary['n_samples']}`",
        f"- Shuffle seed: `{config['shuffle_seed']}`",
        "",
        "## Summary",
        "",
        f"- Device used: `{summary['device']}`",
        f"- Device name: `{summary['device_name'] or 'n/a'}`",
        f"- Embedding shape: `{summary['embedding_shape'][0]} x {summary['embedding_shape'][1]}`",
        f"- Mean used token count: `{summary['mean_used_tokens']:.2f}`",
        f"- Mean rendered char count: `{summary['mean_char_count']:.2f}`",
        f"- Mean embedding L2 norm: `{summary['mean_embedding_norm']:.4f}`",
        f"- Throughput: `{summary['samples_per_second']:.2f}` samples/s",
        "",
        "## Timing",
        "",
        f"- Dataset load: `{timings['dataset_load_seconds']:.2f}` s",
        f"- Index selection: `{timings['selection_seconds']:.2f}` s",
        f"- Model load: `{timings['model_load_seconds']:.2f}` s",
        f"- Render + truncate: `{timings['render_and_truncate_seconds']:.2f}` s",
        f"- Embedding compute: `{timings['embedding_seconds']:.2f}` s",
        f"- Total wall time: `{timings['total_seconds']:.2f}` s",
        "",
        "## Artifacts",
        "",
        f"- Embeddings: `{artifacts['embeddings_npy']}`",
        f"- Dataset indices: `{artifacts['dataset_indices_npy']}`",
        f"- Records: `{artifacts['records_jsonl']}`",
        f"- Run metadata: `{artifacts['results_json']}`",
        "",
        "## Sample Preview",
        "",
        "| row_in_output | dataset_index | id | source | used_token_count |",
        "| ---: | ---: | --- | --- | ---: |",
    ]
    for row in preview:
        lines.append(
            f"| {row['row_in_output']} | {row['dataset_index']} | {row['id']} | "
            f"{row['source']} | {row['used_token_count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create embeddings for a Tulu-3 SFT subset.")
    parser.add_argument("--dataset-name", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", choices=("train_pool", "validation", "full"), default="train_pool")
    parser.add_argument("--validation-size", type=int, default=50_000)
    parser.add_argument("--shuffle-seed", type=int, default=20260330)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-tag", default="benchmarks/tulu_sft_minilm_embeddings")
    args = parser.parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    overall_start = time.perf_counter()
    device = _resolve_device(args.device)
    run_dir = make_run_dir(tag=args.output_tag)

    dataset_load_start = time.perf_counter()
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset_load_seconds = time.perf_counter() - dataset_load_start

    total_rows = len(dataset)
    selection_start = time.perf_counter()
    selected_indices, subset_available = _select_indices(
        total_rows=total_rows,
        validation_size=args.validation_size,
        shuffle_seed=args.shuffle_seed,
        subset=args.subset,
        n_samples=args.n_samples,
    )
    selection_seconds = time.perf_counter() - selection_start

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    model_load_start = time.perf_counter()
    model = SentenceTransformer(args.embedding_model, device=device)
    model.max_seq_length = args.max_tokens
    model_load_seconds = time.perf_counter() - model_load_start

    embeddings_path = run_dir / "embeddings.npy"
    dataset_indices_path = run_dir / "dataset_indices.npy"
    records_path = run_dir / "records.jsonl"
    results_path = run_dir / "results.json"
    report_path = run_dir / "report.md"

    np.save(dataset_indices_path, selected_indices)

    render_and_truncate_seconds = 0.0
    embedding_seconds = 0.0
    embeddings_memmap: np.memmap | None = None
    sample_preview: list[dict[str, Any]] = []

    token_sum = 0
    token_min: int | None = None
    token_max = 0
    char_sum = 0
    char_min: int | None = None
    char_max = 0
    norm_sum = 0.0
    norm_sq_sum = 0.0
    processed = 0

    with records_path.open("w", encoding="utf-8") as handle:
        for start in range(0, selected_indices.shape[0], args.batch_size):
            stop = min(start + args.batch_size, selected_indices.shape[0])
            batch_indices = selected_indices[start:stop]
            batch = dataset[batch_indices.tolist()]

            render_start = time.perf_counter()
            rendered_texts = [_render_messages(messages) for messages in batch["messages"]]
            truncated_texts, used_token_counts = _truncate_texts(
                rendered_texts,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
            )
            render_and_truncate_seconds += time.perf_counter() - render_start

            embedding_start = time.perf_counter()
            batch_embeddings = model.encode(
                truncated_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32, copy=False)
            embedding_seconds += time.perf_counter() - embedding_start

            if batch_embeddings.ndim != 2:
                raise RuntimeError(f"Expected rank-2 embeddings, got shape {batch_embeddings.shape}.")
            if embeddings_memmap is None:
                embeddings_memmap = open_memmap(
                    embeddings_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(selected_indices.shape[0], batch_embeddings.shape[1]),
                )
            embeddings_memmap[start:stop] = batch_embeddings

            norms = np.linalg.norm(batch_embeddings, axis=1)
            for local_offset, dataset_index in enumerate(batch_indices):
                token_count = int(used_token_counts[local_offset])
                char_count = len(truncated_texts[local_offset])
                token_sum += token_count
                char_sum += char_count
                token_min = token_count if token_min is None else min(token_min, token_count)
                token_max = max(token_max, token_count)
                char_min = char_count if char_min is None else min(char_min, char_count)
                char_max = max(char_max, char_count)

                norm_value = float(norms[local_offset])
                norm_sum += norm_value
                norm_sq_sum += norm_value * norm_value

                record = {
                    "row_in_output": start + local_offset,
                    "dataset_index": int(dataset_index),
                    "id": batch["id"][local_offset],
                    "source": batch["source"][local_offset],
                    "message_count": len(batch["messages"][local_offset]),
                    "used_token_count": token_count,
                    "char_count": char_count,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

                if len(sample_preview) < 5:
                    sample_preview.append(record)
            processed = stop

    if embeddings_memmap is None:
        raise RuntimeError("No embeddings were written.")
    embeddings_memmap.flush()
    del embeddings_memmap

    total_seconds = time.perf_counter() - overall_start
    mean_norm = norm_sum / processed
    variance_norm = max((norm_sq_sum / processed) - (mean_norm * mean_norm), 0.0)

    results = {
        "config": {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "subset": args.subset,
            "validation_size": args.validation_size,
            "shuffle_seed": args.shuffle_seed,
            "embedding_model": args.embedding_model,
            "requested_device": args.device,
            "device": device,
            "n_samples": args.n_samples,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "output_tag": args.output_tag,
        },
        "meta": get_repo_state(),
        "summary": {
            "dataset_total_rows": total_rows,
            "train_pool_size": total_rows - args.validation_size,
            "subset_available_rows": subset_available,
            "n_samples": processed,
            "embedding_shape": [processed, int(batch_embeddings.shape[1])],
            "device": device,
            "device_name": _device_name(device),
            "mean_used_tokens": token_sum / processed,
            "min_used_tokens": token_min,
            "max_used_tokens": token_max,
            "mean_char_count": char_sum / processed,
            "min_char_count": char_min,
            "max_char_count": char_max,
            "mean_embedding_norm": mean_norm,
            "std_embedding_norm": variance_norm**0.5,
            "samples_per_second": processed / max(embedding_seconds, 1e-12),
            "selection_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        },
        "timings": {
            "dataset_load_seconds": dataset_load_seconds,
            "selection_seconds": selection_seconds,
            "model_load_seconds": model_load_seconds,
            "render_and_truncate_seconds": render_and_truncate_seconds,
            "embedding_seconds": embedding_seconds,
            "total_seconds": total_seconds,
        },
        "artifacts": {
            "embeddings_npy": embeddings_path.name,
            "dataset_indices_npy": dataset_indices_path.name,
            "records_jsonl": records_path.name,
            "results_json": results_path.name,
            "report_md": report_path.name,
        },
        "sample_preview": sample_preview,
    }

    write_json(results_path, results)
    report_path.write_text(_render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

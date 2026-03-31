from __future__ import annotations

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _truncate_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    stream_index: int,
) -> dict[str, Any] | None:
    text = str(record.get("text") or "").strip()
    if not text:
        return None
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if len(token_ids) < min_tokens:
        return None
    truncated_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    if not truncated_text:
        return None
    return {
        "stream_index": stream_index,
        "url": record.get("url"),
        "timestamp": record.get("timestamp"),
        "used_token_count": len(token_ids),
        "truncated_text": truncated_text,
    }


def _collect_records(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    tokenizer: Any,
    n_samples: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )
    records: list[dict[str, Any]] = []
    scanned = 0
    for stream_index, row in enumerate(dataset):
        scanned = stream_index + 1
        maybe_record = _truncate_record(
            row,
            tokenizer=tokenizer,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            stream_index=stream_index,
        )
        if maybe_record is None:
            continue
        records.append(maybe_record)
        if len(records) >= n_samples:
            break
    if len(records) < n_samples:
        raise RuntimeError(
            f"Only collected {len(records)} usable samples after scanning {scanned} rows; "
            f"needed {n_samples}."
        )
    return records, scanned


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _render_report(results: dict[str, Any]) -> str:
    config = results["config"]
    summary = results["summary"]
    artifacts = results["artifacts"]
    preview = results["sample_preview"]
    lines = [
        "# C4 MiniLM Embedding Sanity Check",
        "",
        "## What This Run Does",
        "",
        "This is the first stage of a larger real-world application benchmark for Flash-SD-KDE.",
        "It does not run KDE sampling or language-model training yet. Instead, it verifies the",
        "data-ingestion and embedding pipeline on C4 and saves the outputs needed for later",
        "density-based subset selection.",
        "",
        "Concretely, this run:",
        f"- streams `{config['dataset_name']}` / `{config['dataset_config']}` split `{config['split']}`",
        f"- keeps `{summary['n_samples']}` samples with `{config['min_tokens']}..{config['max_tokens']}` used tokens",
        f"- truncates each sample to at most `{config['max_tokens']}` tokens",
        f"- embeds each sample with `{config['embedding_model']}`",
        f"- writes the selected records and the `384`-D embedding matrix to disk",
        "",
        "## Why This Matters",
        "",
        "The downstream experiment depends on reusing exactly the same sampled texts and embeddings",
        "across random, approximate-KDE, and Flash-SD-KDE subset construction. Persisting the",
        "embeddings now avoids recomputing them and makes later sampling comparisons deterministic",
        "at the data-representation stage.",
        "",
        "## Summary",
        "",
        f"- Device used: `{summary['device']}`",
        f"- Rows scanned from C4 stream: `{summary['rows_scanned']}`",
        f"- Accepted samples: `{summary['n_samples']}`",
        f"- Embedding shape: `{summary['embedding_shape'][0]} x {summary['embedding_shape'][1]}`",
        f"- Mean used token count: `{summary['mean_used_tokens']:.2f}`",
        f"- Mean embedding L2 norm: `{summary['mean_embedding_norm']:.4f}`",
        "",
        "## Artifacts",
        "",
        f"- Embeddings: `{artifacts['embeddings_npy']}`",
        f"- Selected records: `{artifacts['records_jsonl']}`",
        f"- Run metadata: `{artifacts['results_json']}`",
        "",
        "## Sample Preview",
        "",
        "| stream_index | used_token_count | url |",
        "| ---: | ---: | --- |",
    ]
    for row in preview:
        lines.append(
            f"| {row['stream_index']} | {row['used_token_count']} | {row['url'] or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check C4 embedding generation with MiniLM.")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-tag", default="benchmarks/c4_minilm_embedding_sanity")
    args = parser.parse_args()

    if args.min_tokens <= 0:
        raise ValueError("--min-tokens must be positive.")
    if args.max_tokens < args.min_tokens:
        raise ValueError("--max-tokens must be >= --min-tokens.")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")

    device = _resolve_device(args.device)
    run_dir = make_run_dir(tag=args.output_tag)

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    records, rows_scanned = _collect_records(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    model = SentenceTransformer(args.embedding_model, device=device)
    model.max_seq_length = args.max_tokens
    texts = [record["truncated_text"] for record in records]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected a rank-2 embedding matrix, got shape {embeddings.shape}.")

    embeddings_path = run_dir / "embeddings.npy"
    records_path = run_dir / "records.jsonl"
    results_path = run_dir / "results.json"
    report_path = run_dir / "report.md"

    np.save(embeddings_path, embeddings)
    _write_jsonl(records_path, records)

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    results = {
        "config": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "embedding_model": args.embedding_model,
            "requested_device": args.device,
            "device": device,
            "n_samples": args.n_samples,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "output_tag": args.output_tag,
        },
        "meta": get_repo_state(),
        "summary": {
            "rows_scanned": rows_scanned,
            "n_samples": len(records),
            "embedding_shape": list(embeddings.shape),
            "device": device,
            "mean_used_tokens": float(np.mean([record["used_token_count"] for record in records])),
            "min_used_tokens": int(min(record["used_token_count"] for record in records)),
            "max_used_tokens": int(max(record["used_token_count"] for record in records)),
            "mean_embedding_norm": float(np.mean(embedding_norms)),
            "std_embedding_norm": float(np.std(embedding_norms)),
        },
        "artifacts": {
            "embeddings_npy": embeddings_path.name,
            "records_jsonl": records_path.name,
            "results_json": results_path.name,
            "report_md": report_path.name,
        },
        "sample_preview": [
            {
                "stream_index": record["stream_index"],
                "used_token_count": record["used_token_count"],
                "url": record["url"],
            }
            for record in records[:5]
        ],
    }
    write_json(results_path, results)
    report_path.write_text(_render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

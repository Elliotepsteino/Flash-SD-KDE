from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


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


def _messages_to_texts(tokenizer, messages: list[dict[str, Any]]) -> dict[str, str]:
    if len(messages) < 2:
        raise ValueError("Each example must contain at least one prompt turn and one assistant turn.")
    if str(messages[-1].get("role") or "").strip() != "assistant":
        raise ValueError("Final message must be an assistant turn.")
    if not str(messages[-1].get("content") or "").strip():
        raise ValueError("Final assistant message is empty.")
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"prompt_text": prompt_text, "full_text": full_text}


def _mask_to_last_assistant(tokenizer, prompt_text: str, full_text: str, max_len: int) -> dict[str, list[int]]:
    im_end_tok = "<|im_end|>"
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_tok)

    prompt_nt = tokenizer(prompt_text, add_special_tokens=False)
    full_nt = tokenizer(full_text, add_special_tokens=False)
    prompt_ids = prompt_nt["input_ids"]
    full_ids = full_nt["input_ids"]

    if not (len(prompt_ids) <= len(full_ids) and prompt_ids == full_ids[: len(prompt_ids)]):
        raise RuntimeError("Prompt is not a prefix of full serialization.")

    needs_append = False
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        if not full_ids or full_ids[-1] != im_end_id:
            needs_append = True
    else:
        needs_append = not full_text.rstrip().endswith(im_end_tok)

    if needs_append:
        full_text = full_text + im_end_tok
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    drop = max(0, len(full_ids) - max_len)
    cropped_ids = full_ids[drop:]
    prompt_len = max(0, len(prompt_ids) - drop)

    attention_mask = [1] * len(cropped_ids)
    labels = [-100] * len(cropped_ids)

    end_pos = len(cropped_ids)
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        try:
            end_pos = cropped_ids.index(im_end_id, prompt_len) + 1
        except ValueError:
            end_pos = len(cropped_ids)

    for idx in range(prompt_len, min(end_pos, len(cropped_ids))):
        labels[idx] = cropped_ids[idx]

    return {
        "input_ids": cropped_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalLMPaddingCollator:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        def _pad(field: str, pad_value: int) -> torch.Tensor:
            return torch.tensor(
                [
                    feature[field] + [pad_value] * (max_len - len(feature[field]))
                    for feature in features
                ],
                dtype=torch.long,
            )

        return {
            "input_ids": _pad("input_ids", self.pad_token_id),
            "attention_mask": _pad("attention_mask", 0),
            "labels": _pad("labels", self.label_pad_token_id),
        }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Tulu Subset Training",
        "",
        "## Configuration",
        "",
        f"- Train file: `{results['config']['train_jsonl']}`",
        f"- Eval file: `{results['config']['eval_jsonl']}`",
        f"- Base model: `{results['config']['model_name']}`",
        f"- Output dir: `{results['config']['output_dir']}`",
        f"- Num epochs: `{results['config']['num_epochs']}`",
        f"- Batch size: `{results['config']['batch_size']}`",
        f"- Eval batch size: `{results['config']['eval_batch_size']}`",
        f"- Grad accumulation: `{results['config']['grad_accum']}`",
        f"- Valid train rows: `{results['dataset']['train_rows']}` / `{results['dataset']['train_rows_before_filter']}`",
        f"- Valid eval rows: `{results['dataset']['eval_rows']}` / `{results['dataset']['eval_rows_before_filter']}`",
        "",
        "## Results",
        "",
        f"- Train runtime: `{results['timings']['train_seconds']:.2f}` s",
        f"- Eval runtime: `{results['timings']['eval_seconds']:.2f}` s",
        f"- Final eval loss: `{results['metrics'].get('eval_loss', float('nan')):.6f}`",
        f"- Final eval perplexity: `{results['metrics'].get('eval_perplexity', float('nan')):.4f}`",
        f"- Merged model dir: `{results['artifacts'].get('merged_model_dir', 'n/a')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 LoRA on a Tulu subset and report held-out perplexity.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="/home/epsteine/post-training/model_weights/Qwen3-4B-Base")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--merge-after-train", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    train_raw = _load_json_dataset(args.train_jsonl)
    eval_raw = _load_json_dataset(args.eval_jsonl)
    train_rows_before_filter = int(train_raw.num_rows)
    eval_rows_before_filter = int(eval_raw.num_rows)

    train_raw = train_raw.filter(
        lambda example: _has_valid_supervised_target(example["messages"]),
        batched=False,
        desc="Filtering invalid train examples",
    )
    eval_raw = eval_raw.filter(
        lambda example: _has_valid_supervised_target(example["messages"]),
        batched=False,
        desc="Filtering invalid eval examples",
    )

    if train_raw.num_rows == 0:
        raise RuntimeError("No valid train examples remain after filtering invalid supervision rows.")
    if eval_raw.num_rows == 0:
        raise RuntimeError("No valid eval examples remain after filtering invalid supervision rows.")

    def _to_features(example: dict[str, Any]) -> dict[str, Any]:
        texts = _messages_to_texts(tokenizer, example["messages"])
        return _mask_to_last_assistant(tokenizer, texts["prompt_text"], texts["full_text"], args.max_seq_len)

    train_tok = train_raw.map(_to_features, batched=False, desc="Tokenizing train subset")
    eval_tok = eval_raw.map(_to_features, batched=False, desc="Tokenizing eval subset")

    collator = CausalLMPaddingCollator(tokenizer.pad_token_id, -100)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16_ok,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
    )

    train_start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_start
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    eval_start = time.perf_counter()
    eval_metrics = trainer.evaluate(eval_dataset=eval_tok)
    eval_seconds = time.perf_counter() - eval_start
    eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
    eval_ppl = float(math.exp(eval_loss)) if math.isfinite(eval_loss) else float("nan")
    eval_metrics["eval_perplexity"] = eval_ppl

    merged_model_dir = None
    if args.merge_after_train:
        merged_model_dir = output_dir / "merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_model_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_model_dir))

    results = {
        "config": {
            "train_jsonl": str(Path(args.train_jsonl).expanduser().resolve()),
            "eval_jsonl": str(Path(args.eval_jsonl).expanduser().resolve()),
            "output_dir": str(output_dir),
            "model_name": args.model_name,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        },
        "dataset": {
            "train_rows_before_filter": train_rows_before_filter,
            "train_rows": int(train_raw.num_rows),
            "eval_rows_before_filter": eval_rows_before_filter,
            "eval_rows": int(eval_raw.num_rows),
        },
        "timings": {
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        },
        "metrics": {
            **{key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
            **{key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))},
        },
        "artifacts": {
            "adapter_dir": str(output_dir),
            "merged_model_dir": str(merged_model_dir) if merged_model_dir is not None else None,
            "trainer_state_json": str(output_dir / "trainer_state.json"),
        },
    }

    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    report_path.write_text(_render_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

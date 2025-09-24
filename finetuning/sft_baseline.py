#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Supervised/Instruction Fine-Tuning (SFT) for Causal LMs on (instruction,input)->output pairs.

Expected data format (JSONL):
{"instruction": "Summarize:", "input": "Text...", "output": "Summary..."}
{"instruction": "Translate to French:", "input": "Hello", "output": "Bonjour"}
{"instruction": "Write a a bedtime story", "output": "..."}

Usage (full SFT):
  python sft_pairs.py --model gpt2 --train_file train.jsonl --eval_file dev.jsonl --out_dir ./sft_out

Usage (LoRA):
  python sft_pairs.py --model meta-llama/Llama-3.1-8B --train_file train.jsonl \
      --eval_file dev.jsonl --out_dir ./lora_out --use_lora \
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
"""
import argparse, json, math, os
from typing import Dict

import datasets
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


PROMPT_TEMPLATE = """### Instruction:
{instruction}
{maybe_input}### Response:
{output}"""

def format_example(ex: Dict) -> str:
    instr = ex.get("instruction", "").strip()
    inp = (ex.get("input") or "").strip()
    out = ex.get("output", "").strip()
    maybe_input = f"### Input:\n{inp}\n" if inp else ""
    return PROMPT_TEMPLATE.format(instruction=instr, maybe_input=maybe_input, output=out).strip()


def tokenize_fn(examples: Dict, tokenizer: AutoTokenizer, eos_token_id: int, max_len: int):
    # Build full prompt (including target) and make labels == input_ids (causal LM objective)
    texts = [format_example(ex) + tokenizer.eos_token for ex in examples["raw"]]
    toks = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name (e.g., meta-llama/Llama-3.2-8B)")
    ap.add_argument("--train_file", required=True, help="Path to train.jsonl")
    ap.add_argument("--eval_file", help="Path to eval.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA PEFT")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="*", default=None,
                    help="Module name patterns for LoRA (e.g., q_proj k_proj v_proj o_proj)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        # For decoder-only models, often pad == eos
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Optional LoRA
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed. Try: pip install peft")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Load data (keeps original rows as "raw" to format later)
    def _load(path):
        # load_dataset handles jsonl if we specify 'json' and 'lines=True'
        ds = load_dataset("json", data_files=path, split="train")
        # Store the raw dict per row so we can format flexibly downstream
        ds = ds.map(lambda ex: {"raw": {k: ex.get(k) for k in ex.keys()}}, remove_columns=ds.column_names)
        return ds

    train_ds = _load(args.train_file)
    eval_ds = _load(args.eval_file) if args.eval_file else None

    # Tokenize
    def _tok(batch):
        return tokenize_fn(batch, tokenizer, tokenizer.eos_token_id, args.max_length)

    train_ds = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training config
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps if eval_ds else None,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Perplexity on eval (optional)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Shift to align predictions with labels
        import numpy as np
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        # Cross-entropy
        from torch.nn import CrossEntropyLoss
        ce = CrossEntropyLoss()
        # Convert to torch tensors
        shift_logits = torch.from_numpy(shift_logits).float()
        shift_labels = torch.from_numpy(shift_labels).long()
        if torch.cuda.is_available():
            shift_logits = shift_logits.cuda()
            shift_labels = shift_labels.cuda()
        loss = ce(shift_logits, shift_labels)
        ppl = float(math.exp(loss.item())) if loss.item() < 20 else float("inf")
        return {"eval_loss_ce": loss.item(), "perplexity": ppl}

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_ds else None,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # If LoRA, also save adapters nicely
    if args.use_lora:
        try:
            model.save_pretrained(args.out_dir)
        except Exception:
            pass

    print("Training complete. Model saved to:", args.out_dir)


if __name__ == "__main__":
    main()

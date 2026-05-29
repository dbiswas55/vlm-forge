"""Smoke test — run this before every full training run.

    python -m src.test     (from the repo root)

Loads the model in 4-bit, processes a tiny subset of ChartQA, and runs ~10
training steps. Should finish in ~2 minutes on a single L40S and print
`Smoke test PASSED` at the end.

Catches the four bugs that kill most first runs:
    - bad image-token masking (NaN loss)
    - processor / chat template mismatches
    - OOM at the chosen batch size
    - environment / library version issues
"""
from __future__ import annotations

import sys
import traceback

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from src.utils import (
    MODEL_ID,
    build_collator,
    format_example,
    get_image_token_id,
    load_model_and_processor,
)


def main() -> int:
    print(f"[test] Model: {MODEL_ID}")
    print(f"[test] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[test] Device: {torch.cuda.get_device_name(0)}")

    # 1) Load a tiny slice of the dataset
    print("[test] Loading 32 train + 8 eval examples from ChartQA...")
    train_ds = load_dataset("HuggingFaceM4/ChartQA", split="train[:32]")
    eval_ds = load_dataset("HuggingFaceM4/ChartQA", split="val[:8]")

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

    # 2) Load model + processor (4-bit)
    print("[test] Loading model (4-bit QLoRA)...")
    try:
        model, processor = load_model_and_processor(
            quantize=True, use_flash_attn=True
        )
    except Exception:
        print("[test] flash-attn unavailable, retrying without it...")
        model, processor = load_model_and_processor(
            quantize=True, use_flash_attn=False
        )

    image_token_id = get_image_token_id(processor)
    print(f"[test] image_token_id={image_token_id}")
    if image_token_id is None:
        print("[test] WARNING: image token not detected — "
              "check src/utils/model.py:get_image_token_id")

    # 3) Build the collator and confirm one batch works
    collate = build_collator(processor)
    sample_batch = collate([train_ds[0], train_ds[1]])
    print(f"[test] sample batch keys: {list(sample_batch.keys())}")
    print(f"[test] input_ids shape: {sample_batch['input_ids'].shape}")
    print(f"[test] pixel_values present: {'pixel_values' in sample_batch}")

    # 4) Tiny LoRA + tiny SFTConfig
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = SFTConfig(
        output_dir="outputs/smoke_test",
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=2,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        max_length=None,                    # CRITICAL for VLMs
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    print("[test] Building trainer and running 10 steps...")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        data_collator=collate,
    )
    trainer.train()

    # Sanity check: loss finite?
    if trainer.state.log_history:
        last_loss = next(
            (h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h),
            None,
        )
        print(f"[test] last logged loss: {last_loss}")
        if last_loss is None or not (last_loss == last_loss):   # NaN check
            print("[test] FAIL: loss was NaN or missing")
            return 1

    print("[test] Smoke test PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("[test] Smoke test FAILED")
        sys.exit(1)

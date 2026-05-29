"""Main training script.

Single GPU (from the repo root):
    python -m src.train

Multi-GPU (DDP via Accelerate):
    accelerate launch --config_file config/accelerate_ddp.yaml \
        --num_processes 4 -m src.train

The LoRA adapter is saved to outputs/gemma3-4b-chartqa-qlora/.
"""
from __future__ import annotations

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from src.utils import (
    MODEL_ID,
    build_collator,
    format_example,
    load_model_and_processor,
)

OUTPUT_DIR = "outputs/gemma3-4b-chartqa-qlora"


def main() -> None:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Loading HuggingFaceM4/ChartQA for {MODEL_ID}")
    raw = load_dataset("HuggingFaceM4/ChartQA")
    train_ds = raw["train"]
    eval_ds = raw["val"].select(range(500))     # small eval for speed

    # Keep the "image" column — removing it causes PIL→dict serialization issues
    drop = [c for c in train_ds.column_names if c != "image"]
    train_ds = train_ds.map(format_example, remove_columns=drop)
    drop = [c for c in eval_ds.column_names if c != "image"]
    eval_ds = eval_ds.map(format_example, remove_columns=drop)

    print(f"train: {len(train_ds)}  eval: {len(eval_ds)}")

    # ------------------------------------------------------------------
    # Model + processor (4-bit QLoRA)
    # ------------------------------------------------------------------
    try:
        model, processor = load_model_and_processor(
            quantize=True, use_flash_attn=True
        )
    except Exception as e:
        print(f"flash-attn unavailable ({e}); falling back to SDPA")
        model, processor = load_model_and_processor(
            quantize=True, use_flash_attn=False
        )

    collate = build_collator(processor)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=200,
        report_to="wandb",
        max_length=None,                    # CRITICAL for VLMs
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        data_collator=collate,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

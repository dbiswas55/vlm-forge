"""Predict on a single image + question with a trained LoRA adapter.

Usage (from the repo root):
    python -m src.predict \
        --adapter_dir outputs/gemma3-4b-chartqa-qlora \
        --image path/to/chart.png \
        --question "What is the highest value shown?"

Omit --adapter_dir (or pass "") to run the base model — handy for
before/after qualitative comparisons.
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from PIL import Image

from src.utils import SYSTEM_PROMPT, load_model_and_processor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", default="",
                   help="Path to LoRA adapter (empty = base model only)")
    p.add_argument("--image", required=True, help="Path to the image")
    p.add_argument("--question", required=True, help="Question about the image")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--no_quantize", action="store_true",
                   help="Load the base model in bf16 instead of 4-bit")
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")

    print("Loading model...")
    model, processor = load_model_and_processor(
        quantize=not args.no_quantize,
        use_flash_attn=False,
    )
    if args.adapter_dir:
        print(f"Loading adapter from {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [
             {"type": "image"},   # placeholder; image passed via processor images=
             {"type": "text", "text": args.question},
         ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]],
                       return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    answer = processor.tokenizer.decode(
        out[0][prompt_len:], skip_special_tokens=True
    ).strip()
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()

"""Evaluate a trained LoRA adapter on the ChartQA test split.

Reports relaxed accuracy:
    - numeric answers match if within 5% (standard ChartQA metric)
    - non-numeric answers match if equal after case/whitespace normalization

Usage (from the repo root):
    python -m src.evaluate \
        --adapter_dir outputs/gemma3-4b-chartqa-qlora \
        --num_samples 500
"""
from __future__ import annotations

import argparse
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from src.utils import SYSTEM_PROMPT, load_model_and_processor


# ---------------------------------------------------------------------------
# Metric: ChartQA relaxed accuracy
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def to_number(s: str):
    s = s.strip().rstrip("%").replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    pred = pred.strip()
    gold = gold.strip()

    p_num = to_number(pred)
    g_num = to_number(gold)
    if p_num is not None and g_num is not None:
        if g_num == 0.0:
            return abs(p_num - g_num) <= tol
        return abs(p_num - g_num) / abs(g_num) <= tol

    return pred.lower() == gold.lower()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_answer(model, processor, image, question: str,
                    max_new_tokens: int = 32) -> str:
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": image},
             {"type": "text", "text": question},
         ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]],
                       return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.tokenizer.decode(
        out[0][prompt_len:], skip_special_tokens=True
    )
    return decoded.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", required=True,
                   help="Path to the saved LoRA adapter")
    p.add_argument("--split", default="test",
                   help="ChartQA split: train / val / test")
    p.add_argument("--num_samples", type=int, default=500,
                   help="How many examples to evaluate (-1 for all)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    args = p.parse_args()

    print(f"Loading base model and adapter from {args.adapter_dir}")
    model, processor = load_model_and_processor(quantize=True,
                                                use_flash_attn=False)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    if args.num_samples > 0:
        split_str = f"{args.split}[:{args.num_samples}]"
    else:
        split_str = args.split
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split_str)
    print(f"Evaluating on {len(ds)} examples ({args.split})")

    correct = 0
    samples_printed = 0
    for ex in tqdm(ds):
        gold = ex["label"][0]
        pred = generate_answer(model, processor, ex["image"], ex["query"],
                               max_new_tokens=args.max_new_tokens)
        if relaxed_match(pred, gold):
            correct += 1
        if samples_printed < 5:
            print(f"\nQ: {ex['query']}\nGOLD: {gold}\nPRED: {pred}")
            samples_printed += 1

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    print(f"\nRelaxed accuracy: {correct}/{len(ds)} = {acc:.4f}")


if __name__ == "__main__":
    main()

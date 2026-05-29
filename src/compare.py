"""Before/after evaluation: base model vs. fine-tuned LoRA adapter.

Generates answers on a held-out ChartQA split (default `test`, which is NOT
used during training) with BOTH the base model and the fine-tuned adapter,
scores each with ChartQA relaxed accuracy, and writes a comparative report.

The base model is loaded once; the adapter is then attached on top of the same
4-bit weights, so both models see identical inputs and the comparison is fair.

Usage (from the repo root):
    python -m src.compare \
        --adapter_dir outputs/gemma3-4b-chartqa-qlora \
        --split test \
        --num_samples 500 \
        --out_dir outputs/eval_compare

Outputs (written to --out_dir):
    predictions.jsonl   one row per example: query, gold, base/ft preds + flags
    summary.json        accuracies, deltas, and per-bucket counts
    report.md           human-readable before/after report
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from src.utils import MODEL_ID, load_model_and_processor
# Reuse the exact same generation + metric as src.evaluate so numbers are
# directly comparable to a single-model `python -m src.evaluate` run.
from src.evaluate import generate_answer, relaxed_match


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", default="outputs/gemma3-4b-chartqa-qlora",
                   help="Path to the saved LoRA adapter (the fine-tuned model)")
    p.add_argument("--split", default="test",
                   help="ChartQA split: train / val / test. test is held out.")
    p.add_argument("--num_samples", type=int, default=500,
                   help="How many examples to evaluate (-1 for the whole split)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--out_dir", default="outputs/eval_compare",
                   help="Directory for predictions.jsonl, summary.json, report.md")
    p.add_argument("--num_examples_in_report", type=int, default=10,
                   help="How many improved/regressed cases to show in report.md")
    return p.parse_args()


def run_model(model, processor, dataset, max_new_tokens: int) -> list[str]:
    """Greedy-generate one answer per example, in dataset order."""
    preds = []
    for ex in tqdm(dataset, desc="generating"):
        pred = generate_answer(
            model, processor, ex["image"], ex["query"],
            max_new_tokens=max_new_tokens,
        )
        preds.append(pred)
    return preds


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.adapter_dir or not os.path.isdir(args.adapter_dir):
        raise SystemExit(
            f"--adapter_dir '{args.adapter_dir}' is not a directory. "
            "Point it at the trained adapter, e.g. outputs/gemma3-4b-chartqa-qlora"
        )

    # ------------------------------------------------------------------
    # Data (held-out split, not seen during training)
    # ------------------------------------------------------------------
    if args.num_samples > 0:
        split_str = f"{args.split}[:{args.num_samples}]"
    else:
        split_str = args.split
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split_str)
    golds = [ex["label"][0] for ex in ds]
    queries = [ex["query"] for ex in ds]
    n = len(ds)
    print(f"Evaluating BASE vs FINE-TUNED on {n} '{args.split}' examples\n")

    # ------------------------------------------------------------------
    # (a) Base model — no adapter
    # Read the base model id from the adapter config so the base ALWAYS
    # matches the adapter (a 12B adapter on a 4B base would be meaningless).
    # ------------------------------------------------------------------
    with open(os.path.join(args.adapter_dir, "adapter_config.json")) as f:
        base_model_id = json.load(f).get("base_model_name_or_path", MODEL_ID)
    print(f"[1/2] Loading base model {base_model_id} (4-bit, no adapter)")
    model, processor = load_model_and_processor(
        model_id=base_model_id, quantize=True, use_flash_attn=False
    )
    model.eval()
    base_preds = run_model(model, processor, ds, args.max_new_tokens)

    # ------------------------------------------------------------------
    # (b) Fine-tuned model — same base weights + LoRA adapter on top
    # ------------------------------------------------------------------
    print(f"\n[2/2] Attaching adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    ft_preds = run_model(model, processor, ds, args.max_new_tokens)

    # ------------------------------------------------------------------
    # Score + per-example comparison
    # ------------------------------------------------------------------
    rows = []
    base_correct = ft_correct = 0
    improved = regressed = both_right = both_wrong = 0
    for i in range(n):
        b_ok = relaxed_match(base_preds[i], golds[i])
        f_ok = relaxed_match(ft_preds[i], golds[i])
        base_correct += b_ok
        ft_correct += f_ok
        if b_ok and f_ok:
            both_right += 1
            category = "both_right"
        elif not b_ok and f_ok:
            improved += 1
            category = "improved"
        elif b_ok and not f_ok:
            regressed += 1
            category = "regressed"
        else:
            both_wrong += 1
            category = "both_wrong"
        rows.append({
            "index": i,
            "query": queries[i],
            "gold": golds[i],
            "base_pred": base_preds[i],
            "base_correct": b_ok,
            "ft_pred": ft_preds[i],
            "ft_correct": f_ok,
            "category": category,
        })

    base_acc = base_correct / n if n else 0.0
    ft_acc = ft_correct / n if n else 0.0
    summary = {
        "model_id": base_model_id,
        "adapter_dir": args.adapter_dir,
        "split": args.split,
        "num_samples": n,
        "max_new_tokens": args.max_new_tokens,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base": {"correct": base_correct, "accuracy": base_acc},
        "finetuned": {"correct": ft_correct, "accuracy": ft_acc},
        "absolute_gain": ft_acc - base_acc,
        "relative_gain": (ft_acc - base_acc) / base_acc if base_acc else None,
        "buckets": {
            "improved": improved,        # base wrong -> fine-tuned right
            "regressed": regressed,      # base right -> fine-tuned wrong
            "both_right": both_right,
            "both_wrong": both_wrong,
        },
    }

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    pred_path = os.path.join(args.out_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(args.out_dir, "report.md")
    write_report(report_path, summary, rows, args.num_examples_in_report)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 56)
    print(f"{'':20}{'BASE':>12}{'FINE-TUNED':>14}")
    print(f"{'Relaxed accuracy':20}{base_acc:>12.4f}{ft_acc:>14.4f}")
    print(f"{'Correct / total':20}{f'{base_correct}/{n}':>12}{f'{ft_correct}/{n}':>14}")
    print("-" * 56)
    print(f"Absolute gain : {ft_acc - base_acc:+.4f} "
          f"({(ft_acc - base_acc) * 100:+.2f} pts)")
    if base_acc:
        print(f"Relative gain : {(ft_acc - base_acc) / base_acc * 100:+.1f}%")
    print(f"Improved (wrong->right): {improved}   "
          f"Regressed (right->wrong): {regressed}")
    print(f"Both right: {both_right}   Both wrong: {both_wrong}")
    print("=" * 56)
    print(f"\nWrote:\n  {pred_path}\n  {summary_path}\n  {report_path}")


def write_report(path: str, summary: dict, rows: list[dict], k: int) -> None:
    b = summary["base"]
    ft = summary["finetuned"]
    gain = summary["absolute_gain"]
    rel = summary["relative_gain"]
    lines = [
        f"# ChartQA before/after — {summary['model_id']}",
        "",
        f"- Adapter: `{summary['adapter_dir']}`",
        f"- Split: `{summary['split']}` (held out from training)",
        f"- Examples: {summary['num_samples']}",
        f"- Generated: {summary['timestamp_utc']}",
        "",
        "## Relaxed accuracy",
        "",
        "| Model | Correct | Accuracy |",
        "|---|---|---|",
        f"| Base (no fine-tuning) | {b['correct']}/{summary['num_samples']} | {b['accuracy']:.4f} |",
        f"| Fine-tuned (QLoRA) | {ft['correct']}/{summary['num_samples']} | {ft['accuracy']:.4f} |",
        "",
        f"**Absolute gain:** {gain:+.4f} ({gain * 100:+.2f} pts)  ",
        f"**Relative gain:** {rel * 100:+.1f}%" if rel is not None else "**Relative gain:** n/a",
        "",
        "## Where the change came from",
        "",
        "| Bucket | Count |",
        "|---|---|",
        f"| Improved (base wrong → fine-tuned right) | {summary['buckets']['improved']} |",
        f"| Regressed (base right → fine-tuned wrong) | {summary['buckets']['regressed']} |",
        f"| Both right | {summary['buckets']['both_right']} |",
        f"| Both wrong | {summary['buckets']['both_wrong']} |",
        "",
    ]

    def example_block(title: str, category: str) -> None:
        picked = [r for r in rows if r["category"] == category][:k]
        lines.append(f"## {title} (showing up to {k})")
        lines.append("")
        if not picked:
            lines.append("_None._")
            lines.append("")
            return
        for r in picked:
            lines.append(f"- **Q:** {r['query']}")
            lines.append(f"  - GOLD: `{r['gold']}`")
            lines.append(f"  - base: `{r['base_pred']}`  →  fine-tuned: `{r['ft_pred']}`")
        lines.append("")

    example_block("Examples fixed by fine-tuning", "improved")
    example_block("Examples broken by fine-tuning", "regressed")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

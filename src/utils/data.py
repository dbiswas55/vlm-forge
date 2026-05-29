"""Data formatting: ChartQA rows -> chat messages -> padded model batches."""
from __future__ import annotations

from .model import get_image_token_id

SYSTEM_PROMPT = (
    "You are a vision-language model specialized in interpreting charts. "
    "Answer the question with a single number, word, or short phrase."
)


def format_example(example: dict, include_answer: bool = True) -> dict:
    """Format a ChartQA row into the TRL chat-messages structure.

    The dataset's `label` field is a list; we take the first element.
    Set include_answer=False when formatting prompts for generation.
    """
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": example["image"]},
             {"type": "text", "text": example["query"]},
         ]},
    ]
    if include_answer:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example["label"][0]}],
        })
    return {"messages": messages}


def build_collator(processor, max_length: int = 2048):
    """Returns a function that turns a list of formatted examples into a
    padded model batch with labels masked correctly (pad + image tokens).
    """
    image_token_id = get_image_token_id(processor)
    pad_token_id = processor.tokenizer.pad_token_id

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            msgs = ex["messages"]
            texts.append(
                processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
            imgs = [c["image"] for m in msgs for c in m["content"]
                    if c["type"] == "image"]
            images.append(imgs)

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = batch["input_ids"].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate

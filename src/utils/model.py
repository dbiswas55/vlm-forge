"""Model and processor loading.

Change MODEL_ID and ModelClass here to swap to Gemma 3 12B or Qwen2.5-VL.
The rest of the pipeline is generic.
"""
from __future__ import annotations

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# The only two lines you change to swap models.
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-4b-it"
ModelClass = Gemma3ForConditionalGeneration


def load_model_and_processor(
    model_id: str = MODEL_ID,
    quantize: bool = True,
    use_flash_attn: bool = True,
):
    """Load the VLM (optionally 4-bit quantized) and its processor."""
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    kwargs = dict(
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if quantization_config is not None:
        kwargs["device_map"] = "auto"   # required for multi-shard 4-bit loading
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = ModelClass.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor


def get_image_token_id(processor) -> int | None:
    """Best-effort lookup of the soft image-placeholder token id.

    Different VLM families and library versions expose this differently.
    Returning None is treated as "don't mask image tokens"; if your loss
    is NaN from step 0, this is almost always the cause — print
    `processor.tokenizer.special_tokens_map` and add the right candidate.
    """
    # Newer processors expose it directly
    for attr in ("image_token_id", "image_token"):
        val = getattr(processor, attr, None)
        if isinstance(val, int):
            return val

    # Fall back to a list of known candidates
    tokenizer = processor.tokenizer
    for candidate in ("<image_soft_token>", "<image>", "<|image|>"):
        tid = tokenizer.convert_tokens_to_ids(candidate)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return None

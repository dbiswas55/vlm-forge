# vlm-forge

QLoRA fine-tuning of vision-language models. Default config trains
`google/gemma-3-4b-it` on `HuggingFaceM4/ChartQA` using HuggingFace TRL, PEFT,
and bitsandbytes. Single- and multi-GPU launchers, plus test/train/eval/predict
scripts. Built for L40S clusters.

## What this is

A minimal, single-day fine-tuning setup that covers four things in one go:

- **SFT / LoRA** — supervised fine-tuning of a vision-language model via
  HuggingFace `TRL` + `PEFT`.
- **Quantization (QLoRA)** — base model loaded in 4-bit NF4 via `bitsandbytes`.
- **Efficient inference** — generate from the merged adapter at ~half the memory
  cost of full fine-tuning.
- **Distributed training** — same code runs on 1 GPU or N GPUs via
  `accelerate launch`. No FSDP / DeepSpeed required because QLoRA's per-GPU
  footprint is small.

## Repository layout

```
vlm-forge/
├── README.md
├── requirements.txt
├── .gitignore
├── submit.sh                    # SLURM entrypoint
├── config/
│   └── accelerate_ddp.yaml      # Multi-GPU DDP config
├── outputs/                     # gitignored, training writes here
└── src/
    ├── __init__.py
    ├── test.py                  # smoke test (~2 min)
    ├── train.py                 # main training loop
    ├── evaluate.py              # relaxed accuracy on the test split
    ├── predict.py               # single-image CLI
    └── utils/
        ├── __init__.py          # re-exports for clean imports
        ├── model.py             # MODEL_ID, model + processor loading
        └── data.py              # SYSTEM_PROMPT, message formatting, collator
```

All scripts run as modules from the repo root: `python -m src.train`,
`python -m src.evaluate`, etc.

## Hardware

- Designed for **NVIDIA L40S (48 GB)**. Will also work on A100, H100,
  RTX 6000 Ada, or anything with bf16 + Flash-Attention support.
- 1 GPU is enough for the 4B model. 12B also fits on a single L40S with
  QLoRA; use 2+ GPUs to halve wall-clock.

### About Mac (M-series) support

**Not supported.** `bitsandbytes` requires CUDA — there is no Apple Silicon /
MPS backend for 4-bit quantization, and `flash-attention` is also CUDA-only.
Run everything on the server. Use `src/test.py` for the fast iteration loop.

## Setup

Gemma 3 is a gated model — accept the license on the
[model page](https://huggingface.co/google/gemma-3-4b-it) first, then log in:

```bash
huggingface-cli login
```

Create a venv and install:

```bash
module load cuda/12.4   # whatever your cluster provides
python -m venv ~/envs/vlm-ft
source ~/envs/vlm-ft/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional but recommended on L40S
```

## Usage

The intended workflow is **test → train → evaluate → predict**. All commands
are run from the repo root.

### 1. Smoke test (~2 minutes on 1× L40S)

Verifies the model loads, data flows through the collator, and a few training
steps run without errors. Run this **before** every full training run —
especially after changing the model, dataset, or collator.

```bash
python -m src.test
```

Successful output ends with `Smoke test PASSED`.

### 2. Train

Single GPU:

```bash
python -m src.train
```

Multi-GPU (DDP via Accelerate):

```bash
accelerate launch --config_file config/accelerate_ddp.yaml \
    --num_processes 4 \
    -m src.train
```

On a SLURM cluster:

```bash
sbatch submit.sh
```

The LoRA adapter is written to `outputs/gemma3-4b-chartqa-qlora/`.

### 3. Evaluate

Runs inference on the ChartQA test split and reports relaxed accuracy
(numeric answers match within 5%, text answers match case-insensitively).

```bash
python -m src.evaluate \
    --adapter_dir outputs/gemma3-4b-chartqa-qlora \
    --num_samples 500
```

### 4. Predict

Single-example CLI. Pass any chart image and a question:

```bash
python -m src.predict \
    --adapter_dir outputs/gemma3-4b-chartqa-qlora \
    --image path/to/chart.png \
    --question "What is the highest value shown?"
```

Omit `--adapter_dir` (or pass `""`) to run the base model — useful for
before/after qualitative comparisons.

## Configuration knobs

Training hyperparameters live in `src/train.py` (`SFTConfig`):

| Knob | Default | Notes |
|---|---|---|
| `MODEL_ID` (`src/utils/model.py`) | `google/gemma-3-4b-it` | Swap to `-12b-it` or Qwen2.5-VL |
| `num_train_epochs` | 1 | 1 is plenty for ChartQA at 28k examples |
| `per_device_train_batch_size` | 4 | Raise to 8 if you have headroom |
| `gradient_accumulation_steps` | 4 | Effective batch = bs × grad_acc × num_gpus |
| `learning_rate` | 2e-4 | Standard QLoRA LR |
| LoRA `r` | 16 | 32 if you want more capacity |
| LoRA `target_modules` | attention + MLP | Vision tower stays frozen |

## Notes on the data path

- The dataset's `image` field is a PIL Image, passed straight to the processor.
- `label` is a list; we take `label[0]`.
- `SFTConfig(max_length=None)` is **mandatory** for VLMs — truncating image
  tokens silently breaks training.
- Image tokens and pad tokens are masked from the loss in the collator.

## Switching models later

Edit `src/utils/model.py` — change `MODEL_ID` and (for non-Gemma families) the
`ModelClass`. For Qwen2.5-VL:

```python
from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
```

LoRA target modules and the chat template work as-is for Qwen2.5-VL. Always
re-run `python -m src.test` after a swap to confirm the image-token detection
finds the right token for the new family.

## Troubleshooting

- **`flash-attn` won't install** — drop `attn_implementation="flash_attention_2"`
  from `src/utils/model.py`. SDPA still works, just slightly slower.
- **OOM on 1× L40S with 4B** — lower `per_device_train_batch_size` to 2 and
  raise `gradient_accumulation_steps` to 8.
- **Loss is `nan` from step 0** — almost always the image-token mask. Print
  `processor.tokenizer.special_tokens_map` and confirm the image token name in
  `src/utils/model.py:get_image_token_id`.
- **`huggingface-cli` says gated** — accept the license on the model page first.

## License

MIT.

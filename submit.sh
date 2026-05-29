#!/bin/bash
#SBATCH -J vlm-forge
#SBATCH -o logs/vlm-forge.o%j
#SBATCH -e logs/vlm-forge.e%j
#SBATCH --mail-user=dipayan1109033@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -t 18:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=ada:1            # change to ada:4 for multi-GPU
 
cd /project/subhlok/dipayan/vlm-forge
source /project/subhlok/dipayan/my_envs/venv312/bin/activate

# flash-attn skipped: cluster compute nodes have no nvcc (CUDA compiler).
# train.py and test.py fall back to torch SDPA automatically.

mkdir -p logs
 
# Load secrets from .env (gitignored) — copy .env.example to .env and fill in values
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your tokens."
    exit 1
fi

export HF_HOME="/project/subhlok/dipayan/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export WANDB_PROJECT="vlm-forge"

# ---- Run configuration ----------------------------------------------------
# Swap models here. train.py reads these env vars (defaults reproduce the 4B
# run). 12B in 4-bit fits one L40S, but needs a smaller per-device batch; we
# keep the effective batch at 16 (2 x 8) to match the 4B run.
#   4B  : MODEL_ID=google/gemma-3-4b-it   bs=4 ga=4  OUTPUT_DIR=...gemma3-4b...
#   12B : MODEL_ID=google/gemma-3-12b-it  bs=2 ga=8  OUTPUT_DIR=...gemma3-12b...
export MODEL_ID="google/gemma-3-12b-it"
export OUTPUT_DIR="outputs/gemma3-12b-chartqa-qlora"
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8

# Detect GPU count from SLURM env (set by --gpus-per-node)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching on $NUM_GPUS GPU(s)"
nvidia-smi --query-gpu=name,memory.total --format=csv
 
# Run smoke test before training — aborts the job if it fails
echo "Running smoke test..."
python -m src.test
if [ $? -ne 0 ]; then
    echo "Smoke test FAILED — aborting job."
    exit 1
fi
echo "Smoke test passed. Launching training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch --config_file config/accelerate_ddp.yaml \
        --num_processes "$NUM_GPUS" \
        -m src.train
else
    python -m src.train
fi
 
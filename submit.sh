#!/bin/bash
#SBATCH --job-name=vlm-forge
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1            # change to gpu:l40s:4 for multi-GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

# Activate your env (edit path as needed)
source ~/envs/vlm-ft/bin/activate

# Use fast scratch storage for HF cache
export HF_HOME=${SCRATCH:-$HOME}/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching on $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch --config_file config/accelerate_ddp.yaml \
        --num_processes "$NUM_GPUS" \
        -m src.train
else
    python -m src.train
fi

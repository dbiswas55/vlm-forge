#!/bin/bash
#SBATCH -J vlm-forge
#SBATCH -o logs/vlm-forge.o%j
#SBATCH -e logs/vlm-forge.e%j
#SBATCH --mail-user=dipayan1109033@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=ada:1            # change to ada:4 for multi-GPU
 
cd /project/subhlok/dipayan/vlm-forge
source /project/subhlok/dipayan/my_envs/venv312/bin/activate

# flash-attn skipped: cluster compute nodes have no nvcc (CUDA compiler).
# train.py and test.py fall back to torch SDPA automatically.

mkdir -p logs
 
export HF_TOKEN="REPLACE_WITH_YOUR_TOKEN"
export HF_HOME="/project/subhlok/dipayan/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export WANDB_PROJECT="vlm-forge"
 
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
 
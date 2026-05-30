#!/bin/bash
#SBATCH -J vlm-eval
#SBATCH -o logs/vlm-eval.o%j
#SBATCH -e logs/vlm-eval.e%j
#SBATCH --mail-user=dipayan1109033@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=16GB
#SBATCH --gpus-per-node=ada:1

cd /project/subhlok/dipayan/vlm-forge
source /project/subhlok/dipayan/my_envs/venv312/bin/activate

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

# Evaluation knobs — override on the command line if you like:
#   sbatch submit_eval.sh outputs/gemma3-12b-chartqa-qlora test 1000
# The base model is NOT taken from MODEL_ID here — src.compare reads it from the
# adapter's adapter_config.json, so it always matches the adapter you point at.
#   4B  adapter: outputs/gemma3-4b-chartqa-qlora
#   12B adapter: outputs/gemma3-12b-chartqa-qlora  (matches OUTPUT_DIR in submit.sh)
ADAPTER_DIR=${1:-outputs/gemma3-12b-chartqa-qlora}
SPLIT=${2:-test}
NUM_SAMPLES=${3:-500}
# Keep results separate per model variation so a new run never overwrites an old
# one. The adapter dir name already encodes the model (gemma3-4b / gemma3-12b),
# so derive the tag from it: outputs/eval_compare/gemma3-12b-chartqa-qlora/...
MODEL_TAG=$(basename "$ADAPTER_DIR")
OUT_DIR=${4:-outputs/eval_compare/$MODEL_TAG}

nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Comparing BASE vs FINE-TUNED:"
echo "  adapter=$ADAPTER_DIR  split=$SPLIT  num_samples=$NUM_SAMPLES  out=$OUT_DIR"

python -m src.compare \
    --adapter_dir "$ADAPTER_DIR" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --out_dir "$OUT_DIR"

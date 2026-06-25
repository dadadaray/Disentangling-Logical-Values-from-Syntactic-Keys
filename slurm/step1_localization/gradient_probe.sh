#!/bin/bash
#SBATCH --job-name=grad_probe
#SBATCH --output=logs/grad_probe_%j.out
#SBATCH --error=logs/grad_probe_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ===== 环境 =====
source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit

export PYTHONPATH=$(pwd):$PYTHONPATH

# ===== 参数 =====
MODEL_PATH="/data/users/yanrongen/AnyEdit/LLM-Qwen2.5-7B"
DATASET_PATH="data/AKEW/MQuAKE-CF.json"
SAVE_DIR="results/gradient_probe"
NUM_SAMPLES=500

mkdir -p logs ${SAVE_DIR}

echo "🚀 Running Gradient Probe"
echo "Model:   ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "--------------------------------------"

python experiments/gradient_probe.py \
    --model_name ${MODEL_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --save_dir ${SAVE_DIR}

echo "✅ Gradient Probe Finished"

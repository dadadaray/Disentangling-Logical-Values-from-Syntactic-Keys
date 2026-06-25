#!/bin/bash
#SBATCH --job-name=mom2_spec
#SBATCH --output=logs/mom2_spec_%j.out
#SBATCH --error=logs/mom2_spec_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00

source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit
export PYTHONPATH=$(pwd):$PYTHONPATH

# ================= 路径配置 =================
BLIMP_ROOT="/data/users/yanrongen/ROME-MEMIT/data/blimp"
OUTPUT_DIR="analysis_results/mom2_spectral"
STATS_ROOT="/data/users/yanrongen/ROME-MEMIT/data/stats"
LAYERS=(26)

mkdir -p logs "${OUTPUT_DIR}"

# ========================================================
# 任务 2: Llama-3-8B
# ========================================================
echo "--------------------------------------------------------"
echo "🤖 Processing: Llama"
echo "--------------------------------------------------------"

MODEL_PATH="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
STATS_FOLDER="LLM-Llama-3-8B-Instruct"

for LAYER in "${LAYERS[@]}"; do
    echo "--> Llama Layer ${LAYER}..."
    python analyze_spectrum.py \
        --model_name "${MODEL_PATH}" \
        --model_alias "llama" \
        --layer "${LAYER}" \
        --blimp_root "${BLIMP_ROOT}" \
        --stats_dir "${STATS_ROOT}" \
        --stats_model_name "${STATS_FOLDER}" \
        --output_dir "${OUTPUT_DIR}"
done

echo "🎉 All done."
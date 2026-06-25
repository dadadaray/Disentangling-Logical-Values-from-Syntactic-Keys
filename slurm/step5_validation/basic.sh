#!/bin/bash

# ==============================================================================
# 🎛️ 变量控制台
# ==============================================================================

TARGET_LAYERS="26"

# REMA
REMA_K="64"
REMA_SRC="math"

# MOM2 / TRIM
COV_MODE="pca_k"        # deltaI | mom2 | pca_k
PCA_K="128"             # MOM2 PCA 维度
DELTA="0.001"
MOM2_WEIGHT="15000"

# 基础配置
ALG_NAME="MEMIT"
ENV_NAME="yanrongen_anyedit"
MODEL_PATH="/data/users/yanrongen/AnyEdit/LLM-Qwen2.5-7B"
JSON_CONFIG="Llama3-8B-Instruct.json"
DATASET="mquake"

# ==============================================================================
# ⚙️ 自动生成
# ==============================================================================

L_TAG=$(echo $TARGET_LAYERS | awk '{print "L"$1"-"$NF}')
JOB_NAME="${L_TAG}_${REMA_SRC}_r${REMA_K}_pca${PCA_K}_d${DELTA}"

LOG_DIR="logs"
SCRIPT_DIR="logs/slurm_scripts"
mkdir -p $LOG_DIR $SCRIPT_DIR
SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.sh"

# ==============================================================================
# 🚀 SLURM
# ==============================================================================
cat <<EOT > "$SCRIPT_PATH"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --time=72:00:00

source /usr/local/bin/init-conda.sh
conda activate ${ENV_NAME}
export PYTHONPATH=$(pwd):\$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

echo "🚀 Job Start: ${JOB_NAME}"
echo "Layers: ${TARGET_LAYERS}"
echo "REMA:   Src=${REMA_SRC} | K=${REMA_K}"
echo "MOM2:   Mode=${COV_MODE} | PCA_K=${PCA_K}"
echo "Delta:  ${DELTA}"
echo "------------------------------------------------"

python experiments/evaluate.py \\
    --ds_name ${DATASET} --dataset_size_limit 50 --seed 42 \\
    --alg_name ${ALG_NAME} --model_name ${MODEL_PATH} --hparams_fname ${JSON_CONFIG} \\
    --layers ${TARGET_LAYERS} \\
    --cov_mode ${COV_MODE} --delta ${DELTA} --pca_k ${PCA_K} \\
    --use_rema True --rema_source ${REMA_SRC} --rema_k ${REMA_K} \\
    --use_icsp False --mom2_update_weight ${MOM2_WEIGHT}

echo "✅ Job Finished"
EOT

sbatch "$SCRIPT_PATH"

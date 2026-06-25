#!/bin/bash

# ==============================================================================
# 🎛️ 变量控制台 (CONTROL PANEL)
# ==============================================================================

# --- 1. 实验开关 (控制跑哪些部分) ---
RUN_DEBUG="true"       # ❌ 关闭 Debug (只跑少量数据测试)

# >> 主实验表格 (Table Results)
RUN_GSM8K_SWEEP="false"  # ✅ Exp 1: GSM8K REMA Sweep
RUN_MATH_SWEEP="false"  # ✅ Exp 2: MATH REMA Sweep
RUN_MOM2_SWEEP="false"   # ✅ Exp 3: MOM2 Rank Sweep
RUN_LAYER_SWEEP="false"  # ✅ Exp 4: Layer 6/16/26 Sweep

# >> 新增实验 (New Tasks)
RUN_ABLATION="false"     # ✅ Exp 5: Ablation (RandCov/RandVec), 验证几何形状重要性
RUN_CAPTURE="false"      # ✅ Exp 6: Capture L16 Vectors, 用于画 t-SNE/PCA 主图

# --- 2. 核心参数设置 ---
TARGET_LAYER=16         # 默认层
REMA_TYPE="math"        # 默认类型
REMA_K=64               # 默认秩

# --- 3. 随机种子 ---
# 跑全量建议用三个种子取平均
SEEDS="42" 

# --- 4. 基础配置 ---
ENV_NAME="yanrongen_anyedit"
PYTHON_SCRIPT="ablation.py"
PROBE_LIMIT=200         # 保持 200 以获得稳定结果

# --- 5. 路径配置 ---
MODEL_PATH="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
REMA_DIR="/data/users/yanrongen/ROME-MEMIT/rema_matrices"
MOM2_DIR="/data/users/yanrongen/ROME-MEMIT/mom2_eig"
DATA_ROOT="/data/users/yanrongen/ROME-MEMIT/data"
CACHE_DIR="./matrix_cache" 

# ==============================================================================
# 🛠️ 自动构建命令与任务名
# ==============================================================================

CMD_FLAGS=""
JOB_TAG=""

if [ "$RUN_DEBUG" = "true" ]; then
    CMD_FLAGS+=" --run_debug"
    JOB_TAG+="_DEBUG"
fi

# --- 原有 Sweep ---
if [ "$RUN_GSM8K_SWEEP" = "true" ]; then 
    CMD_FLAGS+=" --run_gsm8k_sweep"
    JOB_TAG+="_G"
fi

if [ "$RUN_MATH_SWEEP" = "true" ]; then 
    CMD_FLAGS+=" --run_math_sweep"
    JOB_TAG+="_M"
fi

if [ "$RUN_MOM2_SWEEP" = "true" ]; then 
    CMD_FLAGS+=" --run_mom2_sweep"
    JOB_TAG+="_Mom"
fi

if [ "$RUN_LAYER_SWEEP" = "true" ]; then 
    CMD_FLAGS+=" --run_layer_sweep"
    JOB_TAG+="_Lay"
fi

# --- 新增功能开关 ---
if [ "$RUN_ABLATION" = "true" ]; then 
    CMD_FLAGS+=" --run_ablation"
    JOB_TAG+="_Abl"  # 给任务名加个标记，方便看 Log
fi

if [ "$RUN_CAPTURE" = "true" ]; then 
    CMD_FLAGS+=" --run_capture"
    JOB_TAG+="_Cap"
fi

# 最终任务名
JOB_NAME="V12_Exp_S42${JOB_TAG}"
LOG_DIR="logs_v12_exp"
mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

# ==============================================================================
# ⚙️ SLURM 提交逻辑
# ==============================================================================

SLURM_SCRIPT="${LOG_DIR}/submit_${JOB_NAME}.sh"

cat <<EOT > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source /usr/local/bin/init-conda.sh
conda activate ${ENV_NAME}
export PYTHONPATH=$(pwd):\$PYTHONPATH

echo "========================================================"
echo "🚀 Starting EXPERIMENT (Custom Config)"
echo "Job Name:    ${JOB_NAME}"
echo "Probe Limit: ${PROBE_LIMIT}"
echo "Seeds:       ${SEEDS}"
echo "Flags:       ${CMD_FLAGS}"
echo "========================================================"

# 运行 Python 脚本
# 注意：--tasks 增加了 mquake，因为 Capture 需要它
python -u ${PYTHON_SCRIPT} \\
    --model_path "${MODEL_PATH}" \\
    --rema_dir "${REMA_DIR}" \\
    --mom2_dir "${MOM2_DIR}" \\
    --data_root "${DATA_ROOT}" \\
    --cache_dir "${CACHE_DIR}" \\
    --layer ${TARGET_LAYER} \\
    --rema_type "${REMA_TYPE}" \\
    --rema_k ${REMA_K} \\
    --mom2_k 128 \\
    --tasks gsm8k math blimp mquake \\
    --probe_limit ${PROBE_LIMIT} \\
    --seeds ${SEEDS} \\
    ${CMD_FLAGS}

echo "--------------------------------------------------------"
echo "✅ Job Finished"
EOT

# 提交作业
sbatch "$SLURM_SCRIPT"
echo "Job submitted: ${JOB_NAME}"
echo "Logs dir:      ${LOG_DIR}"
echo "Check logs:    tail -f ${LOG_DIR}/${JOB_NAME}_*.out"
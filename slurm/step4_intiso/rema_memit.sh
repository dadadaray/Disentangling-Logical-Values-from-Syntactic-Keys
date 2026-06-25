#!/bin/bash

# ================= 🔧 基础环境配置 =================
# 这里的配置会写入每个生成的 slurm 脚本中
ENV_SETUP="source /usr/local/bin/init-conda.sh; conda activate yanrongen_anyedit; export PYTHONPATH=$(pwd):\$PYTHONPATH; export CUDA_VISIBLE_DEVICES=0,1"
MODEL="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
JSON="Llama3-8B-Instruct.json"

# ================= ⚙️ 参数定义 =================
# 层级定义
LAYERS_FRAGILE="9 10 11 12 13"   # 脆弱核心
LAYERS_ROBUST="24 25 26 27 28"   # 鲁棒外壳 (ZSRE)

# 最佳 Delta 参数
DELTA_FRAGILE="0.01"    # L9-13 需要强攻击
DELTA_ROBUST="0.001"    # L24-28 最佳参数 (基于你的分析)

# 确保日志文件夹存在
mkdir -p logs/slurm_scripts

# ================= 🛠️ 核心提交函数 =================
submit_job() {
    local job_name="$1"
    local py_cmd="$2"
    local script_path="logs/slurm_scripts/${job_name}.sh"
    
    # 生成独立的 SLURM 脚本
    cat <<EOT > "$script_path"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --time=24:00:00

echo "🚀 Starting Job: ${job_name}"
echo "📅 Time: \$(date)"
echo "-------------------------------------"

${ENV_SETUP}

${py_cmd}

echo "-------------------------------------"
echo "✅ Job Finished at \$(date)"
EOT

    # 提交给调度器
    echo "📄 Submitting: $job_name"
    sbatch "$script_path"
}

# ==============================================================================
# 📦 实验 A: 基线对比 (MOM2 Baseline)
# ==============================================================================
echo "--- [Part A] Submitting MOM2 Baselines ---"

# 1. MOM2 @ L9-13
submit_job "MOM2_Baseline_L9-13" \
"python experiments/evaluate.py --ds_name mquake --dataset_size_limit 50 --seed 42 \
 --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
 --layers $LAYERS_FRAGILE --cov_mode mom2"

# 2. MOM2 @ L24-28
submit_job "MOM2_Baseline_L24-28" \
"python experiments/evaluate.py --ds_name mquake --dataset_size_limit 50 --seed 42 \
 --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
 --layers $LAYERS_ROBUST --cov_mode mom2"


# ==============================================================================
# 📦 实验 B: L9-13 脆弱性验证 (The Fragile Core)
# 目的: 证明即便上了最强配置 (Delta 0.01 + REMA 1024)，前层依然不可编辑。
# ==============================================================================
echo "--- [Part B] Submitting Fragile Core (L9-13) Check ---"

submit_job "TRIM_Max_L9-13_k1024_d${DELTA_FRAGILE}" \
"python experiments/evaluate.py --ds_name mquake --dataset_size_limit 50 --seed 42 \
 --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
 --layers $LAYERS_FRAGILE --cov_mode deltaI --delta $DELTA_FRAGILE \
 --use_rema True --rema_source lite --rema_k 1024 --damping 1.0 \
 --use_icsp True"


# ==============================================================================
# 📦 实验 C: L24-28 全量对比 (Full Method Comparison)
# 变量: Source (lite vs gsm8k) x Rank (128 vs 1024)
# ==============================================================================
echo "--- [Part C] Submitting Robust Layer (L24-28) Full Method ---"

for SRC in lite gsm8k; do
    for K in 128 1024; do
        TASK_NAME="Full_TRIM_L24-28_${SRC}_k${K}_d${DELTA_ROBUST}"
        
        CMD="python experiments/evaluate.py \
            --ds_name mquake --dataset_size_limit 50 --seed 42 \
            --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
            --layers $LAYERS_ROBUST --cov_mode deltaI --delta $DELTA_ROBUST \
            --use_rema True --rema_source $SRC --rema_k $K --damping 1.0 \
            --use_icsp True"
        
        submit_job "$TASK_NAME" "$CMD"
    done
done


# ==============================================================================
# 📦 实验 D: 消融实验 (Ablation Studies @ L24-28)
# ==============================================================================
echo "--- [Part D] Submitting Ablations (L24-28) ---"

# 1. ICSP Only (无 REMA)
submit_job "Ablation_ICSP_Only_L24-28_d${DELTA_ROBUST}" \
"python experiments/evaluate.py --ds_name mquake --dataset_size_limit 50 --seed 42 \
 --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
 --layers $LAYERS_ROBUST --cov_mode deltaI --delta $DELTA_ROBUST \
 --use_rema False \
 --use_icsp True"

# 2. REMA Only (无 ICSP) - 同样遍历 Source 和 K
for SRC in lite gsm8k; do
    for K in 128 1024; do
        TASK_NAME="Ablation_REMA_Only_L24-28_${SRC}_k${K}_d${DELTA_ROBUST}"
        
        CMD="python experiments/evaluate.py \
            --ds_name mquake --dataset_size_limit 50 --seed 42 \
            --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
            --layers $LAYERS_ROBUST --cov_mode deltaI --delta $DELTA_ROBUST \
            --use_rema True --rema_source $SRC --rema_k $K --damping 1.0 \
            --use_icsp False"
        
        submit_job "$TASK_NAME" "$CMD"
    done
done


# ==============================================================================
# 📦 实验 E: 种子验证 (Seeds)
# 目的: 选取最优配置 (Full + Lite + k1024) 跑不同随机种子
# ==============================================================================
echo "--- [Part E] Submitting Seed Verification ---"

for SEED in 123 2024; do
    submit_job "TRIM_Seed${SEED}_L24-28_d${DELTA_ROBUST}" \
    "python experiments/evaluate.py --ds_name mquake --dataset_size_limit 50 --seed $SEED \
     --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \
     --layers $LAYERS_ROBUST --cov_mode deltaI --delta $DELTA_ROBUST \
     --use_rema True --rema_source lite --rema_k 1024 --damping 1.0 \
     --use_icsp True"
done

echo "🎉 All jobs have been generated and submitted to SLURM!"
echo "👉 Check queue: squeue -u \$(whoami)"
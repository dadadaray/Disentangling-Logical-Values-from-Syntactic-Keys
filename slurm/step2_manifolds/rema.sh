#!/bin/bash

# ================= 🔧 基础配置 =================
ENV_SETUP="source /usr/local/bin/init-conda.sh; conda activate yanrongen_anyedit; export PYTHONPATH=$(pwd):\$PYTHONPATH; export CUDA_VISIBLE_DEVICES=0,1"
MODEL="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
JSON="Llama3-8B-Instruct.json"
LOG_DIR="logs"
SCRIPT_DIR="logs/slurm_scripts"

# 确保文件夹存在
mkdir -p $LOG_DIR
mkdir -p $SCRIPT_DIR

# 存储任务ID和相关信息的关联数组
declare -A JOB_INFO_MAP

# ================= 🛠️ 核心提交函数 =================
submit_job() {
    local job_name="$1"
    local layers="$2"
    local delta="$3"
    local src="$4"
    local k="$5"
    local icsp="$6"
    local dataset="$7" 
    
    if [ -z "$dataset" ]; then dataset="mquake"; fi

    local script_path="${SCRIPT_DIR}/${job_name}.sh"
    
    # 生成 SLURM 脚本
    cat <<EOT > "$script_path"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --time=12:00:00

echo "🚀 Starting Job: ${job_name} (ID: \$SLURM_JOB_ID)"
${ENV_SETUP}

python experiments/evaluate.py \\
    --ds_name $dataset --dataset_size_limit 50 --seed 2024 \\
    --alg_name MEMIT --model_name $MODEL --hparams_fname $JSON \\
    --layers $layers --cov_mode deltaI --delta $delta \\
    --use_rema True --rema_source $src --rema_k $k \\
    --use_icsp $icsp --mom2_update_weight 15000

echo "✅ Job Finished at \$(date)"
EOT

    # 提交任务并捕获 Job ID
    local submit_out=$(sbatch "$script_path")
    local job_id=$(echo "$submit_out" | awk '{print $4}')
    
    echo "📄 Submitted: $job_name (ID: $job_id)"
    
    # 记录任务信息用于监控: "ScriptPath|JobName"
    JOB_INFO_MAP[$job_id]="${script_path}|${job_name}"
}

# ================= 🕵️ 监控守护函数 (核心逻辑) =================
monitor_jobs() {
    echo "🕵️ Starting Monitor Watchdog..."
    echo "   Timeout threshold: 3 minutes"
    echo "   Check interval: 60 seconds"
    echo "---------------------------------------------------"

    while true; do
        # 获取当前所有正在运行或排队的任务ID
        current_jobs=$(squeue -u $(whoami) -h -o "%i")
        
        # 如果没有任务了，退出监控
        if [ -z "$current_jobs" ]; then
            echo "🎉 All jobs finished or cleared. Monitor exiting."
            break
        fi

        # 遍历我们需要监控的任务
        for job_id in "${!JOB_INFO_MAP[@]}"; do
            # 检查任务是否还在队列中
            if echo "$current_jobs" | grep -q "$job_id"; then
                
                info="${JOB_INFO_MAP[$job_id]}"
                script_path="${info%|*}"
                job_name="${info#*|}"
                log_file="${LOG_DIR}/${job_name}_${job_id}.out"

                # 1. 检查日志文件是否存在
                if [ -f "$log_file" ]; then
                    # 获取文件最后修改时间戳
                    last_mod=$(stat -c %Y "$log_file")
                    current_time=$(date +%s)
                    diff=$((current_time - last_mod))

                    # 阈值：3分钟 = 180秒
                    if [ $diff -gt 180 ]; then
                        echo "⚠️  [TIMEOUT] Job $job_id ($job_name) hang detected!"
                        echo "    Log untouched for $diff seconds. Cancelling..."
                        
                        # A. 取消卡住的任务
                        scancel "$job_id"
                        
                        # B. 从 Map 中移除旧 ID
                        unset JOB_INFO_MAP[$job_id]
                        
                        # C. 重新提交
                        echo "🔄 Resubmitting $job_name..."
                        # 这里直接调用 sbatch 因为 .sh 文件已经存在
                        new_submit_out=$(sbatch "$script_path")
                        new_job_id=$(echo "$new_submit_out" | awk '{print $4}')
                        
                        echo "✅ Resubmitted as ID: $new_job_id"
                        
                        # D. 更新 Map 监控新 ID
                        JOB_INFO_MAP[$new_job_id]="${script_path}|${job_name}"
                    fi
                else
                    # 日志文件还没生成（可能还在 Pending 或 Configuring），跳过检查
                    : 
                fi
            else
                # 任务已结束（成功或失败），从监控列表中移除
                unset JOB_INFO_MAP[$job_id]
            fi
        done

        # 等待 60 秒再轮询
        sleep 60
    done
}

# ==============================================================================
# 🎯 提交逻辑
# ==============================================================================
echo "--- 1. Submitting Jobs ---"

# 核心参数配置
DELTA_TARGET="0.001"
ICSP_SETTING="False" 
DATASET_TARGET="mquake"

declare -A LAYER_MAP
LAYER_MAP["L4-8"]="4 5 6 7 8" 
LAYER_MAP["L12-16"]="12 13 14 15 16" 
LAYER_MAP["L24-28"]="24 25 26 27 28" 

REMA_SOURCES="lite gsm8k"
REMA_KS="32 128"
#REMA_SOURCES="gsm8k"
#REMA_KS="32"

# 循环提交
for LAYER_GROUP in "${!LAYER_MAP[@]}"; do
    LAYERS=${LAYER_MAP[$LAYER_GROUP]}
    for SRC in $REMA_SOURCES; do
        for K in $REMA_KS; do
            JOB_NAME="PreserveREMA_${LAYER_GROUP}_${SRC//./-}_k${K}_d${DELTA_TARGET//./-}"
            submit_job "$JOB_NAME" "$LAYERS" "$DELTA_TARGET" "$SRC" "$K" "$ICSP_SETTING" "$DATASET_TARGET"
        done
    done
done

echo "🎉 All initial jobs submitted."

# ==============================================================================
# 🚀 启动后台监控
# ==============================================================================
# 将 monitor_jobs 放入后台运行，不阻塞当前终端，但会持续监控
monitor_jobs &
WATCHDOG_PID=$!

echo "🛡️  Watchdog is running in background (PID: $WATCHDOG_PID)."
echo "    You can close this terminal, monitoring will continue until all jobs are done."
echo "    To kill monitor manually: kill $WATCHDOG_PID"
echo "👉 Check queue: squeue -u \$(whoami)"
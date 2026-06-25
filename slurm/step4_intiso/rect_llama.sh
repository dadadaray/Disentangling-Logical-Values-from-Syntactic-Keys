#!/bin/bash
#SBATCH --job-name=Rect-Llama
#SBATCH --output=logs/Rect_Llama_%j.out
#SBATCH --error=logs/Rect_Llama_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --time=72:00:00

PROJECT_ROOT="/data/users/yanrongen/ROME-MEMIT"
EVAL_SCRIPT_PATH="experiments/evaluate.py"

echo "========== R-AnyEdit 单进程实验启动 =========="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURMD_NODENAME"
echo "分配 GPU 数量: 2"
echo "开始时间: $(date)"
echo "项目目录: /data/users/yanrongen/ROME-MEMIT"
echo ""

# 创建日志目录
mkdir -p logs

echo "========== 环境初始化 =========="
source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python)"
echo ""

echo "========== 资源配置检查 =========="
echo "SLURM分配的GPU: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo ""

echo "========== 项目环境检查 =========="
cd $PROJECT_ROOT
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "工作目录: $(pwd)"
echo ""

# 检查GPU可用性
python -c "
import torch
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('警告: 未检测到可用的GPU')
"

echo "========== 启动 R-AnyEdit (MEMIT_RECT) 实验 =========="

LLAMA_MODEL_PATH="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"

echo "实验配置:"
echo "- 算法: MEMIT"
echo "- 模型: $LLAMA_MODEL_PATH"
echo "- 超参数文件: Llama3-8B-Instruct.json"
echo "- 数据集: zsre"
echo "- 编辑数量: 50"
echo "- 数据限制: 1"
echo ""

# 不设置CUDA_VISIBLE_DEVICES，让SLURM自动管理
python $EVAL_SCRIPT_PATH \
    --alg_name "MEMIT" \
    --model_name "$LLAMA_MODEL_PATH" \
    --hparams_fname "Llama3-8B-Instruct.json" \
    --ds_name "zsre" \
    --dataset_size_limit 50 \
    --num_edits 1 \
    --seed 2024 \

EXIT_CODE=$?
echo ""
echo "========== 实验完成 =========="
echo "结束时间: $(date)"
echo "退出代码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 实验执行成功"
else
    echo "❌ 实验执行失败，请检查错误日志"
fi

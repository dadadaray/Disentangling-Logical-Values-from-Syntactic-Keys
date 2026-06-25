#!/bin/bash
#SBATCH --job-name=gen_noise_manifold
#SBATCH --output=logs/noise_%j.out
#SBATCH --error=logs/noise_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # 1张卡跑推理足够了
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# 1. 环境初始化
source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit

# 2. 关键：将当前目录加入 PYTHONPATH，否则 'from util import nethook' 会报错
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建日志文件夹
mkdir -p logs rema_matrices

echo "====================================================="
echo "🚀 Job Started: Generating Nonsense Manifold Baseline"
echo "====================================================="

# 运行 Python 脚本
python rema_overlap.py

echo "====================================================="
echo "✅ Job Finished."
echo "====================================================="
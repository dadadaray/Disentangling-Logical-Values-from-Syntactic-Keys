#!/bin/bash
#SBATCH --job-name=rema_dim_stats
#SBATCH --output=logs/dims_%j.out
#SBATCH --error=logs/dims_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:0             # 不需要GPU，纯CPU计算即可

# 1. 环境初始化
source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit

# 2. 关键：设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建日志文件夹
mkdir -p logs

echo "====================================================="
echo "🚀 Job Started: Collecting REMA Dimensionality Stats"
echo "Target: Calculating 95% Energy Thresholds for Scree Plot"
echo "====================================================="

# 运行 Python 脚本
python rema_eigen.py

echo "====================================================="
echo "✅ Job Finished."
echo "Output saved to: rema_dim_stats.pt"
echo "Now you can download this file to plot the Scree Plot."
echo "====================================================="
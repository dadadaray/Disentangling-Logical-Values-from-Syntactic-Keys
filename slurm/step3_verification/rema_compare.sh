#!/bin/bash
#SBATCH --job-name=rema_radar_data
#SBATCH --output=logs/radar_%j.out
#SBATCH --error=logs/radar_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# 1. 环境初始化
source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit

# 2. 设置路径
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建日志文件夹
mkdir -p logs

echo "====================================================="
echo "🚀 Job Started: REMA Manifold Comparison (Radar Data)"
echo "Target Layers: 12-16"
echo "====================================================="

# 运行 Python 脚本
python rema_compare.py

echo "====================================================="
echo "✅ Job Finished."
echo "Output saved to: radar_chart_data_full.pt"
echo "Now you can run the plotting script locally or on cpu."
echo "====================================================="
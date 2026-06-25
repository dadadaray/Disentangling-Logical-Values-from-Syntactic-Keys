#!/bin/bash
#SBATCH --job-name=orth_test
#SBATCH --output=logs/orth_%j.out
#SBATCH --error=logs/orth_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

source /usr/local/bin/init-conda.sh
conda activate yanrongen_anyedit
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p logs

echo "Starting Orthogonality Stress Test..."
python few-shot.py

echo "Done."
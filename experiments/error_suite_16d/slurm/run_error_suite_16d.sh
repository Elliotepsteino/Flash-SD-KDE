#!/bin/bash
#SBATCH --job-name=err_suite_a100_16d
#SBATCH --output=logs/error_suite_a100_16d_%j.out
#SBATCH --error=logs/error_suite_a100_16d_%j.err
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

CONFIG=${1:-configs/error_suite_a100_16d/grid_pareto_16d.yaml}

mkdir -p logs

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m experiments.error_suite_a100_16d.sweep --config "$CONFIG"

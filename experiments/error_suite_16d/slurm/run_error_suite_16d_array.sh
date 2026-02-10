#!/bin/bash
#SBATCH --job-name=err_suite_a100_16d_array
#SBATCH --output=logs/error_suite_a100_16d_array_%A_%a.out
#SBATCH --error=logs/error_suite_a100_16d_array_%A_%a.err
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-0

set -euo pipefail

CONFIG_LIST=${1:-configs/error_suite_a100_16d/grid_pareto_16d.yaml}

mkdir -p logs

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m experiments.error_suite_a100_16d.sweep --config "$CONFIG_LIST"

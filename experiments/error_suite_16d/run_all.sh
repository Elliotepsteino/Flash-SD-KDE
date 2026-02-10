#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
base_out_dir="${RUN_ALL_OUT_DIR:-file_storage/error_suite_16d/all_${timestamp}}"
mkdir -p "${base_out_dir}"
echo "Base output directory: ${base_out_dir}"

configs=(
  "configs/error_suite_16d/grid_oracle_mog_16d.yaml"
  "configs/error_suite_16d/grid_correctness_16d.yaml"
  "configs/error_suite_16d/grid_precision_16d.yaml"
  "configs/error_suite_16d/grid_statistical_vs_n_16d.yaml"
  "configs/error_suite_16d/grid_bandwidth_curve_16d.yaml"
  "configs/error_suite_16d/grid_pareto_16d.yaml"
  "configs/error_suite_16d/failure_modes_16d.yaml"
)

for cfg in "${configs[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  out_dir="${base_out_dir}/${name}"
  echo "Running sweep: ${cfg}"
  python -m experiments.error_suite_16d.sweep --config "${cfg}" --out_dir "${out_dir}"
done

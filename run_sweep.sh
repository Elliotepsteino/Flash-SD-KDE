#!/usr/bin/env bash
# Sweep benchmark_triton_kde.py over powers of two n_train values (512 → 32,768)
# while keeping n_test = n_train / 8 and logging output to a file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${1:-sweep.log}"
SEEDS="${SEEDS:-0,1,2}"
MIXTURE="${MIXTURE:-0}"
DEVICE="${DEVICE:-cuda}"

echo "Writing sweep output to ${LOG_PATH}"
: > "${LOG_PATH}"

n=512
while [ "${n}" -le 32768 ]; do
  m=$(( n / 8 ))
  if [ "${m}" -lt 1 ]; then
    m=1
  fi
  echo "==== n_train=${n}, n_test=${m} ====" | tee -a "${LOG_PATH}"
  python "${SCRIPT_DIR}/benchmark_triton_kde.py" \
    --seeds "${SEEDS}" \
    --n-train "${n}" \
    --n-test "${m}" \
    --mixture-index "${MIXTURE}" \
    --device "${DEVICE}" | tee -a "${LOG_PATH}"
  echo | tee -a "${LOG_PATH}"
  n=$(( n * 2 ))
done

echo "Sweep complete. Log saved to ${LOG_PATH}"

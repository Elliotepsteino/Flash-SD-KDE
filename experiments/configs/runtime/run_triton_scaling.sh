#!/usr/bin/env bash
# Sweep Triton empirical SD-KDE kernel over large powers of two (up to ~4M).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${1:-triton_scaling.log}"
SEED="${SEED:-0}"
MIXTURE="${MIXTURE:-0}"
DEVICE="${DEVICE:-cuda}"
START_POWER="${START_POWER:-15}"  # 2^15 = 32,768
END_POWER="${END_POWER:-22}"      # 2^22 = 4,194,304

echo "Writing Triton scaling sweep to ${LOG_PATH}"
: > "${LOG_PATH}"

for ((p = START_POWER; p <= END_POWER; ++p)); do
  n=$((1 << p))
  m=$(( n / 8 ))
  if [ "${m}" -lt 1 ]; then
    m=1
  fi
  echo "==== Triton only: n_train=${n}, n_test=${m} ====" | tee -a "${LOG_PATH}"
  python "${SCRIPT_DIR}/benchmark_triton_kde.py" \
    --seeds "${SEED}" \
    --n-train "${n}" \
    --n-test "${m}" \
    --mixture-index "${MIXTURE}" \
    --device "${DEVICE}" \
    --emp-kernel-only | tee -a "${LOG_PATH}"
  echo | tee -a "${LOG_PATH}"
done

echo "Large-n Triton sweep complete. Log saved to ${LOG_PATH}"

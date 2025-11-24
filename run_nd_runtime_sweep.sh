#!/usr/bin/env bash
# Collect runtime data for 16-D KDE/SD-KDE (sklearn, Torch, Triton) up to 32k.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${1:-nd_runtime.log}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
START_POWER="${START_POWER:-12}"  # 2^12 = 4096
END_POWER="${END_POWER:-15}"      # 2^15 = 32768

echo "Writing 16-D runtime sweep to ${LOG_PATH}"
: > "${LOG_PATH}"

for ((p = START_POWER; p <= END_POWER; ++p)); do
  n=$((1 << p))
  m=$(( n / 8 ))
  if [ "${m}" -lt 1 ]; then
    m=1
  fi

  echo "==== 16D KDE Runtime: n_train=${n}, n_test=${m} ====" | tee -a "${LOG_PATH}"
  python "${SCRIPT_DIR}/benchmark_triton_kde.py" \
    --multi-d \
    --seeds "${SEED}" \
    --n-train "${n}" \
    --n-test "${m}" \
    --device "${DEVICE}" | tee -a "${LOG_PATH}"

  echo "==== 16D SD-KDE Runtime: n_train=${n}, n_test=${m} ====" | tee -a "${LOG_PATH}"
  python "${SCRIPT_DIR}/benchmark_triton_kde.py" \
    --multi-d-sd \
    --seeds "${SEED}" \
    --n-train "${n}" \
    --n-test "${m}" \
    --device "${DEVICE}" | tee -a "${LOG_PATH}"

  echo | tee -a "${LOG_PATH}"
done

echo "Runtime sweep complete. Log saved to ${LOG_PATH}"

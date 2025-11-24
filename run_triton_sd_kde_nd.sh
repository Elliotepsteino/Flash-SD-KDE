#!/usr/bin/env bash
# Run the 16-D SD-KDE Triton benchmark across a range of sizes (single seed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${1:-triton_sd_kde_nd.log}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
START_POWER="${START_POWER:-12}"  # 2^12 = 4096
END_POWER="${END_POWER:-18}"      # 2^18 = 262,144

echo "Writing 16-D SD-KDE sweep to ${LOG_PATH}"
: > "${LOG_PATH}"

for ((p = START_POWER; p <= END_POWER; ++p)); do
  n=$((1 << p))
  m=$(( n / 8 ))
  if [ "${m}" -lt 1 ]; then
    m=1
  fi
  echo "==== 16D SD-KDE: n_train=${n}, n_test=${m} ====" | tee -a "${LOG_PATH}"
  python "${SCRIPT_DIR}/benchmark_triton_kde.py" \
    --multi-d-sd \
    --sd-nd-triton-only \
    --seeds "${SEED}" \
    --n-train "${n}" \
    --n-test "${m}" \
    --device "${DEVICE}" | tee -a "${LOG_PATH}"
  echo | tee -a "${LOG_PATH}"
done

echo "16-D SD-KDE sweep complete. Log saved to ${LOG_PATH}"

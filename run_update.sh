#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
LOG_DIR="/var/log/tradebot"
LOCK_FILE="/tmp/tradebot.lock"

mkdir -p "${LOG_DIR}"

# Activate venv if needed
if [[ -f "${ROOT_DIR}/venv/bin/activate" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

# UTC enforced by cron (TZ=UTC)
(
  flock -n 9 || exit 0
  python3 "${ROOT_DIR}/update_and_infer.py" >> "${LOG_DIR}/tradebot.log" 2>&1
) 9>"${LOCK_FILE}"

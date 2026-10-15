#!/usr/bin/env bash
# Stop the server ./start.sh started and remove its container.
# Waits up to STOP_LOCK_WAIT seconds (default 30) if another lifecycle command holds the lock,
# then exits 1 without stopping anything. The server gets STOP_TIMEOUT seconds (default 30) to shut down.
# Usage: ./stop.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

mkdir -p "${STATE_DIR}"
exec 8>"${STATE_DIR}/lifecycle.lock"
if ! flock -w "${STOP_LOCK_WAIT}" 8; then
  die "another lifecycle command still holds the lock after ${STOP_LOCK_WAIT}s; nothing was stopped"
fi
stop_container

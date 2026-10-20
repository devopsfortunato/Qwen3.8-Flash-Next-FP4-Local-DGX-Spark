#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local with vLLM on one DGX Spark.
# Runs scripts/prepare.sh when the image or the checkpoint is not ready, starts
# eugr/spark-vllm-b12x on port 8000, waits until /v1/models answers, then runs a smoke test.
# Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart|stop] [extra vllm serve args]
#   ./start.sh                  # model-card Spark flags, port 8000
#   ./start.sh restart          # stop, then start; checks run before the stop
#   ./start.sh stop             # same as ./stop.sh
#   DRY_RUN=1 ./start.sh        # print the docker command and change nothing
#   PORT=9000 ./start.sh        # environment wins over .env
#   PROFILE=1m ./start.sh       # opt-in profile from profiles/ (kv-max, 1m); .env can set PROFILE too
# Extra arguments are appended to `vllm serve` after the defaults.
# Settings live in scripts/config.sh (environment, then ./.env, then the defaults).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  stop) exec ./stop.sh ;;
  help|-h|--help) usage; exit 0 ;;
esac
for arg in "$@"; do
  [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { usage; exit 0; }
done

validate_config
DRY=0
[[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${VLLM_CACHE}" "${FLASHINFER_CACHE}" "${TRITON_CACHE}" "${B12X_CACHE}"
exec 8>"${STATE_DIR}/lifecycle.lock"
flock -n 8 || die "another ./start.sh or ./stop.sh is running; wait for it to finish"

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
printf '\n%sQwen3.8 Flash Next FP4 Local%s\n' "${M}" "${R}"
printf '%s %s · %s seqs · %s-token window · %s KV · MTP %s · PLE %s · port %s%s\n' \
  "${D}" "${MODEL_ID}" "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" "${KV_CACHE_DTYPE}" \
  "${MTP_TOKENS}" "${VLLM_PLE_TABLE_MEMORY}" "${PORT}" "${R}"
PROFILE_LINE="default"
if [[ -n "${PROFILE}" ]]; then
  PROFILE_LINE="${PROFILE} (profiles/${PROFILE}.env)"
fi
KV_LINE="GPU_MEMORY_UTILIZATION ${GPU_MEMORY_UTILIZATION}"
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  KV_LINE="${KV_CACHE_MEMORY_BYTES} bytes (KV_CACHE_MEMORY_BYTES)"
fi
if (( ${YARN_FACTOR:-1} > 1 )); then
  KV_LINE="${KV_LINE} · YaRN ${YARN_FACTOR}x"
fi
printf '%s profile %s · KV pool %s%s\n\n' "${D}" "${PROFILE_LINE}" "${KV_LINE}" "${R}"

api_base() {
  local host="${HOST}"
  if [[ "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    host="127.0.0.1"
  fi
  if [[ "${host}" == *:* && "${host}" != \[* ]]; then
    host="[${host}]"
  fi
  printf 'http://%s:%s' "${host}" "${PORT}"
}
URL="$(api_base)"

running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)" == true ]]
}

port_listening() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN
    return $?
  fi
  return 1
}

if (( DRY == 0 )) && [[ "${MODE}" == start ]] && command -v docker >/dev/null 2>&1 && running; then
  log "${CONTAINER_NAME} is already running on port ${PORT}."
  log "Use ./start.sh restart to restart it, or ./stop.sh to stop it."
  exit 0
fi

step() { printf '%s[%s/5]%s %s%s%s\n' "${M}" "$1" "${R}" "${B}" "$2" "${R}"; }

step 1 "Setup: image and checkpoint"
ready_now="$(prepared_state 2>/dev/null || true)"
ready_was="$(cat "${PREPARED_MARKER}" 2>/dev/null || true)"
if [[ "${PREPARE}" == 1 || ( "${PREPARE}" != 0 && "${ready_now}" != "${ready_was}" ) ]]; then
  if (( DRY )); then
    log "DRY_RUN: scripts/prepare.sh would run now"
  else
    log "Not ready yet: running scripts/prepare.sh"
    ./scripts/prepare.sh
  fi
else
  log "Ready: ${IMAGE} and ${MODEL_ID}"
fi

if (( DRY == 0 )); then
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} is missing (scripts/prepare.sh pulls it)"
  snapshot_dir >/dev/null 2>&1 || die "${MODEL_ID} is not in ${HF_CACHE} (scripts/prepare.sh downloads it)"
else
  command -v docker >/dev/null 2>&1 || warn "DRY_RUN: docker is not installed"
fi

step 2 "Checks: previous server, port, memory"
if (( DRY )); then
  if command -v docker >/dev/null 2>&1 && running; then
    if [[ "${MODE}" == restart ]]; then
      log "DRY_RUN: ${CONTAINER_NAME} is running; a real restart would stop it first"
    else
      log "DRY_RUN: ${CONTAINER_NAME} is running; a real start would leave it alone"
    fi
  fi
elif [[ "${MODE}" == restart ]] && running; then
  stop_container
fi

if port_listening; then
  msg="port ${PORT} is already in use"
  if command -v ss >/dev/null 2>&1; then
    msg="${msg}: $(ss -ltnp "sport = :${PORT}" 2>/dev/null | tail -n +2 | tr '\n' ' ' || true)"
  fi
  if (( DRY )); then warn "DRY_RUN: ${msg}"; else die "${msg}"; fi
else
  log "Port ${PORT} is free"
fi

if avail="$(mem_available_gib)"; then
  if (( avail < MEM_NEED_GIB )); then
    msg="only ${avail} GiB MemAvailable; this recipe asks for ~${MEM_NEED_GIB} GiB before launch (MEM_NEED_GIB)"
    if (( DRY )) || [[ "${MEM_CHECK}" == 0 ]]; then
      warn "${msg}"
    else
      die "${msg} (MEM_CHECK=0 starts anyway)"
    fi
  else
    log "${avail} GiB MemAvailable (launch asks for ~${MEM_NEED_GIB} GiB)"
  fi
else
  warn "MemAvailable could not be read"
fi

if (( DRY == 0 )); then
  patch_check
else
  wanted_patches
  if (( PLE_PATCH_ON || DV_PATCH_ON )); then
    log "DRY_RUN: a real start checks that ${IMAGE} has the expected files before mounting the patches"
  fi
fi
if [[ "${PLE_PATCH_ON}" == 1 ]]; then
  log "PLE queue depth 256: ${PLE_PATCH_FILE##*/} mounted read-only"
fi
if [[ "${DV_PATCH_ON}" == 1 ]]; then
  log "Draft vocabulary latin: ${DV_PATCH_FILE##*/} and ${DV_IDS_FILE##*/} mounted read-only"
fi
DRAFT_LINE="full"
if [[ "${DV_PATCH_ON}" == 1 ]]; then
  DRAFT_LINE="latin (147,840 of 248,320 tokens)"
fi

build_docker_cmd "$@"
step 3 "Launch: container ${CONTAINER_NAME}"
log "docker run ${IMAGE} vllm serve ${MODEL_ID} --port ${PORT}"
if (( DRY )); then
  printf '[dry-run]\n%s\n' "$(print_docker_cmd)"
  exit 0
fi

"${DOCKER_CMD[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap 'stop_container; exit 130' INT TERM
  docker logs -f "${CONTAINER_NAME}" || true
  exit "$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo 1)"
fi

step 4 "Loading: waiting for /v1/models"
fail() {
  printf '\n%s── last server log lines ──%s\n' "${D}" "${R}"
  docker logs --tail 40 "${CONTAINER_NAME}" 2>&1 | sed 's/^/ │ /' || true
  die "$1"
}
start_s="${SECONDS}"
next_beat=15
served=""
low_since=""
while (( SECONDS - start_s < WAIT_TIMEOUT )); do
  if curl -sf --max-time 5 "${URL}/v1/models" -o "${STATE_DIR}/models.json"; then
    if served="$(python3 -c 'import json,sys
r=json.load(open(sys.argv[1]))
ids=[m.get("id","") for m in r.get("data",[]) if m.get("id")]
if not ids:
    sys.exit(1)
print(ids[0])' "${STATE_DIR}/models.json" 2>/dev/null)"; then
      break
    fi
  fi
  if ! running; then
    code="$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo unknown)"
    fail "the server exited (code ${code}) before /v1/models answered"
  fi
  if floor="$(mem_available_gib)"; then
    if (( floor < MEM_HARD_FLOOR_GIB )); then
      docker stop -t 5 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
      fail "only ${floor} GiB memory left while loading (MEM_HARD_FLOOR_GIB ${MEM_HARD_FLOOR_GIB}): stopped"
    elif (( floor < MEM_FLOOR_GIB )); then
      low_since="${low_since:-${SECONDS}}"
      if (( SECONDS - low_since >= MEM_FLOOR_GRACE )); then
        docker stop -t 5 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        fail "under ${MEM_FLOOR_GIB} GiB memory for ${MEM_FLOOR_GRACE}s while loading (${floor} GiB now, MEM_FLOOR_GIB): stopped"
      fi
    else
      low_since=""
    fi
  fi
  if (( SECONDS - start_s >= next_beat )); then
    free_now="$(mem_available_gib || echo "?")"
    printf ' %s⋯ %ss elapsed, %s GiB MemAvailable%s\n' "${D}" "$((SECONDS - start_s))" "${free_now}" "${R}"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
if [[ -z "${served}" ]]; then
  fail "not ready after ${WAIT_TIMEOUT}s; it is still running: docker logs -f ${CONTAINER_NAME}"
fi
LOADED_S=$((SECONDS - start_s))
log "Server answered /v1/models after ${LOADED_S}s (${served})"
KV_TOKENS="$(docker logs "${CONTAINER_NAME}" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -n 1 | grep -oE '[0-9,]+' || true)"

step 5 "Smoke test: one chat completion"
body="$(SERVED="${served}" python3 -c 'import json,os
print(json.dumps({
  "model": os.environ["SERVED"],
  "max_tokens": 64,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": False},
  "messages": [{"role": "user", "content": "Reply with OK."}],
}))')"
if curl -s --max-time 180 "${URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "${body}" -o "${STATE_DIR}/smoke.json"; then
  if smoke="$(python3 -c 'import json,sys
r=json.load(open(sys.argv[1]))
msg=(r.get("choices") or [{}])[0].get("message") or {}
text=(msg.get("content") or "") + (msg.get("reasoning_content") or "")
if not text.strip():
    sys.exit(1)
usage=r.get("usage") or {}
print(repr(text.strip()[:40]), usage.get("completion_tokens", "?"), "tokens")' "${STATE_DIR}/smoke.json" 2>/dev/null)"; then
    log "OK: ${smoke}"
  else
    fail "the smoke test returned no text; the server is still running"
  fi
else
  fail "the smoke test request failed; the server is still running"
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
[[ -n "${IP}" ]] || IP="127.0.0.1"
if [[ "${HOST}" != "0.0.0.0" && "${HOST}" != "::" ]]; then
  IP="${HOST}"
fi
printf '\n%s ✔ %s is now LIVE! on port %s%s\n\n' "${G}" "${served}" "${PORT}" "${R}"
cat <<EOF
  OpenAI API   http://${IP}:${PORT}/v1
  Model        ${served}
  Checkpoint   ${MODEL_ID}
  Image        ${IMAGE}
  Window       ${MAX_MODEL_LEN} tokens · ${MAX_NUM_SEQS} at once · ${KV_CACHE_DTYPE} KV · MTP ${MTP_TOKENS}
  Draft vocab  ${DRAFT_LINE}
  Profile      ${PROFILE_LINE}
  KV cache     ${KV_TOKENS:-see docker logs} tokens · pool ${KV_LINE}
  PLE table    ${VLLM_PLE_TABLE_MEMORY} (${B12X_DISK_BACKEND})
  Loaded in    ${LOADED_S}s
  Logs         docker logs -f ${CONTAINER_NAME}
  Restart      ./start.sh restart
  Stop         ./stop.sh

  curl -s http://${IP}:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"${served}","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":1000}'
EOF

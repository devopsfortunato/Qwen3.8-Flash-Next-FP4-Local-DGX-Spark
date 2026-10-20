#!/usr/bin/env bash
# Prepare this machine to serve nvidia/Qwen3.8-Flash-Next-FP4-Local:
#   1. preflight: docker, disk, a low-memory floor
#   2. the image: the recipe image is built from docker/ on top of BASE_IMAGE (pulled by digest when it is
#      not local); any other IMAGE is pulled
#   3. the checkpoint in HF_CACHE (~96 GiB; NVIDIA model card)
#   4. a check that the snapshot is Qwen4-Exp modelopt MIXED_PRECISION and its shards are present
# ./start.sh runs this when the prepared marker does not match. Safe to re-run.
# PULL=0 never pulls: the recipe image then needs BASE_IMAGE local, any other IMAGE needs to be local.
# DRY_RUN=1 prints the plan and does not pull or download.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

for arg in "$@"; do
  case "${arg}" in
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: ${arg}" ;;
  esac
done

validate_config
mkdir -p "${HF_CACHE}/hub" "${VLLM_CACHE}" "${FLASHINFER_CACHE}" "${TRITON_CACHE}" "${B12X_CACHE}" "${STATE_DIR}" "${LOG_DIR}"
exec 9>"${STATE_DIR}/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it to finish"

DRY=0
[[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

free_gb() {
  local n
  n="$(df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
  printf '%s' "${n:-0}"
}

log "Preflight checks"
if avail="$(mem_available_gib)"; then
  if (( avail < MEM_FLOOR_GIB )); then
    msg="only ${avail} GiB MemAvailable (MEM_FLOOR_GIB=${MEM_FLOOR_GIB})"
    if (( DRY )); then
      warn "DRY_RUN: ${msg}"
    elif [[ "${MEM_CHECK}" == 0 ]]; then
      warn "${msg}"
    else
      die "${msg} (MEM_CHECK=0 continues anyway)"
    fi
  else
    log "Memory: ${avail} GiB MemAvailable (floor ${MEM_FLOOR_GIB})"
  fi
else
  warn "MemAvailable could not be read"
fi

if ! command -v docker >/dev/null 2>&1; then
  if (( DRY )); then warn "DRY_RUN: docker is not installed"; else die "docker is not installed"; fi
elif ! docker info >/dev/null 2>&1; then
  if (( DRY )); then warn "DRY_RUN: cannot talk to the docker daemon"; else die "cannot talk to the docker daemon (is your user in the docker group?)"; fi
else
  if docker info 2>/dev/null | grep -qi nvidia; then
    log "Docker reports an nvidia runtime"
  else
    warn "docker does not list an nvidia runtime; --gpus all may fail"
  fi
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  warn "nvidia-smi failed: is the NVIDIA driver working?"
fi

have_image=0 have_base=1
if command -v docker >/dev/null 2>&1 && docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  have_image=1
fi
if is_recipe_image && ! { command -v docker >/dev/null 2>&1 && docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; }; then
  have_base=0
fi
need_img=0
if is_recipe_image; then
  if (( have_base == 0 )); then
    need_img="${IMAGE_FREE_GB}"
  elif (( have_image == 0 )); then
    need_img=1
  fi
elif (( have_image == 0 )); then
  need_img="${IMAGE_FREE_GB}"
elif (( PULL == 1 )); then
  need_img=5
fi
need_ckpt=0
ckpt_what="the checkpoint is already in the cache"
if ! snapshot_dir >/dev/null 2>&1; then
  need_ckpt="${MIN_FREE_GB}"
  ckpt_what="the checkpoint is not in the cache yet (~96 GiB; MIN_FREE_GB=${MIN_FREE_GB})"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
else
  DOCKER_ROOT="/var/lib/docker"
fi
cache_free="$(free_gb "${HF_CACHE}")"
[[ -n "${cache_free}" ]] || cache_free=0
root_free="$(free_gb "${DOCKER_ROOT}")"
[[ -n "${root_free}" ]] || root_free=0

if [[ -d "${DOCKER_ROOT}" && "$(stat -c %d "${HF_CACHE}")" == "$(stat -c %d "${DOCKER_ROOT}")" ]]; then
  need=$(( need_ckpt + need_img ))
  if (( cache_free < need )); then
    msg="only ${cache_free} GB free under ${HF_CACHE} (also Docker's root ${DOCKER_ROOT}); ~${need} GB needed. ${ckpt_what}"
    if (( DRY )); then warn "DRY_RUN: ${msg}"; else die "${msg}"; fi
  fi
else
  if (( cache_free < need_ckpt )); then
    msg="only ${cache_free} GB free under ${HF_CACHE}; ~${need_ckpt} GB needed. ${ckpt_what}"
    if (( DRY )); then warn "DRY_RUN: ${msg}"; else die "${msg}"; fi
  fi
  if (( root_free < need_img )); then
    msg="only ${root_free} GB free under Docker's root (${DOCKER_ROOT}); the image needs ~${need_img} GB"
    if (( DRY )); then warn "DRY_RUN: ${msg}"; else die "${msg}"; fi
  fi
fi
log "Disk: ${cache_free} GB free under ${HF_CACHE} (${ckpt_what})"

if (( DRY )); then
  if is_recipe_image; then
    (( have_base == 1 || PULL == 0 )) || log "DRY_RUN: docker pull ${BASE_IMAGE}"
    (( have_image == 1 )) || log "DRY_RUN: docker build -t ${IMAGE} docker/ (BASE_IMAGE=${BASE_IMAGE})"
  elif (( PULL == 1 )); then
    log "DRY_RUN: docker pull ${IMAGE}"
  fi
  log "DRY_RUN: download ${MODEL_ID} into ${HF_CACHE}"
  exit 0
fi

if is_recipe_image; then
  if (( have_base == 1 )); then
    log "Base image ${BASE_IMAGE} is local"
  elif (( PULL == 1 )); then
    log "Pulling ${BASE_IMAGE}"
    docker pull "${BASE_IMAGE}"
  else
    die "base image ${BASE_IMAGE} is not local and PULL=0"
  fi
  if (( have_image == 1 )); then
    log "Recipe image ${IMAGE} is local"
  else
    log "Building ${IMAGE} from docker/ on ${BASE_IMAGE}"
    docker build --pull=false --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${IMAGE}" "${RECIPE_DOCKER_DIR}"
  fi
elif (( PULL == 1 )); then
  log "Pulling ${IMAGE}"
  docker pull "${IMAGE}"
elif (( have_image == 0 )); then
  die "image ${IMAGE} is not local and PULL=0"
else
  log "PULL=0: using local image ${IMAGE}"
fi
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} is not present after pull or build"

download_model() {
  local -a rev_args=() token_args=()
  if [[ -n "${MODEL_REVISION}" ]]; then
    rev_args=(--revision "${MODEL_REVISION}")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    token_args=(-e HF_TOKEN)
  fi
  if command -v hf >/dev/null 2>&1; then
    log "Downloading ${MODEL_ID} with hf into ${HF_CACHE}"
    HF_HOME="${HF_CACHE}" HF_HUB_OFFLINE=0 hf download "${MODEL_ID}" "${rev_args[@]}" || return 1
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    log "Downloading ${MODEL_ID} with huggingface-cli into ${HF_CACHE}"
    HF_HOME="${HF_CACHE}" HF_HUB_OFFLINE=0 huggingface-cli download "${MODEL_ID}" "${rev_args[@]}" || return 1
    return 0
  fi
  log "Host hf CLI not found; downloading ${MODEL_ID} inside ${IMAGE}"
  docker run --rm --network host --user "$(id -u):$(id -g)" \
    "${token_args[@]}" \
    -e HF_HUB_OFFLINE=0 \
    -e HF_HOME=/hf \
    -e HOME=/tmp \
    -v "${HF_CACHE}:/hf" \
    --entrypoint python3 \
    "${IMAGE}" -c \
    'import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
snapshot_download(repo, revision=rev or None, cache_dir="/hf/hub")' \
    "${MODEL_ID}" "${MODEL_REVISION}" || return 1
  return 0
}

if ! download_model; then
  if snapshot_dir >/dev/null 2>&1; then
    warn "could not reach Hugging Face; using the snapshot already in ${HF_CACHE}"
  else
    die "download of ${MODEL_ID} failed (accept the model terms on Hugging Face and set HF_TOKEN if the Hub asks for it)"
  fi
fi

snap="$(snapshot_dir)" || die "${MODEL_ID} is not in ${HF_CACHE} after the download"
log "Verifying ${snap}"
CHECK_PY='
import json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "qwen4_exp":
    sys.exit("model_type is %r, not qwen4_exp" % cfg.get("model_type"))
q = cfg.get("quantization_config") or {}
if q.get("quant_method") != "modelopt" or q.get("quant_algo") != "MIXED_PRECISION":
    sys.exit("quantization_config is %r / %r, not modelopt MIXED_PRECISION" % (q.get("quant_method"), q.get("quant_algo")))
idx_path = os.path.join(d, "model.safetensors.index.json")
if not os.path.isfile(idx_path):
    sys.exit("model.safetensors.index.json is missing")
idx = json.load(open(idx_path))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
if bad:
    sys.exit("missing or empty shards: " + ", ".join(bad[:5]))
print("qwen4_exp, modelopt MIXED_PRECISION, %d tensors in %d shards" % (len(idx["weight_map"]), len(shards)))
'
if ! info="$(python3 -c "${CHECK_PY}" "${snap}" 2>&1)"; then
  die "the checkpoint at ${snap} is not ready: ${info}"
fi
log "Checkpoint OK: ${info}"
prepared_state > "${PREPARED_MARKER}"
log "Done. Start the server with ./start.sh (port ${PORT})."

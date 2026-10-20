#!/usr/bin/env bash
# Shared settings for start.sh, stop.sh, and scripts/*.sh.
# A value comes from the first of these that sets it:
#   1. the environment: PORT=9000 ./start.sh
#   2. ./.env next to start.sh (KEY=value lines, read, never executed)
#   3. profiles/${PROFILE}.env when PROFILE is set (same format): PROFILE=1m ./start.sh
#   4. the default below
# start.sh and scripts/prepare.sh cd to the repository root before sourcing this file.

# Export KEY=value lines from file $1 for keys that are not set yet.
load_env_file() {
  local _line _key _value
  while IFS= read -r _line || [[ -n "${_line}" ]]; do
    [[ "${_line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key="${BASH_REMATCH[2]}"
    _value="${BASH_REMATCH[3]}"
    if [[ "${_value}" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "${_value}" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value="${BASH_REMATCH[1]}"
    else
      _value="${_value%%#*}"
      _value="${_value%"${_value##*[![:space:]]}"}"
    fi
    [[ -n "${!_key+x}" ]] || export "${_key}=${_value}"
  done < "$1"
}

if [[ -f .env ]]; then
  load_env_file .env
fi
# Opt-in launch profiles (README, Profiles). Empty is the default recipe.
PROFILE="${PROFILE:-}"
PROFILE_DIR="${PROFILE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/profiles}"
PROFILE_FILE=""
if [[ -n "${PROFILE}" && "${PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && -f "${PROFILE_DIR}/${PROFILE}.env" ]]; then
  PROFILE_FILE="${PROFILE_DIR}/${PROFILE}.env"
  load_env_file "${PROFILE_FILE}"
fi

# Checkpoint and container. Flags match the NVIDIA model card's DGX Spark command.
MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"
MODEL_REVISION="${MODEL_REVISION:-}"
IMAGE="${IMAGE:-eugr/spark-vllm-b12x:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-fp4}"
SERVED_NAME="${SERVED_NAME:-${MODEL_ID}}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${QUANTIZATION:-modelopt_mixed}"
LOAD_FORMAT="${LOAD_FORMAT:-b12x}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# Requests decoded at once. 8 users with 1k prompts get 111 t/s combined against 78 with a cap of 5, and the
# last first token comes after 4.4 s instead of 32 s. All 8 fit in the pool while they average under about
# 196k tokens each; past that vLLM pauses one and recomputes it later (slower, never wrong).
# Requests past the cap wait for a free slot.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Bytes for the KV cache (--kv-cache-memory-bytes). 31.15 GB holds 1,575,180 FP8 tokens: six full
# 262,144-token requests at once, with 6.9 GiB or more of memory left free while serving. Set it empty
# (KV_CACHE_MEMORY_BYTES=) for the GPU_MEMORY_UTILIZATION sizing, which drifts by about 0.7 GiB between boots.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES-31150000000}"
# YaRN rope scaling for windows past the checkpoint's 262,144 positions (NATIVE_MAX_LEN).
# Empty or 1 leaves rope alone. 4 allows MAX_MODEL_LEN up to 1,048,576.
YARN_FACTOR="${YARN_FACTOR:-}"
NATIVE_MAX_LEN="${NATIVE_MAX_LEN:-262144}"
# 0 omits --speculative-config. The model card uses 4 draft tokens. 3 decodes as fast on a Spark
# and leaves 4-9% more KV cache (README, Benchmarks).
MTP_TOKENS="${MTP_TOKENS:-3}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
GDN_DECODE_KERNEL="${GDN_DECODE_KERNEL:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"pass_config":{"fuse_act_quant":true}}'
fi
SHM_SIZE="${SHM_SIZE:-16g}"

# Runtime environment from the model card's DGX Spark section.
export VLLM_PLE_TABLE_MEMORY="${VLLM_PLE_TABLE_MEMORY:-disk}"
export B12X_DISK_BACKEND="${B12X_DISK_BACKEND:-io_uring}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export B12X_POLICY_MODE="${B12X_POLICY_MODE:-auto}"
export VLLM_MXFP8_LM_HEAD="${VLLM_MXFP8_LM_HEAD:-1}"
export VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE="${VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE:-w13}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-1}"
export B12X_W4A16_A4_PREFILL="${B12X_W4A16_A4_PREFILL:-1}"
# Reads in flight for the PLE n-gram table on NVMe (README, PLE queue depth). The image hard-codes 64. 256
# mounts patches/ple_disk_qd256.py, the image's own file with only that number changed, read-only over the
# original: prefill about 4% faster, identical output. start.sh mounts it only when the image's file is the
# one the patch was made from (PLE_DISK_SHA256); otherwise it warns and serves with 64. 64 mounts nothing.
PLE_QUEUE_DEPTH="${PLE_QUEUE_DEPTH:-256}"
PLE_DISK_PY="/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/ple_embedding/_disk.py"
PLE_DISK_SHA256="2232d15d5034d9c0cfb9dc3bdbf0e3f49bcc619f51c9ccee77364a0553c6ddeb"
PLE_PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/ple_disk_qd256.py"
# Set by ple_patch_check: 1 when the patch is mounted.
PLE_PATCH_ON=0
# Activation precision (README, Precision). fast (the default): the model card's settings, one FP4 term for
# prefill MoE activations and the dense format the autotuner picks for speed. quality: an FP4 residual term on
# top, and BF16 activations in the dense NVFP4/MXFP8 layers; closer to full precision token by token, about 21%
# less prefill speed. Setting either variable below directly wins.
PRECISION="${PRECISION:-fast}"
if [[ "${PRECISION}" == quality ]]; then
  export B12X_W4A16_A4_PREFILL_TERMS="${B12X_W4A16_A4_PREFILL_TERMS:-2}"
  export VLLM_B12X_DENSE_ACTIVATION_MODE="${VLLM_B12X_DENSE_ACTIVATION_MODE:-a16}"
else
  export B12X_W4A16_A4_PREFILL_TERMS="${B12X_W4A16_A4_PREFILL_TERMS:-1}"
  export VLLM_B12X_DENSE_ACTIVATION_MODE="${VLLM_B12X_DENSE_ACTIVATION_MODE:-auto}"
fi
# The serve container stays offline once the snapshot is on disk. prepare.sh turns this off for the download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Passed into the container with -e. HF_TOKEN is added separately, by name only, when it is set.
SPARK_ENV=(
  VLLM_PLE_TABLE_MEMORY
  B12X_DISK_BACKEND
  CUTE_DSL_ARCH
  SAFETENSORS_FAST_GPU
  VLLM_WORKER_MULTIPROC_METHOD
  VLLM_SSM_CONV_STATE_LAYOUT
  VLLM_USE_AOT_COMPILE
  VLLM_USE_MEGA_AOT_ARTIFACT
  VLLM_USE_V2_MODEL_RUNNER
  B12X_POLICY_MODE
  VLLM_MXFP8_LM_HEAD
  VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE
  VLLM_B12X_MOE_FP4_FORCE_A16
  B12X_W4A16_A4_PREFILL
  B12X_W4A16_A4_PREFILL_TERMS
  VLLM_B12X_DENSE_ACTIVATION_MODE
  HF_HUB_OFFLINE
)

# More container environment, as space-separated NAME=value pairs: EXTRA_ENV="B12X_FOO=1 B12X_BAR=2".
EXTRA_ENV="${EXTRA_ENV:-}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
VLLM_CACHE="${VLLM_CACHE:-${HOME}/.cache/vllm}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-${HOME}/.cache/flashinfer}"
TRITON_CACHE="${TRITON_CACHE:-${HOME}/.triton}"
B12X_CACHE="${B12X_CACHE:-${HOME}/.cache/b12x}"
STATE_DIR="${STATE_DIR:-${HOME}/.local/state/qwen38-flash-next-fp4}"
LOG_DIR="${LOG_DIR:-${HOME}/.cache/qwen38-flash-next-fp4/logs}"

# Disk and memory guards. These are launch checks, not a measured footprint.
# The checkpoint is about 96 GiB on disk (NVIDIA model card).
MIN_FREE_GB="${MIN_FREE_GB:-120}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-30}"
# MemAvailable required before docker run. Override, or set MEM_CHECK=0 to warn instead of exit.
MEM_NEED_GIB="${MEM_NEED_GIB:-90}"
# While weights load, start.sh stops the container if MemAvailable stays below MEM_FLOOR_GIB for
# MEM_FLOOR_GRACE seconds, or falls below MEM_HARD_FLOOR_GIB at all. The grace covers the dip of a few
# seconds to about 3-4 GiB that every load goes through while vLLM repacks the loaded weights. With the
# default pool allocated the server sits near 7 GiB, so the floor is 6.
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-6}"
MEM_FLOOR_GRACE="${MEM_FLOOR_GRACE:-30}"
MEM_HARD_FLOOR_GIB="${MEM_HARD_FLOOR_GIB:-2}"
MEM_CHECK="${MEM_CHECK:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
STOP_LOCK_WAIT="${STOP_LOCK_WAIT:-30}"
PREPARE="${PREPARE:-auto}"
PULL="${PULL:-1}"

_c() {
  if [[ -t "$1" ]]; then
    printf '\033[%sm' "$2"
  fi
}
log() { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die() { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { echo "${HF_CACHE}/hub/models--${MODEL_ID//\//--}"; }

snapshot_dir() {
  local dir rev
  dir="$(model_cache_dir)"
  if [[ -n "${MODEL_REVISION}" && -f "${dir}/snapshots/${MODEL_REVISION}/config.json" ]]; then
    echo "${dir}/snapshots/${MODEL_REVISION}"
    return 0
  fi
  rev="$(cat "${dir}/refs/main" 2>/dev/null || true)"
  [[ -n "${rev}" && -f "${dir}/snapshots/${rev}/config.json" ]] || return 1
  echo "${dir}/snapshots/${rev}"
}

prepared_state() {
  local id="missing"
  if command -v docker >/dev/null 2>&1; then
    id="$(docker image inspect -f '{{.Id}}' "${IMAGE}" 2>/dev/null || echo missing)"
  fi
  echo "model=${MODEL_ID} rev=${MODEL_REVISION:-main} image=${id} ple=${VLLM_PLE_TABLE_MEMORY}"
}

# Read by start.sh and scripts/prepare.sh after they source this file.
# shellcheck disable=SC2034
PREPARED_MARKER="${STATE_DIR}/prepared"

mem_available_gib() {
  [[ -r /proc/meminfo ]] || return 1
  awk '/^MemAvailable:/ { printf "%d", $2 / 1048576; found = 1 } END { if (!found) exit 1 }' /proc/meminfo
}

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

validate_config() {
  if [[ -z "${MODEL_ID}" || "${MODEL_ID}" != */* ]]; then
    die "MODEL_ID must look like org/name, not ${MODEL_ID}"
  fi
  if [[ -z "${IMAGE}" ]]; then
    die "IMAGE is empty"
  fi
  if [[ -z "${CONTAINER_NAME}" || ! "${CONTAINER_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    die "CONTAINER_NAME is not a docker name: ${CONTAINER_NAME}"
  fi
  if [[ -z "${SERVED_NAME}" ]]; then
    die "SERVED_NAME is empty"
  fi
  if [[ -z "${HOST}" ]]; then
    die "HOST is empty"
  fi
  if ! is_uint "${PORT}" || (( PORT < 1 || PORT > 65535 )); then
    die "PORT is 1 to 65535, not ${PORT}"
  fi
  if ! is_uint "${TP_SIZE}" || (( TP_SIZE < 1 )); then
    die "TP_SIZE is a positive integer, not ${TP_SIZE}"
  fi
  if ! is_uint "${PP_SIZE}" || (( PP_SIZE < 1 )); then
    die "PP_SIZE is a positive integer, not ${PP_SIZE}"
  fi
  if [[ ! "${DTYPE}" =~ ^(bfloat16|float16|float32)$ ]]; then
    die "DTYPE is bfloat16, float16, or float32, not ${DTYPE}"
  fi
  if [[ ! "${KV_CACHE_DTYPE}" =~ ^(auto|fp8|fp8_e4m3|fp8_e5m2|bfloat16)$ ]]; then
    die "KV_CACHE_DTYPE is not a known dtype: ${KV_CACHE_DTYPE}"
  fi
  if [[ -z "${QUANTIZATION}" ]]; then
    die "QUANTIZATION is empty"
  fi
  if [[ -z "${LOAD_FORMAT}" ]]; then
    die "LOAD_FORMAT is empty"
  fi
  if ! is_uint "${BLOCK_SIZE}" || (( BLOCK_SIZE < 1 )); then
    die "BLOCK_SIZE is a positive integer, not ${BLOCK_SIZE}"
  fi
  if [[ -n "${PROFILE}" && -z "${PROFILE_FILE}" ]]; then
    local names="" f
    for f in "${PROFILE_DIR}"/*.env; do
      [[ -f "${f}" ]] && names+=" $(basename "${f}" .env)"
    done
    die "PROFILE ${PROFILE} has no file ${PROFILE_DIR}/${PROFILE}.env (profiles:${names:- none})"
  fi
  if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]] && { ! is_uint "${KV_CACHE_MEMORY_BYTES}" || (( KV_CACHE_MEMORY_BYTES < 1073741824 )); }; then
    die "KV_CACHE_MEMORY_BYTES is a byte count of at least 1 GiB, not ${KV_CACHE_MEMORY_BYTES}"
  fi
  if [[ -n "${YARN_FACTOR}" ]] && { ! is_uint "${YARN_FACTOR}" || (( YARN_FACTOR < 1 || YARN_FACTOR > 16 )); }; then
    die "YARN_FACTOR is 1 to 16, not ${YARN_FACTOR}"
  fi
  if [[ ! "${PRECISION}" =~ ^(quality|fast)$ ]]; then
    die "PRECISION is quality or fast, not ${PRECISION}"
  fi
  if [[ ! "${B12X_W4A16_A4_PREFILL_TERMS}" =~ ^[12]$ ]]; then
    die "B12X_W4A16_A4_PREFILL_TERMS is 1 or 2, not ${B12X_W4A16_A4_PREFILL_TERMS}"
  fi
  if [[ ! "${VLLM_B12X_DENSE_ACTIVATION_MODE}" =~ ^(auto|a16|quantized)$ ]]; then
    die "VLLM_B12X_DENSE_ACTIVATION_MODE is auto, a16, or quantized, not ${VLLM_B12X_DENSE_ACTIVATION_MODE}"
  fi
  if ! is_uint "${NATIVE_MAX_LEN}" || (( NATIVE_MAX_LEN < 1 )); then
    die "NATIVE_MAX_LEN is a positive integer, not ${NATIVE_MAX_LEN}"
  fi
  local max_len=$(( NATIVE_MAX_LEN * ${YARN_FACTOR:-1} ))
  if ! is_uint "${MAX_MODEL_LEN}" || (( MAX_MODEL_LEN < 1 || MAX_MODEL_LEN > max_len )); then
    die "MAX_MODEL_LEN is 1 to ${max_len} with YARN_FACTOR ${YARN_FACTOR:-unset}, not ${MAX_MODEL_LEN} (PROFILE=1m sets both)"
  fi
  if ! is_uint "${MAX_NUM_SEQS}" || (( MAX_NUM_SEQS < 1 || MAX_NUM_SEQS > 256 )); then
    die "MAX_NUM_SEQS is 1 to 256, not ${MAX_NUM_SEQS}"
  fi
  if ! is_uint "${MAX_NUM_BATCHED_TOKENS}" || (( MAX_NUM_BATCHED_TOKENS < 1 )); then
    die "MAX_NUM_BATCHED_TOKENS is a positive integer, not ${MAX_NUM_BATCHED_TOKENS}"
  fi
  if ! is_uint "${MTP_TOKENS}" || (( MTP_TOKENS > 16 )); then
    die "MTP_TOKENS is 0 to 16, not ${MTP_TOKENS}"
  fi
  if ! awk -v v="${GPU_MEMORY_UTILIZATION}" 'BEGIN { if (v !~ /^[0-9]*(\.[0-9]+)?$/) exit 1; n = v + 0; if (n <= 0 || n > 1) exit 1 }'; then
    die "GPU_MEMORY_UTILIZATION is between 0 and 1, not ${GPU_MEMORY_UTILIZATION}"
  fi
  if [[ -z "${VLLM_PLE_TABLE_MEMORY}" ]]; then
    die "VLLM_PLE_TABLE_MEMORY is empty"
  fi
  if [[ ! "${PREPARE}" =~ ^(auto|0|1)$ ]]; then
    die "PREPARE is auto, 0, or 1, not ${PREPARE}"
  fi
  if [[ ! "${PULL}" =~ ^[01]$ ]]; then
    die "PULL is 0 or 1, not ${PULL}"
  fi
  if [[ ! "${MEM_CHECK}" =~ ^[01]$ ]]; then
    die "MEM_CHECK is 0 or 1, not ${MEM_CHECK}"
  fi
  if ! is_uint "${MEM_NEED_GIB}"; then
    die "MEM_NEED_GIB is a GiB count, not ${MEM_NEED_GIB}"
  fi
  if ! is_uint "${MEM_FLOOR_GIB}"; then
    die "MEM_FLOOR_GIB is a GiB count, not ${MEM_FLOOR_GIB}"
  fi
  local pair
  local -a pairs=()
  read -r -a pairs <<< "${EXTRA_ENV}"
  for pair in "${pairs[@]}"; do
    [[ "${pair}" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] || die "EXTRA_ENV holds NAME=value pairs, not ${pair}"
  done
  if ! is_uint "${MEM_FLOOR_GRACE}"; then
    die "MEM_FLOOR_GRACE is seconds, not ${MEM_FLOOR_GRACE}"
  fi
  if ! is_uint "${MEM_HARD_FLOOR_GIB}" || (( MEM_HARD_FLOOR_GIB > MEM_FLOOR_GIB )); then
    die "MEM_HARD_FLOOR_GIB is a GiB count no higher than MEM_FLOOR_GIB, not ${MEM_HARD_FLOOR_GIB}"
  fi
  if ! is_uint "${MIN_FREE_GB}"; then
    die "MIN_FREE_GB is a GB count, not ${MIN_FREE_GB}"
  fi
  if ! is_uint "${IMAGE_FREE_GB}"; then
    die "IMAGE_FREE_GB is a GB count, not ${IMAGE_FREE_GB}"
  fi
  if ! is_uint "${WAIT_TIMEOUT}" || (( WAIT_TIMEOUT < 1 )); then
    die "WAIT_TIMEOUT is seconds, not ${WAIT_TIMEOUT}"
  fi
  if [[ ! "${PLE_QUEUE_DEPTH}" =~ ^(64|256)$ ]]; then
    die "PLE_QUEUE_DEPTH is 64 or 256, not ${PLE_QUEUE_DEPTH}"
  fi
}

# PLE_PATCH_ON=1 when PLE_QUEUE_DEPTH is 256 and the image's _disk.py is the file the patch was made from.
# Any other image keeps its own file: a patch made from different code could change the model.
ple_patch_check() {
  PLE_PATCH_ON=0
  [[ "${PLE_QUEUE_DEPTH}" == 256 ]] || return 0
  local sum
  sum="$(docker run --rm --entrypoint sha256sum "${IMAGE}" "${PLE_DISK_PY}" 2>/dev/null | cut -d ' ' -f 1)" || sum=""
  if [[ "${sum}" == "${PLE_DISK_SHA256}" ]]; then
    PLE_PATCH_ON=1
  else
    warn "${IMAGE} has a different ${PLE_DISK_PY##*/} (sha256 ${sum:-unreadable}): PLE queue depth stays 64"
  fi
}

# --hf-overrides for YaRN. vLLM replaces rope_parameters as a whole, so every key of the checkpoint's
# text_config.rope_parameters is restated. vLLM also checks --max-model-len against
# max_position_embeddings, not against the YaRN factor, so that is raised to match.
yarn_overrides() {
  printf '{"text_config":{"max_position_embeddings":%d,"rope_parameters":{"rope_type":"yarn","factor":%d,"original_max_position_embeddings":%d,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}' \
    "$(( NATIVE_MAX_LEN * YARN_FACTOR ))" "${YARN_FACTOR}" "${NATIVE_MAX_LEN}"
}

# Fill the global DOCKER_CMD array. Extra arguments are appended and override earlier flags
# when vLLM keeps the last value. HF_TOKEN is forwarded by name and is not placed on the command line.
build_docker_cmd() {
  local -a cmd=("$@")
  local key
  DOCKER_CMD=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --gpus all
    --ipc host
    --network host
    --shm-size "${SHM_SIZE}"
    --ulimit memlock=-1
    --ulimit stack=67108864
    --security-opt seccomp=unconfined
    --security-opt apparmor=unconfined
  )
  for key in "${SPARK_ENV[@]}"; do
    DOCKER_CMD+=(-e "${key}=${!key}")
  done
  local -a extra=()
  read -r -a extra <<< "${EXTRA_ENV}"
  for key in "${extra[@]}"; do
    DOCKER_CMD+=(-e "${key}")
  done
  if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_CMD+=(-e HF_TOKEN)
  fi
  if [[ "${PLE_PATCH_ON}" == 1 ]]; then
    DOCKER_CMD+=(-v "${PLE_PATCH_FILE}:${PLE_DISK_PY}:ro")
  fi
  DOCKER_CMD+=(
    -e "HF_HOME=/root/.cache/huggingface"
    -v "${HF_CACHE}:/root/.cache/huggingface"
    -v "${VLLM_CACHE}:/root/.cache/vllm"
    -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer"
    -v "${TRITON_CACHE}:/root/.triton"
    -v "${B12X_CACHE}:/root/.cache/b12x"
    --entrypoint vllm
    "${IMAGE}"
    serve "${MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --mamba-cache-mode "${MAMBA_CACHE_MODE}"
    --enable-prefix-caching
    --enable-chunked-prefill
    --dtype "${DTYPE}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --quantization "${QUANTIZATION}"
    --block-size "${BLOCK_SIZE}"
    --load-format "${LOAD_FORMAT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  )
  if (( MTP_TOKENS > 0 )); then
    DOCKER_CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
  fi
  DOCKER_CMD+=(
    --gdn-decode-kernel "${GDN_DECODE_KERNEL}"
    --linear-backend "${LINEAR_BACKEND}"
    --moe-backend "${MOE_BACKEND}"
    --no-enable-flashinfer-autotune
    --mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}"
    --reasoning-parser "${REASONING_PARSER}"
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --enable-auto-tool-choice
    --compilation-config "${COMPILATION_CONFIG}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )
  if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
    DOCKER_CMD+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
  fi
  if (( ${YARN_FACTOR:-1} > 1 )); then
    DOCKER_CMD+=(--hf-overrides "$(yarn_overrides)")
  fi
  if (( ${#cmd[@]} > 0 )); then
    DOCKER_CMD+=("${cmd[@]}")
  fi
}

print_docker_cmd() {
  printf '%q ' "${DOCKER_CMD[@]}"
  printf '\n'
}

# Stop and remove the serve container. Caller holds the lifecycle lock, or accepts the race.
stop_container() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    log "No container named ${CONTAINER_NAME}: nothing to stop"
    return 0
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)" == true ]]; then
    log "Stopping ${CONTAINER_NAME} (up to ${STOP_TIMEOUT}s)"
    docker stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null
  log "Stopped and removed ${CONTAINER_NAME}"
}

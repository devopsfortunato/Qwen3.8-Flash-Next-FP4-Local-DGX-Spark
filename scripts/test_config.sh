#!/usr/bin/env bash
# Check scripts/config.sh defaults, .env versus environment precedence, and build_docker_cmd.
# No docker, GPU, or network. Run from anywhere: ./scripts/test_config.sh
# Each case is a subshell so one case's PORT cannot leak into the next.
# shellcheck disable=SC2030,SC2031
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CONFIG="${ROOT}/scripts/config.sh"
N=0

pass() { N=$((N + 1)); printf 'ok  %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1" >&2; exit 1; }

unset_recipe() {
  unset MODEL_ID MODEL_REVISION IMAGE CONTAINER_NAME SERVED_NAME HOST PORT TP_SIZE PP_SIZE \
    DTYPE KV_CACHE_DTYPE QUANTIZATION LOAD_FORMAT BLOCK_SIZE MAX_MODEL_LEN MAX_NUM_SEQS \
    MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION MTP_TOKENS MAMBA_CACHE_MODE \
    GDN_DECODE_KERNEL LINEAR_BACKEND MOE_BACKEND MM_ENCODER_TP_MODE REASONING_PARSER \
    TOOL_CALL_PARSER COMPILATION_CONFIG SHM_SIZE \
    VLLM_PLE_TABLE_MEMORY B12X_DISK_BACKEND CUTE_DSL_ARCH SAFETENSORS_FAST_GPU \
    VLLM_WORKER_MULTIPROC_METHOD VLLM_SSM_CONV_STATE_LAYOUT VLLM_USE_AOT_COMPILE \
    VLLM_USE_MEGA_AOT_ARTIFACT VLLM_USE_V2_MODEL_RUNNER B12X_POLICY_MODE \
    VLLM_MXFP8_LM_HEAD VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE VLLM_B12X_MOE_FP4_FORCE_A16 \
    B12X_W4A16_A4_PREFILL HF_HUB_OFFLINE HF_TOKEN HF_HOME HF_CACHE VLLM_CACHE \
    FLASHINFER_CACHE TRITON_CACHE B12X_CACHE STATE_DIR LOG_DIR MIN_FREE_GB IMAGE_FREE_GB \
    MEM_NEED_GIB MEM_FLOOR_GIB MEM_CHECK WAIT_TIMEOUT STOP_TIMEOUT STOP_LOCK_WAIT PREPARE PULL \
    DRY_RUN FOREGROUND || true
}

has_pair() {
  local flag="$1" val="$2" i
  for (( i = 0; i < ${#DOCKER_CMD[@]} - 1; i++ )); do
    if [[ "${DOCKER_CMD[i]}" == "${flag}" && "${DOCKER_CMD[i + 1]}" == "${val}" ]]; then
      return 0
    fi
  done
  return 1
}

has_word() {
  local want="$1" word
  for word in "${DOCKER_CMD[@]}"; do
    [[ "${word}" == "${want}" ]] && return 0
  done
  return 1
}

check_defaults() {
  [[ "${MODEL_ID}" == "nvidia/Qwen3.8-Flash-Next-FP4-Local" ]] || return 1
  [[ "${IMAGE}" == "eugr/spark-vllm-b12x:latest" ]] || return 1
  [[ "${PORT}" == "8000" ]] || return 1
  [[ "${HOST}" == "0.0.0.0" ]] || return 1
  [[ "${SERVED_NAME}" == "${MODEL_ID}" ]] || return 1
  [[ "${QUANTIZATION}" == "modelopt_mixed" ]] || return 1
  [[ "${LOAD_FORMAT}" == "b12x" ]] || return 1
  [[ "${KV_CACHE_DTYPE}" == "fp8" ]] || return 1
  [[ "${MTP_TOKENS}" == "4" ]] || return 1
  [[ "${MAX_MODEL_LEN}" == "262144" ]] || return 1
  [[ "${MAX_NUM_SEQS}" == "8" ]] || return 1
  [[ "${GPU_MEMORY_UTILIZATION}" == "0.8" ]] || return 1
  [[ "${VLLM_PLE_TABLE_MEMORY}" == "disk" ]] || return 1
  [[ "${B12X_DISK_BACKEND}" == "io_uring" ]] || return 1
  [[ "${CUTE_DSL_ARCH}" == "sm_121a" ]] || return 1
  [[ "${VLLM_B12X_MOE_FP4_FORCE_A16}" == "1" ]] || return 1
  [[ "${B12X_W4A16_A4_PREFILL}" == "1" ]] || return 1
}

check_cmd() {
  validate_config
  build_docker_cmd
  has_pair docker run || return 1
  has_pair --gpus all || return 1
  has_pair --network host || return 1
  has_pair --ipc host || return 1
  has_pair --entrypoint vllm || return 1
  has_pair --port 8000 || return 1
  has_pair --host 0.0.0.0 || return 1
  has_pair serve nvidia/Qwen3.8-Flash-Next-FP4-Local || return 1
  has_pair --load-format b12x || return 1
  has_pair --quantization modelopt_mixed || return 1
  has_pair --kv-cache-dtype fp8 || return 1
  has_pair --dtype bfloat16 || return 1
  has_pair --block-size 16 || return 1
  has_pair --max-model-len 262144 || return 1
  has_pair --max-num-seqs 8 || return 1
  has_pair --max-num-batched-tokens 4096 || return 1
  has_pair --gdn-decode-kernel b12x || return 1
  has_pair --linear-backend b12x || return 1
  has_pair --moe-backend b12x || return 1
  has_pair --reasoning-parser qwen3 || return 1
  has_pair --tool-call-parser qwen3_xml || return 1
  has_pair --gpu-memory-utilization 0.8 || return 1
  has_pair --mamba-cache-mode align || return 1
  has_pair --mm-encoder-tp-mode data || return 1
  has_word --enable-prefix-caching || return 1
  has_word --enable-chunked-prefill || return 1
  has_word --no-enable-flashinfer-autotune || return 1
  has_word --enable-auto-tool-choice || return 1
  has_word "VLLM_PLE_TABLE_MEMORY=disk" || return 1
  has_word "B12X_DISK_BACKEND=io_uring" || return 1
  has_word "CUTE_DSL_ARCH=sm_121a" || return 1
  has_word "SAFETENSORS_FAST_GPU=1" || return 1
  has_word "VLLM_WORKER_MULTIPROC_METHOD=spawn" || return 1
  has_word "VLLM_SSM_CONV_STATE_LAYOUT=DS" || return 1
  has_word "VLLM_USE_AOT_COMPILE=1" || return 1
  has_word "VLLM_USE_MEGA_AOT_ARTIFACT=1" || return 1
  has_word "VLLM_USE_V2_MODEL_RUNNER=1" || return 1
  has_word "B12X_POLICY_MODE=auto" || return 1
  has_word "VLLM_MXFP8_LM_HEAD=1" || return 1
  has_word "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13" || return 1
  has_word "VLLM_B12X_MOE_FP4_FORCE_A16=1" || return 1
  has_word "B12X_W4A16_A4_PREFILL=1" || return 1
  has_word "--speculative-config" || return 1
  has_word '{"method":"mtp","num_speculative_tokens":4}' || return 1
  has_word '{"pass_config":{"fuse_act_quant":true}}' || return 1
  has_word "eugr/spark-vllm-b12x:latest" || return 1
  printf '%s\n' "${DOCKER_CMD[@]}" | grep -q 'secret-token' && return 1
  return 0
}

check_dotenv() {
  [[ "${PORT}" == "9000" ]] || return 1
  [[ "${MAX_MODEL_LEN}" == "8192" ]] || return 1
  [[ "${SERVED_NAME}" == "Quoted Name" ]] || return 1
  [[ "${MODEL_ID}" == "nvidia/Qwen3.8-Flash-Next-FP4-Local" ]] || return 1
}

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

(
  unset_recipe
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  check_defaults
)
pass "defaults match the model-card Spark recipe"

(
  unset_recipe
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  check_cmd
)
pass "build_docker_cmd carries the Spark flags"

env_dir="${tmp}/env"
mkdir -p "${env_dir}"
printf 'PORT=9000\nMAX_MODEL_LEN=8192\nSERVED_NAME="Quoted Name"\n' > "${env_dir}/.env"
(
  unset_recipe
  cd "${env_dir}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  check_dotenv
)
pass ".env fills unset keys"

win_dir="${tmp}/win"
mkdir -p "${win_dir}"
printf 'PORT=9000\nIMAGE=from-dotenv:latest\n' > "${win_dir}/.env"
(
  unset_recipe
  PORT=1234
  cd "${win_dir}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  [[ "${PORT}" == "1234" ]]
  [[ "${IMAGE}" == "from-dotenv:latest" ]]
)
pass "environment wins over .env"

(
  unset_recipe
  PORT=9001 MTP_TOKENS=0 VLLM_PLE_TABLE_MEMORY=gpu KV_CACHE_DTYPE=bfloat16 HF_TOKEN=secret-token
  export PORT MTP_TOKENS VLLM_PLE_TABLE_MEMORY KV_CACHE_DTYPE HF_TOKEN
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  build_docker_cmd --max-num-seqs 2
  has_pair --port 9001
  has_word "VLLM_PLE_TABLE_MEMORY=gpu"
  has_pair --kv-cache-dtype bfloat16
  has_pair --max-num-seqs 2
  has_pair -e HF_TOKEN
  if has_word "--speculative-config"; then
    exit 1
  fi
  if printf '%s\n' "${DOCKER_CMD[@]}" | grep -q 'secret-token'; then
    exit 1
  fi
)
pass "environment overrides reach the docker command and the token stays off the command line"

for bad in \
  'PORT=0' \
  'PORT=70000' \
  'GPU_MEMORY_UTILIZATION=0' \
  'GPU_MEMORY_UTILIZATION=1.5' \
  'MTP_TOKENS=99' \
  'MAX_MODEL_LEN=0' \
  'PREPARE=maybe' \
  'MODEL_ID=not-a-repo'
do
  key="${bad%%=*}"
  val="${bad#*=}"
  set +e
  (
    unset_recipe
    export "${key}=${val}"
    cd "${tmp}"
    # shellcheck source=scripts/config.sh
    source "${CONFIG}"
    validate_config
  ) >/dev/null 2>&1
  status=$?
  set -e
  [[ "${status}" -ne 0 ]] || fail "validate_config accepted ${bad}"
done
pass "validate_config rejects bad values"

dry="$(DRY_RUN=1 "${ROOT}/start.sh")"
printf '%s\n' "${dry}" | grep -q '\[dry-run\]' || fail "DRY_RUN did not print a command"
printf '%s\n' "${dry}" | grep -q 'VLLM_PLE_TABLE_MEMORY=disk' || fail "DRY_RUN dropped the PLE flag"
printf '%s\n' "${dry}" | grep -q 'eugr/spark-vllm-b12x:latest' || fail "DRY_RUN dropped the image"
printf '%s\n' "${dry}" | grep -q -- '--network' || fail "DRY_RUN dropped host networking"
printf '%s\n' "${dry}" | grep -q -- '--gpus' || fail "DRY_RUN dropped --gpus"
pass "DRY_RUN=1 ./start.sh prints the docker command"

echo "test_config: ${N} checks passed"

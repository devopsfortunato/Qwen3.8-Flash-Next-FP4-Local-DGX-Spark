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
    B12X_W4A16_A4_PREFILL PRECISION B12X_W4A16_A4_PREFILL_TERMS VLLM_B12X_DENSE_ACTIVATION_MODE \
    HF_HUB_OFFLINE HF_TOKEN HF_HOME HF_CACHE VLLM_CACHE \
    FLASHINFER_CACHE TRITON_CACHE B12X_CACHE STATE_DIR LOG_DIR MIN_FREE_GB IMAGE_FREE_GB \
    MEM_NEED_GIB MEM_FLOOR_GIB MEM_FLOOR_GRACE MEM_HARD_FLOOR_GIB MEM_CHECK WAIT_TIMEOUT STOP_TIMEOUT \
    STOP_LOCK_WAIT PREPARE PULL DRY_RUN FOREGROUND PROFILE PROFILE_DIR PROFILE_FILE \
    KV_CACHE_MEMORY_BYTES YARN_FACTOR NATIVE_MAX_LEN EXTRA_ENV PLE_QUEUE_DEPTH DRAFT_VOCAB || true
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
  [[ "${MTP_TOKENS}" == "3" ]] || return 1
  [[ -z "${PROFILE}" && "${KV_CACHE_MEMORY_BYTES}" == "31150000000" && -z "${YARN_FACTOR}" ]] || return 1
  [[ "${MAX_MODEL_LEN}" == "262144" ]] || return 1
  [[ "${MAX_NUM_SEQS}" == "8" ]] || return 1
  [[ "${PLE_QUEUE_DEPTH}" == "256" && "${PLE_PATCH_ON}" == 0 ]] || return 1
  [[ "${DRAFT_VOCAB}" == "full" && "${DV_PATCH_ON}" == 0 ]] || return 1
  [[ "${GPU_MEMORY_UTILIZATION}" == "0.8" ]] || return 1
  [[ "${VLLM_PLE_TABLE_MEMORY}" == "disk" ]] || return 1
  [[ "${B12X_DISK_BACKEND}" == "io_uring" ]] || return 1
  [[ "${CUTE_DSL_ARCH}" == "sm_121a" ]] || return 1
  [[ "${VLLM_B12X_MOE_FP4_FORCE_A16}" == "1" ]] || return 1
  [[ "${B12X_W4A16_A4_PREFILL}" == "1" ]] || return 1
  [[ "${PRECISION}" == "fast" ]] || return 1
  [[ "${B12X_W4A16_A4_PREFILL_TERMS}" == "1" ]] || return 1
  [[ "${VLLM_B12X_DENSE_ACTIVATION_MODE}" == "auto" ]] || return 1
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
  has_word "B12X_W4A16_A4_PREFILL_TERMS=1" || return 1
  has_word "VLLM_B12X_DENSE_ACTIVATION_MODE=auto" || return 1
  has_word "--speculative-config" || return 1
  has_word '{"method":"mtp","num_speculative_tokens":3}' || return 1
  has_pair --kv-cache-memory-bytes 31150000000 || return 1
  has_word --hf-overrides && return 1
  has_word '{"pass_config":{"fuse_act_quant":true}}' || return 1
  has_word "eugr/spark-vllm-b12x:latest" || return 1
  printf '%s\n' "${DOCKER_CMD[@]}" | grep -q 'secret-token' && return 1
  printf '%s\n' "${DOCKER_CMD[@]}" | grep -q '_disk.py:ro' && return 1
  printf '%s\n' "${DOCKER_CMD[@]}" | grep -q -e 'mtp.py:ro' -e 'VLLM_MTP_DRAFT_VOCAB' && return 1
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
  KV_CACHE_MEMORY_BYTES=
  export KV_CACHE_MEMORY_BYTES
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  build_docker_cmd
  if has_word --kv-cache-memory-bytes; then
    exit 1
  fi
  has_pair --gpu-memory-utilization 0.8
)
pass "an empty KV_CACHE_MEMORY_BYTES keeps the utilization sizing"

(
  unset_recipe
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  [[ -f "${PLE_PATCH_FILE}" ]]
  grep -q 'queue_depth: int = 256' "${PLE_PATCH_FILE}"
  PLE_PATCH_ON=1
  build_docker_cmd
  has_pair -v "${PLE_PATCH_FILE}:${PLE_DISK_PY}:ro"
)
pass "the PLE queue-depth patch exists and mounts read-only once patch_check accepts the image"

(
  unset_recipe
  cd "${tmp}"
  DRAFT_VOCAB=latin
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  [[ -f "${DV_PATCH_FILE}" && -f "${DV_IDS_FILE}" ]]
  grep -q 'VLLM_MTP_DRAFT_VOCAB' "${DV_PATCH_FILE}"
  wanted_patches
  [[ "${PLE_PATCH_ON}" == 1 && "${DV_PATCH_ON}" == 1 ]]
  build_docker_cmd
  has_pair -v "${DV_PATCH_FILE}:${MTP_PY}:ro"
  has_pair -v "${DV_IDS_FILE}:${DV_IDS_MOUNT}:ro"
  has_pair -e "VLLM_MTP_DRAFT_VOCAB=${DV_IDS_MOUNT}"
  DRAFT_VOCAB=full
  wanted_patches
  [[ "${DV_PATCH_ON}" == 0 ]]
  DRAFT_VOCAB=latin MTP_TOKENS=0
  wanted_patches
  [[ "${DV_PATCH_ON}" == 0 ]]
)
pass "the draft-vocabulary patch and id list mount read-only for DRAFT_VOCAB=latin, not for full or MTP_TOKENS=0"

(
  unset_recipe
  PORT=9001 MTP_TOKENS=0 VLLM_PLE_TABLE_MEMORY=gpu KV_CACHE_DTYPE=bfloat16 HF_TOKEN=secret-token
  export PORT MTP_TOKENS VLLM_PLE_TABLE_MEMORY KV_CACHE_DTYPE HF_TOKEN
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  EXTRA_ENV="B12X_TEST_A=1 B12X_TEST_B=two"
  validate_config
  build_docker_cmd --max-num-seqs 2
  has_pair -e B12X_TEST_A=1
  has_pair -e B12X_TEST_B=two
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

(
  unset_recipe
  PRECISION=quality
  export PRECISION
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  build_docker_cmd
  has_word "B12X_W4A16_A4_PREFILL_TERMS=2"
  has_word "VLLM_B12X_DENSE_ACTIVATION_MODE=a16"
)
(
  unset_recipe
  PRECISION=quality VLLM_B12X_DENSE_ACTIVATION_MODE=auto
  export PRECISION VLLM_B12X_DENSE_ACTIVATION_MODE
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  [[ "${B12X_W4A16_A4_PREFILL_TERMS}" == "2" ]]
  [[ "${VLLM_B12X_DENSE_ACTIVATION_MODE}" == "auto" ]]
)
pass "PRECISION=quality sets the two-term prefill and BF16 dense activations, and a variable set directly wins"

prof_dir="${tmp}/profiles"
mkdir -p "${prof_dir}"
printf 'KV_CACHE_MEMORY_BYTES=30000000000\nYARN_FACTOR=4\nMAX_MODEL_LEN=1048576\nPORT=7000\n' > "${prof_dir}/test.env"
(
  unset_recipe
  PROFILE=test PROFILE_DIR="${prof_dir}"
  export PROFILE PROFILE_DIR
  cd "${tmp}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  validate_config
  build_docker_cmd
  has_pair --kv-cache-memory-bytes 30000000000
  has_pair --max-model-len 1048576
  has_pair --port 7000
  json=""
  for (( i = 0; i < ${#DOCKER_CMD[@]} - 1; i++ )); do
    [[ "${DOCKER_CMD[i]}" == --hf-overrides ]] && json="${DOCKER_CMD[i + 1]}"
  done
  python3 -c 'import json, sys
t = json.loads(sys.argv[1])["text_config"]
r = t["rope_parameters"]
assert t["max_position_embeddings"] == 1048576
assert r["rope_type"] == "yarn" and r["factor"] == 4 and r["original_max_position_embeddings"] == 262144
assert r["mrope_section"] == [11, 11, 10] and r["mrope_interleaved"] is True' "${json}"
)
pass "a profile sets the KV pool and YaRN, and the overrides are valid JSON"

prof_env="${tmp}/prof-env"
mkdir -p "${prof_env}"
printf 'MAX_MODEL_LEN=524288\n' > "${prof_env}/.env"
(
  unset_recipe
  PROFILE=test PROFILE_DIR="${prof_dir}" PORT=7100
  export PROFILE PROFILE_DIR PORT
  cd "${prof_env}"
  # shellcheck source=scripts/config.sh
  source "${CONFIG}"
  [[ "${PORT}" == "7100" ]]
  [[ "${MAX_MODEL_LEN}" == "524288" ]]
  [[ "${YARN_FACTOR}" == "4" ]]
)
pass "environment and .env win over the profile"

for prof in "${ROOT}"/profiles/*.env; do
  [[ -f "${prof}" ]] || continue
  name="$(basename "${prof}" .env)"
  (
    unset_recipe
    PROFILE="${name}"
    export PROFILE
    cd "${tmp}"
    # shellcheck source=scripts/config.sh
    source "${CONFIG}"
    validate_config
    build_docker_cmd
    has_pair --max-model-len "${MAX_MODEL_LEN}"
    has_pair --max-num-seqs "${MAX_NUM_SEQS}"
  ) >/dev/null || fail "profile ${name} does not validate"
  pass "profile ${name} validates and builds"
done

for bad in \
  'PORT=0' \
  'PORT=70000' \
  'GPU_MEMORY_UTILIZATION=0' \
  'GPU_MEMORY_UTILIZATION=1.5' \
  'MTP_TOKENS=99' \
  'MAX_MODEL_LEN=0' \
  'MAX_MODEL_LEN=1048576' \
  'YARN_FACTOR=0' \
  'KV_CACHE_MEMORY_BYTES=30G' \
  'PRECISION=max' \
  'PLE_QUEUE_DEPTH=128' \
  'DRAFT_VOCAB=small' \
  'B12X_W4A16_A4_PREFILL_TERMS=3' \
  'VLLM_B12X_DENSE_ACTIVATION_MODE=bf16' \
  'MEM_HARD_FLOOR_GIB=9' \
  'PROFILE=no-such-profile' \
  'EXTRA_ENV=not-a-pair' \
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

dry="$(DRY_RUN=1 DRAFT_VOCAB=latin "${ROOT}/start.sh")"
printf '%s\n' "${dry}" | grep -q '\[dry-run\]' || fail "DRY_RUN did not print a command"
printf '%s\n' "${dry}" | grep -q 'VLLM_PLE_TABLE_MEMORY=disk' || fail "DRY_RUN dropped the PLE flag"
printf '%s\n' "${dry}" | grep -q 'eugr/spark-vllm-b12x:latest' || fail "DRY_RUN dropped the image"
printf '%s\n' "${dry}" | grep -q -- '--network' || fail "DRY_RUN dropped host networking"
printf '%s\n' "${dry}" | grep -q -- '--gpus' || fail "DRY_RUN dropped --gpus"
printf '%s\n' "${dry}" | grep -q 'ple_disk_qd256.py' || fail "DRY_RUN dropped the PLE queue-depth patch"
printf '%s\n' "${dry}" | grep -q 'mtp_draft_vocab.py' || fail "DRY_RUN dropped the draft-vocabulary patch"
pass "DRY_RUN=1 ./start.sh prints the docker command"

echo "test_config: ${N} checks passed"

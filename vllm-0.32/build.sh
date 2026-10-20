#!/usr/bin/env bash
# Build the upstream-v0.32.0 Qwen3.8 Flash Next FP4 runner image on the Spark.
#
#   ./vllm-0.32/build.sh <path-to-spark-vllm-docker-checkout> <path-to-patched-vllm-checkout> [extra build-and-copy.sh flags]
#
# Produces the local image tag qwen38-flash-next-fp4:vllm-0.32.0-<12-hex>,
# where <12-hex> prefixes the patched source tree (EXPECTED_TREE below), the
# IMAGE profiles/vllm-0.32.env names. Serve it with:
#   PROFILE=vllm-0.32 ./start.sh
#
# The vLLM source is any clean Git checkout of tag v0.32.0 with this
# directory's series applied on top (one commit per patch, e.g. with
# `git am vllm-0.32/*.patch`). The tree is verified by content, not by
# branch name. eugr's build-and-copy.sh compiles the vLLM wheel and the
# runner; the preset vLLM patch series stays OFF for --vllm-source-dir builds
# because the series is already applied. FlashInfer stays on the regular
# lane's default: prebuilt wheels from eugr's prebuilt-flashinfer-current
# release (upstream FlashInfer + the qwen3.8 flashinfer series #5863 and
# #5943). Pass --rebuild-flashinfer to compile FlashInfer from its default
# main ref instead.
#
# Minimal source preparation, from scratch:
#   git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm vllm-v0.32.0-qwen38
#   cd vllm-v0.32.0-qwen38 && git checkout -b qwen38-flash-next-v0.32.0
#   git am <recipe>/vllm-0.32/*.patch
set -euo pipefail

# The spark-vllm-docker revision this series was validated against.
SPARK_VLLM_DOCKER_REF="eadf57da73953fe71326dbd10ef45fc793625cf2"
# The tree every patch of this series produces on top of v0.32.0
# (replay-verified; see README.md).
EXPECTED_TREE="ffc95be46c72f2aec2d073a9234cd1e7703e949a"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# < 2 )); then
  echo "usage: $0 <path-to-spark-vllm-docker-checkout> <path-to-patched-vllm-checkout> [extra build-and-copy.sh flags]" >&2
  exit 1
fi

spark_docker="$1"
vllm_src="$2"
shift 2

if [[ ! -x "${spark_docker}/build-and-copy.sh" ]]; then
  echo "error: ${spark_docker} is not a spark-vllm-docker checkout (no build-and-copy.sh)" >&2
  exit 1
fi

spark_head="$(git -C "${spark_docker}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${spark_head}" != "${SPARK_VLLM_DOCKER_REF}" ]]; then
  echo "error: spark-vllm-docker checkout is at ${spark_head:-unknown}, this series" >&2
  echo "was validated against ${SPARK_VLLM_DOCKER_REF} (docker/patches/qwen3.8)." >&2
  echo "Check out that revision (git -C ${spark_docker} checkout ${SPARK_VLLM_DOCKER_REF})" >&2
  echo "or update vllm-0.32/ after revalidating." >&2
  exit 1
fi

if [[ ! -d "${vllm_src}" ]]; then
  echo "error: patched vLLM source not found at ${vllm_src}" >&2
  echo "       prepare it with (see the header of this script):" >&2
  echo "       git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm ${vllm_src}" >&2
  echo "       git -C ${vllm_src} checkout -b qwen38-flash-next-v0.32.0" >&2
  echo "       git -C ${vllm_src} am ${here}/*.patch" >&2
  exit 1
fi
if [[ -n "$(git -C "${vllm_src}" status --porcelain)" ]]; then
  echo "error: ${vllm_src} has uncommitted changes; build-and-copy.sh requires a clean tree" >&2
  exit 1
fi

tree="$(git -C "${vllm_src}" rev-parse 'HEAD^{tree}')"
if [[ "${tree}" != "${EXPECTED_TREE}" ]]; then
  echo "error: ${vllm_src} HEAD tree is ${tree}, expected ${EXPECTED_TREE}." >&2
  echo "       The checkout must be tag v0.32.0 plus this directory's series," >&2
  echo "       applied in series order ('git am ${here}/*.patch' from a fresh" >&2
  echo "       v0.32.0 produces exactly that tree)." >&2
  exit 1
fi

# The tree, not the commit: every replay of the series has the same tree but its own commit ids.
tag="qwen38-flash-next-fp4:vllm-0.32.0-${tree:0:12}"

echo "vLLM source : ${vllm_src} (HEAD tree ${tree})"
echo "spark-vllm-docker: ${spark_docker} @ ${spark_head}"
echo "image tag   : ${tag}"
echo

exec "${spark_docker}/build-and-copy.sh" --vllm-source-dir "${vllm_src}" -t "${tag}" "$@"

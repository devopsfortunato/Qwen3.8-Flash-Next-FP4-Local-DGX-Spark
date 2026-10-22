# check=skip=InvalidDefaultArgInFrom
# BASE_IMAGE has no default on purpose: scripts/config.sh owns the pinned digest and passes it in.
# The recipe's serve image: eugr/spark-vllm-b12x pinned by digest, with PLE_QUEUE_DEPTH and the optional
# VLLM_PLE_TIMING probe added. The probe changes nothing unless VLLM_PLE_TIMING=1 is set at runtime.
# scripts/prepare.sh builds it as qwen38-flash-next-fp4:<tag>; the tag is a hash of BASE_IMAGE and this
# directory, so any change here or to BASE_IMAGE builds a new image instead of reusing a stale one.
# By hand: docker build --build-arg BASE_IMAGE=eugr/spark-vllm-b12x@sha256:... -t NAME docker/
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The file the patch is written against, and its sha256 in the pinned base. The build stops if they differ.
ARG PLE_DISK_PY=/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/ple_embedding/_disk.py
ARG PLE_DISK_SHA256=2232d15d5034d9c0cfb9dc3bdbf0e3f49bcc619f51c9ccee77364a0553c6ddeb
ARG B12X_PLE_PY=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/b12x_ple.py
ARG B12X_PLE_SHA256=936872e2c7be1d8903e66a47a202a9cf244b0706589656e3576a09b9f3870b9c
ARG DISK_TABLE_PY=/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/_shared/disk_table.py
ARG DISK_TABLE_SHA256=601348d52d488057911b5a0b22d36f0e922d80cb956c2658901bee56e63e7c9a

COPY ple_queue_depth.py /tmp/ple_queue_depth.py
RUN python3 /tmp/ple_queue_depth.py "${PLE_DISK_PY}" "${PLE_DISK_SHA256}" \
 && rm /tmp/ple_queue_depth.py

COPY ple_timing.py /tmp/ple_timing.py
RUN python3 /tmp/ple_timing.py "${B12X_PLE_PY}" "${B12X_PLE_SHA256}" \
                      "${DISK_TABLE_PY}" "${DISK_TABLE_SHA256}" \
 && rm /tmp/ple_timing.py

ARG BASE_IMAGE
LABEL io.github.qwen38-flash-next-fp4.base="${BASE_IMAGE}" \
      io.github.qwen38-flash-next-fp4.ple-queue-depth="env PLE_QUEUE_DEPTH, default 256" \
      io.github.qwen38-flash-next-fp4.ple-timing="env VLLM_PLE_TIMING, default off"

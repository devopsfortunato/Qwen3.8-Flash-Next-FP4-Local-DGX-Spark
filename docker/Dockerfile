# check=skip=InvalidDefaultArgInFrom
# BASE_IMAGE has no default on purpose: scripts/config.sh owns the pinned digest and passes it in.
# The recipe's serve image: eugr/spark-vllm-b12x pinned by digest, with PLE_QUEUE_DEPTH added.
# scripts/prepare.sh builds it as qwen38-flash-next-fp4:<tag>; the tag is a hash of BASE_IMAGE and this
# directory, so any change here or to BASE_IMAGE builds a new image instead of reusing a stale one.
# By hand: docker build --build-arg BASE_IMAGE=eugr/spark-vllm-b12x@sha256:... -t NAME docker/
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The file the patch is written against, and its sha256 in the pinned base. The build stops if they differ.
ARG PLE_DISK_PY=/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/ple_embedding/_disk.py
ARG PLE_DISK_SHA256=2232d15d5034d9c0cfb9dc3bdbf0e3f49bcc619f51c9ccee77364a0553c6ddeb

COPY ple_queue_depth.py /tmp/ple_queue_depth.py
RUN python3 /tmp/ple_queue_depth.py "${PLE_DISK_PY}" "${PLE_DISK_SHA256}" \
 && rm /tmp/ple_queue_depth.py

ARG BASE_IMAGE
LABEL io.github.qwen38-flash-next-fp4.base="${BASE_IMAGE}" \
      io.github.qwen38-flash-next-fp4.ple-queue-depth="env PLE_QUEUE_DEPTH, default 256"

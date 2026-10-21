# Credits

This repository is a serving recipe: shell scripts that pull a container and a checkpoint and start vLLM. The model, the engine, and the kernels are other projects.

- **Alfonso Fortunato** ([MovieMaker93](https://github.com/MovieMaker93)) — this recipe's scripts and documentation.
- **vLLM** — the OpenAI-compatible server. Apache-2.0. https://github.com/vllm-project/vllm
- **eugr** — `eugr/spark-vllm-b12x` (nightly-20261015, pinned by digest) and [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), the DGX Spark image the recipe image is built on.
- **b12x** — the SM12x kernels the image loads with `--load-format b12x`, `--moe-backend b12x`, `--linear-backend b12x`, and `--gdn-decode-kernel b12x`.
- **NVIDIA Flare** — [flare-local-inference](https://github.com/NVIDIA/flare-local-inference), Apache-2.0: the engine compared in the README's Flare section. `flare/qwen38-mxfp8-lm-head.patch` modifies its Qwen recipe; `flare/Dockerfile.ignition` builds its preparation image from its own source.
- **NVIDIA** — the checkpoint [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) and NVIDIA Model Optimizer. The Spark flags in `scripts/config.sh` are the ones on that model card. Download terms: NVIDIA Open Model Agreement.
- **Qwen team (Alibaba)** — Qwen3.8-Flash-Next, the base model. Qwen Community License 1.0.
- **MiaAI-Lab** — the single-Spark TensorFold recipe whose launcher layout this one follows. Layout only. This recipe does not run TensorFold and does not reuse its measurements. https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold

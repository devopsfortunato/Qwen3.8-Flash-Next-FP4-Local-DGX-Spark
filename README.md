<h1 align="center">Qwen3.8 Flash Next FP4 Local</h1>
<p align="center">on one DGX Spark</p>
<p align="center">by <a href="https://github.com/devopsfortunato">Alfonso Fortunato</a> / <a href="https://github.com/devopsfortunato">@devopsfortunato</a></p>
<p align="center">
  <a href="https://github.com/devopsfortunato/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/actions/workflows/ci.yml"><img src="https://github.com/devopsfortunato/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/hardware-DGX%20Spark%20(GB10)-76B900" alt="DGX Spark">
</p>

Serve **nvidia/Qwen3.8-Flash-Next-FP4-Local** from one NVIDIA DGX Spark through an OpenAI-compatible API.
The server is vLLM, in the container `eugr/spark-vllm-b12x:latest`, with the DGX Spark flags from the
[NVIDIA model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): PLE (n-gram) tables on disk,
`--load-format b12x`, `--quantization modelopt_mixed`, FP8 KV cache, and MTP with 4 draft tokens.

The checkpoint is about 96 GiB on disk. NVIDIA's model card says a DGX Spark can run it with or without PLE
offload, and reports 1M tokens of FP8 KV cache with PLE offload. This recipe follows the model card's Spark
command: `VLLM_PLE_TABLE_MEMORY=disk` and a 262,144-token window.

`./start.sh` prepares the machine on the first run and starts the server. `./stop.sh` stops it.

> **Status:** early release. The scripts pass `shellcheck` and `scripts/test_config.sh` in CI, and the flags
> follow NVIDIA's model card for DGX Spark. Tokens/s numbers are not published yet. Please open an issue with your
> `docker logs` if a launch fails on your Spark.

## Quick start

```bash
git clone https://github.com/devopsfortunato/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark.git
cd Qwen3.8-Flash-Next-FP4-Local-DGX-Spark
./start.sh
```

The first run pulls the image and downloads the checkpoint into `~/.cache/huggingface` if it is not there yet.
Later starts load the weights. `start.sh` waits until `/v1/models` answers, runs a smoke test, prints
`nvidia/Qwen3.8-Flash-Next-FP4-Local is now LIVE! on port 8000`, and returns you to the shell.

Accept the model terms on the Hugging Face page first. If the Hub asks for a token, put it in the environment
or in `.env` (see `.env.example`):

```bash
HF_TOKEN=hf_... ./start.sh
```

```bash
curl -s http://<spark-address>:8000/v1/models

curl -s http://<spark-address>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
    "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
    "max_tokens": 1000
  }'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8000/v1"` and the model
`nvidia/Qwen3.8-Flash-Next-FP4-Local`. The recipe enables the Qwen3 reasoning parser and the `qwen3_xml` tool-call
parser, so streaming, tool calls, and reasoning content are the vLLM Qwen3 paths. The model thinks before it
answers; give replies enough `max_tokens`. `"chat_template_kwargs": {"enable_thinking": false}` answers without
a thinking pass.

```bash
./start.sh restart
./stop.sh
DRY_RUN=1 ./start.sh
docker logs -f qwen38-flash-next-fp4
```

`DRY_RUN=1` prints the `docker run` command and does not pull, download, stop, or start anything.
`HF_TOKEN` is passed into the container by name, so the dry-run line does not contain the token.

## Requirements

- One DGX Spark (GB10, 128 GB unified memory) with room to load the model. `start.sh` wants about 90 GiB
  `MemAvailable` before launch (`MEM_NEED_GIB`). `MEM_CHECK=0` turns that into a warning.
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- Disk for a fresh machine: about 96 GiB for the checkpoint under `HF_CACHE` and space for the image under
  Docker's root. `scripts/prepare.sh` checks both (`MIN_FREE_GB` 120 when the snapshot is absent,
  `IMAGE_FREE_GB` 30 when the image is not local).
- Optional: the `hf` CLI on the host. Without it, `prepare.sh` downloads from inside the container.
  A Hugging Face token in `HF_TOKEN` or `~/.cache/huggingface/token` when the Hub requires one.

## What `start.sh` and `scripts/prepare.sh` do

1. **Setup.** `scripts/prepare.sh` when the image or the checkpoint is not the one these settings name.
2. **Checks.** Settings are validated first, so a bad `PORT` or `MTP_TOKENS` does not stop a server that is
   already up. `./start.sh restart` then stops the previous container, and only after that checks that
   the port is free and that `MemAvailable` covers `MEM_NEED_GIB`. Start, restart, and stop share a lock
   (`flock` on `STATE_DIR/lifecycle.lock`). `./stop.sh` waits up to `STOP_LOCK_WAIT` seconds (30) for
   that lock and then exits without stopping anything.
3. **Launch.** `docker run` with `--network host`, `--gpus all`, the model-card environment, and `vllm serve`.
4. **Loading.** Wait until `/v1/models` answers. If `MemAvailable` falls below `MEM_FLOOR_GIB` (8 GiB), the
   container is stopped.
5. **Smoke test**, then the LIVE line.

`FOREGROUND=1 ./start.sh` stays attached to the container log and skips the wait. Extra arguments are appended
to `vllm serve` (`./start.sh restart --max-model-len 131072`).

**`scripts/prepare.sh`:**

1. Preflight: Docker, disk, the memory floor.
2. `docker pull` of `IMAGE` (`PULL=0` uses a local image and does not pull).
3. The checkpoint, into `HF_CACHE`.
4. A check that it is `qwen4_exp`, ModelOpt `MIXED_PRECISION`, and that every shard named by the index is present.

```bash
scripts/prepare.sh
PREPARE=1 ./start.sh restart
PULL=0 scripts/prepare.sh
```

## Configuration

Every setting lives in `scripts/config.sh`. Override it from the environment (`PORT=9000 ./start.sh`) or from
`.env` next to `start.sh` (`KEY=value`; the environment wins). `.env` is read as data. It is not executed.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | Hugging Face checkpoint |
| `IMAGE` | `eugr/spark-vllm-b12x:latest` | vLLM container with the B12X kernels |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | where the API listens |
| `SERVED_NAME` | the model id | id returned by `/v1/models` |
| `MAX_MODEL_LEN` | `262144` | prompt + reply window (`--max-model-len`) |
| `MAX_NUM_SEQS` | `8` | sequences decoded together |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | chunked-prefill token cap |
| `GPU_MEMORY_UTILIZATION` | `0.8` | fraction of GPU memory vLLM may use |
| `KV_CACHE_DTYPE` | `fp8` | KV cache dtype |
| `QUANTIZATION` | `modelopt_mixed` | `--quantization` |
| `LOAD_FORMAT` | `b12x` | `--load-format` |
| `MTP_TOKENS` | `4` | MTP draft tokens; `0` omits `--speculative-config` |
| `BLOCK_SIZE` | `16` | KV block size |
| `DTYPE` | `bfloat16` | model dtype |
| `VLLM_PLE_TABLE_MEMORY` | `disk` | n-gram table on disk, as on the model card |
| `B12X_DISK_BACKEND` | `io_uring` | how that table is read |
| `CUTE_DSL_ARCH` | `sm_121a` | GB10 architecture for the CuTe DSL kernels |
| `HF_CACHE` | `~/.cache/huggingface` | where the checkpoint is stored |
| `MEM_NEED_GIB` | `90` | `MemAvailable` `start.sh` asks for before launch |
| `MEM_FLOOR_GIB` | `8` | stop the load if free memory falls below this |
| `MIN_FREE_GB` | `120` | disk `prepare.sh` asks for when the snapshot is absent |
| `IMAGE_FREE_GB` | `30` | disk asked for when the image is not local |
| `CONTAINER_NAME` | `qwen38-flash-next-fp4` | docker name |
| `WAIT_TIMEOUT` | `1800` | seconds to wait for `/v1/models` |
| `PREPARE` | `auto` | `1` always runs prepare, `0` skips it |
| `PULL` | `1` | `docker pull` during prepare |
| `HF_HUB_OFFLINE` | `1` | serve container; prepare turns it off for the download |

The container also receives the rest of the model-card Spark environment:
`SAFETENSORS_FAST_GPU=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_SSM_CONV_STATE_LAYOUT=DS`,
`VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_MEGA_AOT_ARTIFACT=1`, `VLLM_USE_V2_MODEL_RUNNER=1`,
`B12X_POLICY_MODE=auto`, `VLLM_MXFP8_LM_HEAD=1`, `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`,
`VLLM_B12X_MOE_FP4_FORCE_A16=1`, `B12X_W4A16_A4_PREFILL=1`.

`vllm serve` is started with `--enable-prefix-caching`, `--enable-chunked-prefill`,
`--mamba-cache-mode align`, `--gdn-decode-kernel b12x`, `--linear-backend b12x`, `--moe-backend b12x`,
`--no-enable-flashinfer-autotune`, `--mm-encoder-tp-mode data`, `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_xml`, `--enable-auto-tool-choice`, and
`--compilation-config '{"pass_config":{"fuse_act_quant":true}}'`.

`DRY_RUN=1 ./start.sh` prints the exact `docker run` line for the current settings.

## Troubleshooting

- **401 / gated repo during download.** Accept the terms on the
  [model page](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), then set `HF_TOKEN`.
- **`MemAvailable` too low.** Stop other containers or models (`docker ps`), drop page cache with
  `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`, or run with `MEM_CHECK=0` to only warn.
- **Port 8000 already in use.** `PORT=9000 ./start.sh`, or `./start.sh restart` if it is this recipe's container.
- **Out of memory while loading.** Lower `GPU_MEMORY_UTILIZATION` (for example `0.75`) or `MAX_MODEL_LEN`
  (`131072`).
- **Slow first start.** The first load compiles kernels and reads about 96 GiB; later starts are faster. Follow it
  with `docker logs -f qwen38-flash-next-fp4`.
- **Unknown quantization or load format.** You are not on `eugr/spark-vllm-b12x`; upstream vLLM cannot read this
  checkpoint's NVFP4 scales yet. Keep the default `IMAGE`.

## Checks

`scripts/test_config.sh` checks the defaults, `.env` versus the environment, and `build_docker_cmd`. It does
not need Docker, a GPU, or the network.

```bash
./scripts/test_config.sh
```

## Repository layout

```
start.sh                 prepare if needed, then serve
stop.sh                  stop and remove the container
scripts/config.sh        settings and build_docker_cmd
scripts/prepare.sh       pull the image, download the checkpoint
scripts/test_config.sh   config checks
.env.example             commented overrides
CREDITS.md               what this builds on
NOTICE                   third-party notices
LICENSE                  MIT
```

## License

MIT, Copyright 2026 Alfonso Fortunato. See `LICENSE`. The MIT license covers this repository's scripts and
documentation.

The checkpoint downloaded by `scripts/prepare.sh` is not part of this repository.

- **NVIDIA Open Model Agreement** governs download and use of `nvidia/Qwen3.8-Flash-Next-FP4-Local`.
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
- **Qwen Community License 1.0** governs the Qwen3.8-Flash-Next base model.
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE

The container `eugr/spark-vllm-b12x:latest` is pulled, not built here. Its NVIDIA software and vLLM build stay
under their own licenses. See `NOTICE`.

## Credits

- [vLLM](https://github.com/vllm-project/vllm) serves the model.
- [eugr/spark-vllm-b12x](https://github.com/eugr/spark-vllm-docker) is the container.
- b12x supplies the SM12x kernels that container loads.
- [MiaAI-Lab's single-Spark TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
  is the layout this launcher follows. Layout only: this recipe does not run TensorFold.

The full list is in `CREDITS.md`.

# Upstream vLLM v0.32.0 lane for Qwen3.8-Flash-Next-FP4-Local

This directory carries the patch series that makes
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (model_type `qwen4_exp`, ModelOpt
MIXED_PRECISION, NVFP4 block scales in the lossless CSF compression, PLE
n-gram tables, embedded MTP draft) load and run on **upstream vLLM
v0.32.0** (tag commit `5df63a7203cdb9328d824a27b682e77c7b9928be`) on a DGX
Spark, plus the build entry point for the runner image.

The series is the regular-lane qwen3.8 patch stack from eugr's
`spark-vllm-docker`, revision **`eadf57da73953fe71326dbd10ef45fc793625cf2`**
(2026-10-12, "fix loader" — the Oct 12 refresh onto vLLM
`fe341a53dda00472c281a2d1f4bd34a74e7eae6e` plus the same-day
host-file-gather loader fix). That revision still bundles #56273's packed
NVFP4 PLE support in patch 01, which v0.32.0 lacks (#56273 merged upstream
on Oct 14, after the v0.32.0 cut). The patches were applied onto tag
v0.32.0 one commit per patch, in order, on branch
`qwen38-flash-next-v0.32.0` of the sibling checkout
`../vllm-v0.32.0-qwen38`; the `.patch` files here are exported from those
commits (`git format-patch`), so they replay onto a clean v0.32.0 with the
v0.32.0 conflict resolutions included (numbering follows eugr's
`docker/patches/qwen3.8/vllm/series`; there is no 12, matching upstream).

## The series

| patch | upstream PR | what it does | required? |
|---|---|---|---|
| 01-pr59958 | #59958 + #56273 + #58815 | NVFP4 PLE n-gram tables; engram `host_file_gather` offload; byte-view block scales | **Required** — the checkpoint ships PLE tables; packed NVFP4 PLE is not in v0.32.0 |
| 02-pr59973 | #59973 (PR head `0c2ed06a`) | quantized draft lm_head for drafters sharing the target head | **Required** — the MTP draft head is quantized (#59973 merged as `784d1fff`, after the cut) |
| 03-pr59740-60226 | #59740 + #60226 | context-aware sparse lm_head (`draft_vocab.py`), quantized sparse heads | Optional at runtime (only with `draft_token_map`, which the profile does not set); kept because the later patches were exported on top of it |
| 04-pr60068 | #60068 | opt-in confidence stop for autoregressive drafting | Optional (off by default) |
| 05-pr54912 | #54912 | widen QSA raw-key ring instead of asserting divisibility | Optional (QSA bugfix; not exercised by this checkpoint's serve path) |
| 06-pr50897 | #50897 (`ff08350f`) | lookahead-aware prefix cache hashing | Optional perf (opt-in) |
| 07-pr54614 | #54614 | dynamic NVFP4 W4A4/W4A16 linear kernel selection | Optional perf |
| 08-pr54617 | #54617 | per-forward NVFP4 W4A4/W4A16 selection for routed experts (SM12x) | Optional perf |
| 09-pr60252 | #60252 | keep SW draft block a divisor of the primary block | Optional (DFlash2 sizing bugfix) |
| 10-pr59021 | #59021 | steer the unquantized linear backend independently (`--linear-backend`) | Optional perf knob (the flag itself exists in v0.32.0) |
| 11-pr59025 | #59025 | register the cudnn backend for FlashInfer BF16 mm | Optional perf |
| 13-draft-vocab | local (Flash Next mod asset) | 32,768-token Qwen3.8 Flash Next draft vocabulary JSON | Optional — only for a sparse MTP draft head (`draft_token_map` in `--speculative-config`); the fork image has no draft vocabulary and the profile does not set one |
| 14-pr60646 | #60646 (`00b5ad07`) | decode CSF-compressed NVFP4 block scales at load | **Required** — the core blocker: the checkpoint's NVFP4 scales ship CSF-compressed and stock v0.32.0 cannot decode them |
| 15-package-draft-vocab | local | `setup.py` package-data rule so the wheel ships the draft vocabulary | Optional — goes with 13 |
| 16-flash-next-fp4 | local (mod `mods/draft/flash-next-fp4`) | ModelOpt mixed-precision dispatch, nested NVFP4 scale loading, opt-in resident CSF expert scales (`VLLM_CSF_RESIDENT`), weight-transfer helpers, post-load host-heap trim | **Required** — ModelOpt mixed/nested-scale loading and the qwen4_exp helpers the loader path uses |
| 17-flash-next-b12x-loader | local (adapts `local-inference-lab/vllm` `3a947437`) | opt-in `--load-format b12x` checkpoint loading via FlashInfer's reader | Optional — opt-in load path; safetensors loading (with CSF decode) works without it |
| 18-b12x-nvfp4-a16-k32 | local to this recipe | pad the stored K of b12x NVFP4 W4A16 weights to a multiple of 32, keeping the logical K | **Required** — without it the vision encoder's MLP down projection (K=4304) stops the first boot with `A16 requires stored K32 and N8` |
| 19-pr58449 | #58449 (`848dbf89`) | fused draft metadata updates for the QSA side cache: the MTP drafter no longer rebuilds QSA attention metadata eagerly on every draft step | **Required for speed** — decode 38.64 → 40.47 t/s |
| 20-mxfp8-lm-head | local (ports the fork's `VLLM_MXFP8_LM_HEAD`) | the BF16 LM head loads as shipped and is stored and run as MXFP8 (b12x, BF16 activations); target and MTP heads | **Required for speed** — decode 40.47 → 45.45 t/s; on with `VLLM_MXFP8_LM_HEAD=1` (the profile keeps it) |
| 21-pr58863-recoverssm | #58863 (open) | RecoverSSM for Qwen GDN and the PLE short conv: drafts verify off one GDN state, only accepted tokens are replayed | Optional, off — `--use-replayssm` frees KV memory (1,955,533 instead of 1,751,020 tokens); single-stream decode unchanged |

## Conflicts resolved against v0.32.0

One real conflict: patch 16 inserts its CSF-resident `RoutedExperts` dispatch
in `vllm/model_executor/layers/quantization/modelopt.py::get_quant_method`,
exactly where v0.32.0's #59604 added the b12x block-codec dispatch
(`B12X_BLOCK_CODECS`). Both branches are kept, CSF branch first: their
conditions are disjoint (CSF-encoded NVFP4 weights vs IQ2/Q8_0 codecs).
Everything else applied via three-way merge or direct apply without manual
resolution.

Patch 21: `vllm/config/cache.py`, where v0.32.0 still has `use_kda_recoverssm`;
the PR's rename to `use_recoverssm` is taken together with its Kimi-K3 call
sites. The PR's `vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py`
does not exist in v0.32.0 and is skipped (Kimi-K3 only).

## FlashInfer

v0.32.0 pins `flashinfer-python==0.7.0.post1` (released 2026-09-29). That
release predates FlashInfer #5863 (canonical NVFP4 W4A4 MoE, merged
2026-10-16) and #5943 (SM12x CuTe DSL MXFP8 GEMMs, still open), so the pin
does **not** contain the kernels this checkpoint needs. The build uses eugr's
regular lane: prebuilt FlashInfer wheels from his
`prebuilt-flashinfer-current` release (built from upstream FlashInfer main +
`docker/patches/qwen3.8/flashinfer/{01-pr5863,02-pr5943}.patch`; validation
base `b77de25d433ff5082247b8e306c3aa54857e3ed7`). See `flashinfer-series`.
The runner installs FlashInfer wheels beside vLLM's under a pip `--override`
file, so the 0.7.0.post1 pin does not downgrade them.

## Building the image

On the Spark (or any machine with docker + the two checkouts):

```bash
vllm-0.32/build.sh <path-to-spark-vllm-docker-checkout> <path-to-patched-vllm-checkout>
# -> qwen38-flash-next-fp4:vllm-0.32.0-85567e46546a (12-hex prefix of the patched source tree)
```

The patched vLLM checkout is any clean Git checkout of tag v0.32.0 with this
series applied on top; `build.sh` verifies it by content (`HEAD^{tree}` must
be `85567e46546a84221d484a10eb742cb5db4cfe1a`, which a fresh v0.32.0 plus
`git am vllm-0.32/*.patch` produces exactly), not by branch name or path.
Minimal source preparation from scratch:

```bash
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm vllm-v0.32.0-qwen38
git -C vllm-v0.32.0-qwen38 checkout -b qwen38-flash-next-v0.32.0
git -C vllm-v0.32.0-qwen38 am vllm-0.32/*.patch
```

`build.sh` refuses to run against a different `spark-vllm-docker` revision
than `eadf57d` and requires a clean source tree. Extra flags are forwarded to
`build-and-copy.sh` (e.g. `--copy-to`, `--rebuild-flashinfer`). The preset
vLLM patch series is skipped automatically for `--vllm-source-dir` builds —
the series is already applied.

## Serving

```bash
PROFILE=vllm-0.32 ./start.sh
```

The profile names that image and sets `PULL=0`: the tag exists only where
`build.sh` ran.

`profiles/vllm-0.32.env` maps the fork-only settings to their regular-lane
equivalents:

| fork lane | v0.32 lane |
|---|---|
| `VLLM_PLE_TABLE_MEMORY=disk` | `--engram-config '{"host_file_gather":true}'` |
| `--gdn-decode-kernel b12x` | dropped (flag does not exist upstream) |
| `VLLM_MXFP8_LM_HEAD=1` | kept: patch 20 implements it on this lane |
| `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13` | dropped (fork-only) |
| `B12X_POLICY_MODE`, `B12X_DISK_BACKEND`, `B12X_W4A16_A4_PREFILL*`, `VLLM_B12X_DENSE_ACTIVATION_MODE`, `PLE_QUEUE_DEPTH` | dropped (fork-only; `PRECISION=quality` has no upstream equivalent) |
| `{"method":"mtp","num_speculative_tokens":3}` | unchanged: full-vocabulary MTP draft, as in the fork image |

Kept because they exist in v0.32.0 and apply unchanged:
`CUTE_DSL_ARCH=sm_121a` (read by #5943's CuTe DSL pool),
`VLLM_B12X_MOE_FP4_FORCE_A16=1`, `VLLM_USE_AOT_COMPILE=1`,
`VLLM_USE_MEGA_AOT_ARTIFACT=1`, `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_SSM_CONV_STATE_LAYOUT=DS`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`SAFETENSORS_FAST_GPU=1`, and the flags `--quantization modelopt_mixed`,
`--moe-backend b12x`, `--linear-backend b12x`, `--load-format b12x` (opt-in,
see patch 17), `--mamba-cache-mode align`, `--kv-cache-memory-bytes`.

Optional on this lane: `EXTRA_ENV=VLLM_CSF_RESIDENT=1` opts into resident CSF
expert scales (patch 16; requires the checkpoint's per-layer
`weight_scale_encoding: csf` metadata). `VLLM_TRIM_HOST_HEAP_AFTER_LOAD=0`
disables the post-load host-heap trim.

## Known gaps vs the fork image

- `--gdn-decode-kernel` GDN decode kernel choice and the fork's
  `VLLM_PLE_TABLE_MEMORY` selector (including its `io_uring` disk backend and
  the recipe image's `PLE_QUEUE_DEPTH` tuning) have no upstream counterpart.
- `PRECISION=quality` (two-term prefill, BF16 dense activations) is
  fork-only; the upstream lane always runs the default fast path.
- Numerics differ from the fork image: no FP4 prefill activations
  (`B12X_W4A16_A4_PREFILL`) and upstream kernels in place of the fork's.
  Logprobs are not bit-identical to the pinned-image recipe; see the
  measurements below.
- `--load-format b12x` (patch 17) supports TP 1–2 without EP/DP/pipeline/EPLB,
  sleep mode, or standalone weight reload, and rejects other architectures.
- Patch 17's `read_mode` accepts `auto`/`bounce` only; the fork's shared-GDS
  path is not exposed.

## Verification status

Measured on one DGX Spark (GB10) against the pinned recipe image
(`qwen38-flash-next-fp4:61ab709967e0`, eugr nightly-20261015), one stream,
MTP 3, same campaign scripts (`pbench` 8k/16k/32k cold prefill, `dbench`
512 greedy tokens on three prompts, `ppl.py` + `cmp.py` on the recipe's
perplexity texts). Steps measured with Python overlays on the image built
from patches 1–18, before the final rebuild from the full series:

| image | prefill t/s | decode t/s | KV tokens (31.15 GB) | perplexity prose+code / fresh / 96k book | vs pinned: mean \|Δlogprob\| / top-1 agree |
|---|---:|---:|---:|---|---|
| pinned fork image | 2,411 | 45.58 | 1,575,180 | 1.0902 / 2.2049 / 1.0470 | — |
| patches 1–18 | 2,598 | 38.64 | 1,751,020 | 1.0868 / 2.2034 / 1.0439 | 0.042 / 98.46% (own repeat noise 0.047) |
| + 19 | 2,600 | 40.47 | 1,751,020 | — | — |
| + 19 + 20 (the profile) | 2,633 | 45.45 | 1,751,020 | 1.0872 / 2.2090 / 1.0454 | 0.043 / 98.44% (own repeat noise 0.050) |
| + 19 + 20 + 21 `--use-replayssm` | 2,586 | 45.75 | 1,955,533 | — | 0.042 / 98.49% |

**The image built from this series** (`qwen38-flash-next-fp4:vllm-0.32.0-ffc95be46c72`, `PROFILE=vllm-0.32`):
prefill 2,596 t/s, decode 45.23 t/s, 1,751,020 KV tokens, perplexity 1.0877 / 2.2158 / 1.0444, against the
pinned image 0.042 mean |Δlogprob| and 98.49% top-1 agreement (own repeat noise 0.048).

| quality | pinned image | this lane, MXFP8 LM head | this lane, BF16 LM head |
|---|---|---|---|
| tool-eval-bench 2.8.1, 97 scenarios (points) | 89 · 90 (172 · 175) | 88 · 87 (171 · 169) | 88 · 87 (170 · 169) |
| sixcat-eval 0.8.3, 884 items | 92.1 | 90.3 · 90.8 | — |

Prompts are not the cause: eight tool-calling conversations (tool results, parallel calls, reasoning history,
thinking on and off) render to identical token ids through `/tokenize` on both images, and the tool parser
is the same file. The failing scenarios move with the numerics: switching only the LM head between MXFP8 and
BF16 makes TC-21 and TC-74 pass and TC-87 and TC-91 fail, at the same total. The sixcat gap is its Tools
category, 2 of 20 items, on all runs of this lane. Every run of this lane scores 87-88 on tool-eval and every
run of the pinned image 89-90: a small systematic difference cannot be excluded.

llm_context_benchmarks, one request, 512 output tokens, 2 runs, same evening (prefill / decode t/s):

| context | prose, pinned | prose, this lane | code, pinned | code, this lane |
|---|---|---|---|---|
| 0.5k | 1,057 / 46.4 | **1,252** / 47.5 | 1,073 / 57.0 | **1,279** / 54.0 |
| 8k | 2,086 / 42.4 | **2,343** / 41.7 | 2,078 / 56.3 | **2,371** / 55.6 |
| 32k | 2,174 / 41.3 | **2,373** / 39.9 | 2,187 / 51.3 | **2,414** / 58.0 |
| 128k | 2,100 / 39.2 | **2,247** / 39.7 | 2,138 / 46.3 | **2,322** / 45.2 |

Code decode depends on how long each answer runs (166 to 512 tokens), so those cells are noisy.

Quality on patches 1–18 (before 19/20): tool-eval-bench 89 · 88, sixcat-eval 90.7.

Tried on this lane and left out (no gain at one stream, output unchanged):
Marlin MoE (39.78 t/s), `FULL_AND_PIECEWISE` CUDA graphs (39.99),
`--async-scheduling` (38.88), `--max-num-batched-tokens 8192` (39.33), all on
patches 1–19. Turning off upstream's QSA prepare fusion and merged QSA
projections leaves prefill at 2,599 t/s and costs decode (37.42), so they are
not where this lane's prefill lead comes from. Neither are the GDN prefill
kernel, the fork's FP4 prefill activations or its PLE queue depth (512 and
1024 give 2,398 and 2,409 t/s on the pinned image); the PLE table in memory
does not fit on either image next to the weights.

Source checks (x86): the series replays onto a fresh
`git clone --depth 1 --branch v0.32.0` with `git am` and reproduces tree
`ffc95be46c72f2aec2d073a9234cd1e7703e949a` (also on the Spark); changed `.py`
files byte-compile; the CPU loader/CSF/draft-vocab suites pass (36 passed, 5
CUDA-only skips; 10 `test_draft_vocab.py` cases need vLLM's GPU conftest).

Not verified: `VLLM_CSF_RESIDENT=1` (resident CSF expert scales; eugr marks
it GPU-unvalidated, the profile leaves it off); RecoverSSM under concurrency,
where its PR reports its gains; `--load-format b12x` beyond this checkpoint.

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

## Conflicts resolved against v0.32.0

One real conflict: patch 16 inserts its CSF-resident `RoutedExperts` dispatch
in `vllm/model_executor/layers/quantization/modelopt.py::get_quant_method`,
exactly where v0.32.0's #59604 added the b12x block-codec dispatch
(`B12X_BLOCK_CODECS`). Both branches are kept, CSF branch first: their
conditions are disjoint (CSF-encoded NVFP4 weights vs IQ2/Q8_0 codecs).
Everything else applied via three-way merge or direct apply without manual
resolution.

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
| `VLLM_MXFP8_LM_HEAD=1` | dropped (fork-only): the target LM head runs in BF16; the README's precision table measured that at about 6% slower decode on the fork |
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
  (`B12X_W4A16_A4_PREFILL`), no MXFP8 LM head, and upstream kernels in place of
  the fork's. Logprobs will not be bit-identical to the pinned-image recipe;
  compare perplexity and the task benchmarks instead.
- `--load-format b12x` (patch 17) supports TP 1–2 without EP/DP/pipeline/EPLB,
  sleep mode, or standalone weight reload, and rejects other architectures.
- Patch 17's `read_mode` accepts `auto`/`bounce` only; the fork's shared-GDS
  path is not exposed.

## Verification status

Verified on x86, no GPU in this environment:

- All 16 patches apply onto tag v0.32.0 as one commit each; branch
  `qwen38-flash-next-v0.32.0` in `../vllm-v0.32.0-qwen38`, tree
  `85567e46546a84221d484a10eb742cb5db4cfe1a`, `git status` clean.
- All 123 changed `.py` files byte-compile (`python3 -m compileall -q`).
- Replays verified twice: (1) `git apply --3way` of the exported `series`
  onto a fresh v0.32.0 worktree reproduces the branch tree exactly; (2) a
  fresh `git clone --depth 1 --branch v0.32.0` from GitHub plus
  `git am vllm-0.32/*.patch` applies all 16 patches with no conflicts and
  produces the same tree hash.
- CPU test suites with torch 2.13.0+cpu, safetensors, numpy, pytest
  (`--noconftest --import-mode=importlib`, eugr's documented invocation):
  `tests/model_executor/model_loader/test_flash_next_b12x_loader.py` +
  `tests/model_executor/model_loader/test_csf_scales.py` +
  `tests/model_executor/layers/test_draft_vocab.py` → **36 passed, 5
  skipped** (the skips are the CUDA/native-reader cases).
  spark-vllm-docker's own applier regression suite at the pinned commit
  (`tests/test_regular_patch_series.py`) → **16 passed, 71 subtests passed**.
- The 10 `test_draft_vocab.py` errors in that run are all
  `fixture 'default_vllm_config' not found`: the fixture lives in vLLM's
  shared `tests/conftest.py`, which `--noconftest` deliberately excludes.
  Re-running the file WITH the conftest gets past the fixture and then fails
  at vLLM platform init ("Cannot access accelerator device when none is
  available") — these cases initialize the GPU platform and are not runnable
  on a CPU-only host; they are not failures of the series code. (The conftest
  route also needs vLLM's full server dependency set; that was installed and
  is what moved the errors from fixture-missing to accelerator-missing.)

Not verified (needs the Spark):

- Docker image build (vLLM + FlashInfer wheel compilation), checkpoint load,
  inference, and performance. eugr's README states the same about his lane:
  GPU build/inference remain for CI.
- `VLLM_CSF_RESIDENT=1` (resident CSF expert scales): eugr's README marks
  GPU execution of that path unvalidated; the profile leaves it off and the
  default loader decodes CSF scales through #60646 instead.
- `--load-format b12x` (patch 17) is validated by its CPU loader tests only;
  the native-reader case needs Spark CUDA + FlashInfer.
- The FlashInfer prebuilt wheel set actually downloaded on the Spark
  (`prebuilt-flashinfer-current`) is assumed to contain #5863/#5943 as eugr's
  README describes; not verifiable from here.

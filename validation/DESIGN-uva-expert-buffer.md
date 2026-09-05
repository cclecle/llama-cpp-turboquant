# Qwen3.8-Flash-Next: MTP re-measured, decode anatomy, and the UVA expert buffer type

Date: 2026-09-05. Branch `rebase/upstream-20260904` (upstream `8b4b3558f` + fork commits), all of the
work below is in the working tree, uncommitted. Box: 2x Radeon AI PRO R9700 (gfx1201, 31.86 GiB
usable each, each behind a PCIe switch whose upstream port is Gen5 x8), ROCm 7.2.4, 110 GiB RAM.
Model: Qwen3.8-Flash-Next UD-IQ4_XS (arch `qwen4exp`, 48 layers, 512 experts / 10 used, 12 full
attention layers, 26.8 GiB PLE table kept on the CPU).

Builds on the box:
- v12 `/opt/llamacpp/llama-cpp-mine-v12` - production, untouched.
- v13 `/opt/llamacpp/llama-cpp-mine-v13` - the rebase + unsloth MTP + the is_mem_shared fix. Reference only.
- v14 `/opt/llamacpp/llama-cpp-mine-v14` - v13 + the UVA expert buffer type. CMakeCache identical to v12/v13
  (full diff), RCCL linked. Candidate for validation, NOT deployed.

Production stayed on v12 throughout; the three units were stopped only during measurements and
restarted after. The preset store `/opt/llamacpp-config` was not modified.

## 1. What was designed

### 1.1 The is_mem_shared fix (common/speculative.cpp)

The MTP driver decides "the draft shares the target KV" with `llama_get_ctx_other(ctx_dft) == ctx_tgt`.
Unsloth's shared draft head sets `ctx_other` for a different reason (to borrow the target's `token_embd`
and `output`), so it took the gemma4 branch that reuses one position for every draft token, which M-RoPE
rejects (`X < Y`). Result before the fix: 25,790 failed draft steps, 21% acceptance.

The predicate is now "ctx_other is the target AND the draft arch is `gemma4-assistant`", because the
gemma4 assistant is a separate `-md` model and is the only arch whose KV cache mirrors the target
(`mem_other` in llama-model.cpp). A model-identity check (`llama_get_model(ctx_dft) == llama_get_model(ctx_tgt)`)
would have broken the fleet's gemma-4-31B MTP. The same line exists upstream in PR #28243.

### 1.2 The UVA expert buffer type (ggml-cuda, meta backend, loader)

Problem: experts that do not fit in VRAM are today placed on the CPU (`--n-cpu-moe N`). Each CPU expert
layer splits the GPU graph, costs a host sync, and at prefill the scheduler copies the layer's weights
to the GPU for every ubatch. Measured on this model: the FIRST CPU layer costs 3.5 ms per token, each
further one about 0.9 ms; six CPU layers are ~9 ms of a 38 ms token.

Design: a per-GPU buffer type `ROCm{N}_UVA` whose buffers are pinned host memory mapped into the
device address space (`hipHostMalloc(Mapped | Portable)`, device pointer from `hipHostGetDevicePointer`,
equal to the host pointer on ROCm). The CUDA backend treats it as memory of that device, so `MUL_MAT_ID`
stays on the GPU graph and the stock mmvq (decode) and mmq (prefill) kernels read the experts in place
over PCIe. Nothing is copied per token, the graph never splits, the CPU computes no experts. Experts
live in exactly one place (host RAM); there is no VRAM copy.

Files:
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_backend_cuda_uva_buffer_type(device)`; the buffer shares the
  CUDA buffer interface with a `host_ptr` set, so set/get/memset/clear go through host memcpy;
  `supports_buft` and the `supports_op` source-device check accept it; exposed via the reg proc address
  `ggml_backend_dev_get_extra_bufts`. Env `GGML_CUDA_UVA_NONCOHERENT=1` adds `hipHostMallocNonCoherent`
  (measured: no difference).
- `ggml/include/ggml-cuda.h`, `vendors/hip.h`, `vendors/musa.h`: declaration and the host-alloc aliases.
- `ggml/src/ggml-backend-meta.cpp`: `ggml_backend_meta_buffer_type_from_simple(meta_dev, ROCm0_UVA)`
  returns `Meta(ROCm0_UVA,ROCm1_UVA)`, a meta buffer type over each simple device's matching extra buffer
  type. Under `--split-mode tensor` the expert tensors keep their normal row split; each half lives in the
  host memory mapped to its own GPU. Nothing else in the tensor-split path changes.
- `src/llama-model-loader.cpp`: an `-ot` override naming a per-device buffer type on a layer that lives on
  a meta device is mapped to that meta buffer type.
- `common/arg.cpp`, `tools/llama-bench/llama-bench.cpp`: the `-ot` name lookup includes the extra buffer types.
- `ggml/include/ggml-backend.h`: the new public function.

Usage (ONE `-ot`, llama.cpp keeps only the last `-ot` on the command line):

```
--n-cpu-moe 0 -ot 'per_layer_token_embd=CPU,blk\.(0|1|2|3|4|5)\.ffn_(gate|up|down)_exps\.weight=ROCm0_UVA'
```

Same layers as `--n-cpu-moe 6` (the first N), same VRAM (mapped host memory costs no VRAM).

The generic MoE hybrid kernel (`ggml-cuda/moe-hybrid.cu`, `ggml-moe-hybrid.cpp`, `ggml-cpu/mul-mat-id-hybrid.cpp`)
is byte-identical to the v12 commit. It only intercepts tensors on `CPU`/`CPU_Mapped` buffers, so it never
sees a UVA tensor; the two can coexist on different layers of one model.

### 1.3 Two bugs fixed on the way

- Every meta buffer type was named `Meta()` because the name was built from a vector that had already been
  moved from (same for the meta device name). The model loader keys its tensor contexts by buffer-type NAME,
  so tensors overridden to a second meta buffer type silently merged into the default one. Names are now
  `Meta(ROCm0,ROCm1)` and `Meta(ROCm0_UVA,ROCm1_UVA)`.
- `llama-perplexity` appeared to segfault under tensor split on v13/v14. It was a full root disk: a
  `--kl-divergence-base` file is ~0.47 GB per 2048-token chunk with this 248k vocabulary, and the HIP
  runtime segfaults when it cannot write its kernel cache. Base files go to `/mnt/cache`.

## 2. What was measured

Harness `/root/mtpbench.sh` (v2) on the box: one server per row, args mirror the preset store's `[*]`
block plus the model block, `--parallel 1`, c=131072, greedy, three reps with distinct prompts. Cells:
prose and code chat completions with a natural stop (cap 4096; the thinking model usually hits the cap),
and a 3.2k-token source-text prompt with `ignore_eos` at 512 and 2048 tokens. Medians of 3.

### 2.1 MTP, honest protocol (v13, CPU expert layers)

| config | prose t/s | code t/s | acceptance / mean len | VRAM GB |
|---|---:|---:|---|---|
| nospec nc6 (production) | 26.4 | 26.3 | - | 33.64 / 32.92 |
| nospec nc12 | 24.2 | 24.0 | - | 30.02 / 29.29 |
| shared head n2 nc12 | 38.5 | 39.8 | .53-.78 / 2.1-2.6 | 32.43 / 32.23 |
| shared head n2 nc10 | 39.8 | 43.2 | .53-.78 / 2.1-2.6 | 33.38 / 33.66 |
| shared head n2 nc9 | 40.8 | 43.6 | | 34.10 / 34.15 (60 MiB free, rejected) |
| self head n8 nc12 | 24.0 | 23.9 | .21-.37 | loss |
| self head n12 nc12 | 18.8 | 21.9 | .16-.25 | loss (56 t/s on forced 2048-token filler) |

- The first sweep's 65 t/s was the `ignore_eos` + greedy degeneration artifact. Real acceptance is 45-70%.
- n_max 2, 3 and 5 tie on real text; 8 and 12 lose. The shared head equals the self-contained one after
  the fix and is 0.3 GB/card cheaper.
- MTP combined with the fleet's ngram speculators was not measured.

### 2.2 Where a decode token goes (llama-bench tg128, tensor split, short context)

| CPU expert layers | t/s | ms/token |
|---|---:|---:|
| 0 (does not fit at c=131072) | 35.0 | 28.6 |
| 1 | 31.2 | 32.1 |
| 3 | 29.5 | 33.9 |
| 6 | 26.7 | 37.5 |
| 12 | 23.7 | 42.1 |

rocprofv3 kernel trace (per token, both GPUs): GPU busy ~21 ms per device of a 37.5 ms token, ~4400
launches per device per token, weight matvecs 21.4 ms (1064 launches, ~3x off the bandwidth roofline on
2560-wide matrices), NCCL allreduce 5.1 ms, elementwise kernels ~4 ms. Kernel time is identical at nc6 and
nc12: CPU layers are pure serial host time. HIP graphs are active and worth +5-7% at every residency.

Bandwidth (`/root/uva_bw.cpp`): one GPU reads host memory in place at 29.0 GB/s, both together 27.5 each,
the CPU alone 68.1 GB/s, all three 16.9 + 16.9 + 45.1. Expert traffic is 1.94 MiB per expert per layer
(down iq4_nl + gate/up iq3_s), 19.4 MiB per layer per token.

Split mode: layer split decodes +9% over tensor split at nc6 in llama-bench but OOMs at c=131072 with an
even split. Dropped per the user's decision.

### 2.3 The UVA buffer type (v14), production context

| config | prose t/s | code t/s | prefill 3.2k t/s | VRAM GB |
|---|---:|---:|---:|---|
| nc6 CPU layers (production today) | 26.4 | 26.3 | 843 | 33.64 / 32.92 |
| **uva6** | **32.3** | **32.3** | **1250** | 33.61 / 32.88 |
| uva6, non-coherent host memory | 32.6 | 32.6 | 1249 | 33.61 / 32.88 |
| nc10 CPU layers | 25.6 | 25.5 | 666 | 30.98 / 30.73 |
| uva10 | 30.2 | 30.1 | 855-1168 | 30.98 / 30.73 |
| nc10 + shared MTP n2 | 39.8 | 43.2 | 631 | 33.38 / 33.66 |
| **uva10 + shared MTP n2** | **41.9** | **45.4** | **1087** | 33.36 / 33.64 |
| uva9 + shared MTP n2 | 45.2 | 48.0 | 1110 | 34.08 / 34.12 (rejected, <100 MiB free) |
| uva8 + shared MTP n2 | - | - | - | OOM at load |

Candidate rungs (not written to the preset): uva6 without MTP as a drop-in for the production rung
(+22% decode, +48% prefill); uva10 + shared head n_max 2 as the MTP rung (+59% prose, +73% code, +29%
prefill vs production).

### 2.4 Numerical check (llama-perplexity KL divergence, 16 x 2048 tokens of /root/ppl.txt)

| Q vs base | mean KLD | PPL ratio | same top-1 |
|---|---:|---:|---:|
| v14 uva6 vs v14 nc0 (same layers resident in VRAM, layer split) | 0.000000 (max 6e-5) | 1.0002 | 99.994% |
| v14 nc6 vs v14 nc6 (repeat, noise floor) | 0.000000 | 1.0003 | 100% |
| v14 uva6 vs v14 nc6 (GPU vs CPU compute of 6 layers) | 0.0058 | 1.0010 | 98.8% |
| v14 nc6 vs v12 nc6 (binary change, same placement) | 0.0050 | 1.0006 | 99.0% |
| v14 uva6 vs v12 nc6 (production numerics) | 0.0055 | 1.0014 | 99.0% |

The mapped path is bit-for-bit the resident GPU path. The ~0.005 level is the CPU-vs-GPU kernel
difference (production already runs 42 of 48 expert layers on the GPU) plus the upstream kernel changes
between v12 and v14; it is not an injected error. Runs are deterministic.

## 3. What was NOT done

- No preset change, no cutover, no commit, no push.
- Phase 4 non-regression of v14 on the rest of the fleet: not run yet (test-backend-ops, test-llama-archs
  row count, fork-regress.sh, Qwen3.5-122B and Mistral within noise on their production rungs, slot-state
  portability, mmproj path). Upstream did not bump LLAMA_SESSION_VERSION / LLAMA_STATE_SEQ_VERSION this
  cycle, so no slot-cache re-prime would be needed.
- No convenience flag for UVA layers (a `--n-uva-moe N` analogue of `--n-cpu-moe`); the preset would use
  the explicit `-ot` regex. `llama-params-fit` is not implemented for tensor split, unchanged.
- R9V was not deployed for a direct comparison: it needs Docker and a tens-of-GiB container source build
  (PyTorch 2.11, Triton, AITER, a vLLM fork on ROCm 7.14); the box has no Docker and 16 GB free on root.
  Its design was read from source instead. Its 329/385 expert manifest is a residency ranking only (all
  512 experts computed, cold ones over UVA), so its numbers are a fair comparison.
- MTP + ngram speculators together, and the per-expert (instead of per-layer) hot set, were not measured.

## 4. Relation to the existing MoE hybrid kernel

The hybrid kernel serves models whose experts do not fit at all (Qwen3.5-122B: 80.9 GiB of experts,
Mistral-Small-4-119B: 89.6 GiB). It keeps every expert in host RAM and copies the hot ones into per-expert
VRAM slots by dynamic scoring (84-92% of expert visits served from VRAM); every layer is CPU-owned and the
GPU work is dispatched per layer with host coordination. Reading all of those experts in place instead
would move 2.5-2.9 GiB per token over PCIe (about 50 ms, a ceiling near 20 t/s), so the UVA type cannot
replace it for those models. The UVA type serves models that mostly fit and spill a few whole layers.

What the new measurements say about the old kernel:
- CPU-owned layers cost more than the design assumed (3.5 ms for the first, ~1 ms per layer, and a
  per-ubatch weight copy at prefill). The old kernel pays this on every layer.
- Its class 2 ("GPU reads host memory") lost because class 3 (CPU compute) ran at the same time and
  halved the GPUs' host-read bandwidth (27.5 -> 17 GB/s per card). With no CPU expert compute, in-place
  reads beat CPU compute by 22% decode and 48% prefill.
- Mapped and resident computation are bit-identical, so hot-in-VRAM and cold-over-UVA can be mixed freely.

## 5. Next steps, in the order they pay

1. Validate v14 on the fleet (section 3) and decide the Qwen3.8-Flash-Next rungs (uva6, uva10 + MTP).
2. A `--n-uva-moe N` flag mirroring `--n-cpu-moe`, so presets do not carry the regex.
3. Per-expert hot set for this model: a static VRAM set ranked by route counts, cold experts over UVA.
   Needs a `MUL_MAT_ID` variant whose per-expert source is a device-side pointer table (hot slots in VRAM,
   cold addresses in mapped host memory, fixed addresses so HIP graph replay survives). This is R9V's design
   and the natural successor of the hybrid kernel's dispatch path for the 122B / 119B models too: their pool,
   scoring and migration policy would fill the table, and class 3 plus the per-layer coordination would go.
4. The rest of the R9V gap on this model is not residency: ~4400 launches per device per token (2900 of them
   1-2 us elementwise kernels around the hyper-connections and delta-net), tiny-matvec efficiency on
   2560-wide matrices, and the two allreduces per layer. Fusion work, separate from placement.
5. Multi-token verify expert dedup for MTP (R9V's `reuse3`): only if MTP is adopted; their own doc calls
   the 27% byte saving a projection, not a measurement.

## 6. Fleet smoke check on v14 (2026-09-05, `/root/fleetcheck.sh`, results in `/root/fleetcheck/`)

One v14 `llama-server --models-preset <store>/main.ini --models-max 1` per store with the unit's CPU
flags, one medium profile per family, a ~2.6k-token C++ prompt with a summarise-and-poem instruction,
512 tokens greedy. Every answer was coherent and on topic (heuristic and eyeballed head of the text),
every model loaded within its rung, the embedding model returned finite 4096-dim vectors and the
reranker ranked the right document first. 27 / 27 OK, no failure, no gibberish.

DUALGPU (both GPUs, `llamacpp-both` flags):

| profile | pp t/s | tg t/s | VRAM GB |
|---|---:|---:|---|
| Qwen3.5-122B-A10B:M (hybrid kernel user) | 117 | 17.3 | 33.30 / 33.34 |
| Mistral-Small-4-119B-A6B:M (hybrid kernel user) | 153 | 15.8 | 33.00 / 33.07 |
| Mistral-Medium-3.5-128B:M | 381 | 14.3 | 33.23 / 33.23 |
| Qwen3.6-40B:CODE:M | 738 | 27.2 | 31.86 / 31.86 |
| Qwen3.6-27B:HQ:M | 1235 | 59.3 | 31.97 / 31.97 |
| gemma-4-31B:HQ:M | 1267 | 41.4 | 29.54 / 29.54 |
| Qwen-AgentWorld-35B-A3B:HQ:M | 1765 | 145.8 | 33.28 / 33.28 |
| Qwen3.6-35B-A3B:HQ:M | 2093 | 137.2 | 33.45 / 33.45 |
| Qwen3.6-27B-Fable:HQ:M | 1654 | 55.6 | 31.84 / 31.84 |
| Qwen3.8-27B:HQ:M | 928 | 49.2 | 32.71 / 32.71 |
| Qwen3.8-Flash-Next:M | 351 | 25.3 | 33.72 / 33.47 |

SINGLEGPU (GPU 0, `llamacpp-0` flags):

| profile | pp t/s | tg t/s | VRAM GB |
|---|---:|---:|---|
| Qwen3.6-27B:M | 893 | 36.7 | 32.44 |
| Qwen3.5-9B:M | 2306 | 48.9 | 25.39 |
| Qwen3-Embedding-8B | embeddings, dim 4096, finite | | 12.31 |
| jina-reranker-v2-base-multilingual | rerank, correct top document | | 0.48 |
| GLM-4.7-Flash-30B-A3B:M | 1789 | 74.1 | 32.47 |
| Qwen-AgentWorld-35B-A3B:M | 2191 | 103.7 | 32.53 |
| Qwen3.6-35B-A3B:M | 1544 | 75.9 | 31.08 |
| gemma-4-31B:M | 902 | 52.1 | 32.80 |
| gemma-4-26B-A4B:M | 2695 | 73.9 | 33.69 |
| Qwen3-Coder-Next | 786 | 36.9 | 32.71 |
| Devstral-Small-2-24B-Insctruct:M | 1393 | 35.2 | 33.46 |
| Magistral-Small:M | 1373 | 32.0 | 26.95 |
| Qwen3.6-27B-Fable:M | 940 | 41.2 | 30.30 |
| Muse-Glimmer-30B:M | 966 | 32.2 | 26.06 |
| Qwen3.8-27B:M | 1091 | 45.6 | 32.41 |
| gemma-4-12B:M | 1761 | 64.6 | 33.07 |

Not covered by this check: the other rungs of each family, vision (mmproj) profiles, slot save and
restore, the speculative decoders' acceptance on each model, and throughput within noise of v12 on the
same rung (only a single sample per model here).

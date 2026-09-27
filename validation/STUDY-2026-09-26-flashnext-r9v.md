# Qwen3.8-Flash-Next: our llama.cpp fork vs R9V (vLLM fork) — v19 study (2026-09-26)

R9V is github.com/Dyluhn/R9V v0.4.4, a vLLM fork with gfx1201 kernels. This document is the complete record of the campaign so far: setup, methodology traps, every measurement, the per-step anatomy of both engines, the interpretation, and the optimisation backlog. The chronological details are in section 10.

## 0. Where we stand

**Workload:** one warm request on the same UD-IQ4_XS GGUF, 65k ctx, a 32,222-token prompt, 2,048 generated tokens, greedy.

| config | prefill t/s | decode t/s | notes |
|---|---:|---:|---|
| v16 (production) | 958 | 36.2 | `:XL` rung, 16 UVA layers, MTP n2 + ngram drafts |
| v18 | 979 | 37.2 | v17 + rdna-boosts harvest |
| v19 dev, UVA rung | 975 | 37.4 | + AllReduce hybrid (since made opt-in, ~+1%) |
| **v19 dev, tiered experts** | **1,056** | **48.0** (pooled ABBA **+22%**) | per-expert hot/cold placement, same VRAM |
| v19 dev, tiered + ub 2048 | **1,308** | 44.4 (single run, noisy) | +1.5 GB VRAM per GPU, a rung decision |
| v19 dev, UVA + ub 4096 | 1,387 | 37.9 | 30.8/31.0 GB at 65k ctx, a rung decision |
| **R9V, warm** | **1,727** (1,729/1,724) | **68.2** (68.5/68.0) | qwen38-mtp4 profile |

**Same configuration** (ours set up like R9V: MTP depth 4, MTP-only drafts, p_min 0, greedy):

| | tokens/step | acceptance | ms/step | decode t/s | prefill t/s |
|---|---:|---:|---:|---:|---:|
| ours (tiered), 2 loads | **2.95** | 48.8% | **70-71** | 41.6 / 42.2 | 1,055 / 1,052 |
| R9V, 2 loads | 2.82 / 2.86 | 45.5% / 46.4% | **41-42** | 68.5 / 68.0 | 1,729 / 1,724 |

**The decode gap is entirely step cost.** Drafting matches or beats R9V; each of our steps takes 70% longer. The four structural culprits (section 6) add up to the ~29 ms difference.

## 1. Setup

- **Rig:**
  - Ryzen 9 9950X (1 socket, 1 NUMA node, 2 CCDs), a Proxmox LXC container with 110 GB RAM (129 GB host);
  - 2x R9700 32 GB on PCIe Gen5 x16 each (03:00 = ROCm0 / KFD node 1, 07:00 = ROCm1 / node 2); iGPU gfx1036 = node 3, excluded with `HIP_VISIBLE_DEVICES=0,1`;
  - kernel `7.0.14-6-pve`, `iommu=pt`, `pcie_aspm.policy=performance`, governor `performance`, THP `madvise`;
  - container limits: `/proc/sys` read-only (no `drop_caches`), locked memory capped at 8 MiB for ssh sessions AND systemd units (LimitMEMLOCK=infinity is not honoured), GPU sensors not visible (rocm-smi shows N/A).
- **Model:** Unsloth UD-IQ4_XS, 3 shards, 94 GB. They hash-match R9V's package manifest.
  - 48 layers: 36 GDN (linear attention), 12 full attention (every 4th);
  - 512 experts/layer, top-10, hidden 2560, expert FFN 640; hyper-connections hc=4 (low rank 320);
  - PLE (per-layer embeddings, 28.8 GB table, memory-mapped on the CPU) **in layer 1 only** (`ple.layers [1]`), n-gram 3, conv kernel 4;
  - sparse attention indexer: budget 2048, compress ratio 4.
- **MTP heads:** both come from the same official MTP layer.
  - Ours: Unsloth `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, Q8_0 from BF16, sharing the target's `token_embd`/`output`. Also on disk: the self-contained Q8_0 and a Q4_K_M.
  - R9V's: block-FP8, from the official `Qwen/Qwen3.8-Flash-Next-FP8` release.
- **Our rung (`:XL` args from the router journal, `fn-xl.args`):**
  - `-sm tensor`, ub 1024, b 4096, q8_0 KV, FA on;
  - `-ot per_layer_token_embd=CPU,blk.0-15 experts=ROCm0_UVA`;
  - `--spec-type draft-mtp,ngram-mod,ngram-map-k4v`, `--spec-draft-n-max 2` (p_min default 0);
  - temp 0.6 / top-p 0.95 / top-k 20 in production; the benchmarks send temp 0.
- **R9V (qwen38-mtp4):**
  - TP2 with an expert channel split of 416/224;
  - rank 0: 46 hot + 160-slot LRU cache per layer; rank 1: 434 hot;
  - WMMA grouped MoE prefill in 4096-token chunks; MTP depth 4 (block-FP8 head, local argmax); CED off.
  - It runs as a plain host process from the unpacked image, with no docker or chroot (`r9v-run.sh`): 48 GB host RAM, 27.0/29.0 GB VRAM, 132-155 s load warm.
- **ROCm:** ours 7.2.4; R9V ships its own 7.14 userspace (torch 2.11+rocm7.14).

## 2. Methodology: traps found and fixed

1. **Cold page cache halves prefill.** The rung keeps the 28.8 GB PLE table memory-mapped on the CPU, and prefill reads its rows for every prompt token. Anything that reads big files evicts it: a perplexity run over the GGUF, a 19 GB copy.
   - The same code gave 503 vs 978 t/s. R9V's first run (1,328 / 61.0) was partly cold too; warm it gives 1,727 / 68.2.
   - Fix: `openai_bench.py` warmup 2 (the full request once, then measured) and `cache_prompt: false`.
2. **The "ROCm 7.14 halves prefill" result is SUSPECT:** that run followed a 19 GB copy (cold cache). It must be re-measured warm.
3. **Harness bug: `multi-prompt.sh` picked the A/B arms by release NAME,** so a same-build A/B ran arm A twice (the same bug as the earlier `benchab.sh`). It invalidated the step 1 pooled AllReduce claim. Fixed: arms by position, plus `MP_ABBA=1` (A B B A).
4. **Run-to-run noise:**
   - Two identical pooled runs once differed by 9%.
   - Per server load, the identical first request (same tokens, acceptance 573/898 every time) ran 47.8 / 53.8 / 51.2 t/s, about ±6% **per load**, not per request.
   - Within one load, later requests take different trajectories: the ngram drafters learn across requests.
   - Single flashnext decode numbers vary ±10% (the text differs between runs: speculative decoding under -sm tensor is not deterministic across loads).
   - **Rule:** decode deltas below ~10% need pooled A B B A or repeated loads.
5. **Profiler overhead:** rocprofv3 adds ~12 ms/step to ours (70 → 81.6 ms); vLLM's torch profiler adds ~12 ms/step to R9V (41 → 53.5 ms). Traced numbers compare fairly with each other, not with untraced ones.
6. **rocprofv3 cannot trace R9V:** its worker processes (including the PLE helpers) do not exit on SIGINT, so the per-process traces are never written. Use vLLM's torch profiler (`r9v-torchprof.sh`, `/start_profile`, 1 warm-up + 24 steps).
7. Container: no `drop_caches`, locked memory capped at 8 MiB. Run long jobs as transient units (`systemd-run --unit=... --collect`).
8. Remote-call hygiene:
   - a rejected or interrupted plink call may still have run;
   - kill by PID only; never `pkill -f` or `ps | grep PAT | kill` (the pattern matches the plink shell);
   - a wait loop's timeout is not completion: check `systemctl is-active` before starting the next job.

## 3. v19 changes so far (branch `v19/flashnext`, local only)

1. **AllReduce P2P hybrid** (`5a9fdf8c9`, made opt-in in `41bc74fc7`, `GGML_CUDA_AR_HYBRID=<n>`):
   - FP32 reductions of up to n elements go through the fork's direct-P2P kernel next to RCCL; n ≤ 32768 keeps RCCL's numerics (bit-identical perplexity).
   - ABBA pooled decode: +1% (noise), so it is off by default.
   - The P2P kernel's GPU time equals RCCL's (~61-73 us/call), because both mostly wait for the peer.
   - Note: the dev tree `/opt/llamacpp/tmp-v19` is still built with the hybrid ON (32768). All step 2-5 "v19dev" numbers include it.
2. **Tiered MoE expert tensors** (`f8dc8a8b8`: `ggml/src/ggml-cuda/moe-tiered.{cu,cuh}`, `<dev>_TIERED` buffer type, `--moe-hot-experts FILE`):
   - hot experts per layer compacted in VRAM, all others compacted in mapped pinned host memory (each expert in one place);
   - a per-expert device address table after the hot experts, read by mmvq / mmvq-moe / mmvk / mmq (incl. the fused gate+up) and the sync fallback;
   - `set_tensor_2d` grouped per expert; aborts on any other op reading a tiered tensor; `GGML_CUDA_MOE_TIERED=0` makes every expert cold.
   - `test-moe-tiered`: 105/105 bit-identical on both GPUs (5 types, 1-512 tokens, half/none/all hot).
   - Perplexity bit-identical vs the UVA rung (ub 1024 and ub 8).
   - Hot file: `scripts/fleet/moe_hotset.py` with R9V's catalog route counts, `--match-uva 16` (the same VRAM as 16 UVA layers).
     - uniform 339/512 per layer: 92.7% of held-out routes served from VRAM; greedy: 93.2%; today's whole-layer placement: 66.7%.
   - Result:
     - flashnext 975/37.4 → 1,056/48.0;
     - **pooled ABBA 42.75 → 52.15 t/s (+22%) at identical acceptance**;
     - decode profile: MoE 8.7 s → 3.7 s, p90 dispatch 616 → 148 us;
     - VRAM and host RAM unchanged.

## 4. Other measured findings

- **Prefill ubatch** (config only): ub 1024 → 2048 → 4096 gives 978 → 1,233 → 1,387 t/s (UVA), and 1,056 → 1,308 with tiered at ub 2048. Host experts are streamed once per ubatch. VRAM: +1.5 GB/GPU at ub 2048, 30.8/31.0 GB at ub 4096 (65k ctx).
- **MTP depth** (tiered, with ngram drafters, pooled ABBA): n3 -1.5%, n4 -15%.
  - At n4 our acceptance per draft token (44-46%) equals R9V's (44%): the head is not the problem; the extra verify cost is.
  - `n_rs_seq` follows n_max (`need_n_rs_seq`), so MTP drafts never trigger the host checkpoint. Each extra draft depth adds one rollback slot: +2 conv-state copies per GDN layer and one more GDN state snapshot.
- **Draft-quality levers (parked by the user until the core gap is fixed):** `--spec-draft-p-min` (default 0 = never stop early; the preset does not set it; the MTP drafter honours it at `common/speculative.cpp:2202`), and the MTP head (see section 1; same source weights).
- **Host profile** (gdb, 40 samples during decode): the compute thread sits in `ggml_backend_cuda_synchronize` 38/40; CPU sampling 1/40, graph capture 1/40. Host-side sampling and MTP graph rebuilds are not the bottleneck (steps 1.2/1.3 dropped).
- **HIP knobs** (`HSA_NO_SCRATCH_RECLAIM=1`, R9V's only runtime knob; `HIP_FORCE_DEV_KERNARG=1`): inconclusive, drowned by the ±6% per-load noise. Re-test with repeated loads.
- **ROCm 7.14:** the v18 build ran 482 prefill (cold, suspect) and 37.9 decode. It needs a warm re-run: `/opt/rocm-7.14` from R9V's rootfs `core-7.14`, `build-rocm714.sh`, run with `LD_LIBRARY_PATH=/opt/rocm-7.14/lib`.

## 5. Per decode step, both engines, same configuration

Tiered rung, MTP 4, 32k prompt. Ours: rocprofv3, 161 steps (512 tokens). R9V: torch profiler, 24 steps. Busier GPU; both profilers add ~12 ms/step.

| per step | ours (GPU 0) | R9V rank 0 | R9V rank 1 |
|---|---:|---:|---:|
| wall (traced) | 81.6 ms | 53.5 ms | 53.5 ms |
| GPU busy | 52.2 ms | 37.3 ms | 38.0 ms |
| idle between kernels | 29.5 ms | 16.2 ms | 15.5 ms |
| kernels | 4,744 | 2,761 | 2,614 |

**Ours per step** (verify of 5 tokens + 4 drafts):

| kernel | calls | avg | ms/step |
|---|---:|---:|---:|
| `mul_mat_vec_q_moe` | 95 | 168 us | **16.0** |
| `mul_mat_vec_q` | 572 | 15.7 us | **9.0** |
| `ggml_cuda_ar_hip_kernel` (P2P all-reduce) | 105 | 73 us | **7.6** |
| `flash_attn_ext_f16` / `_vec` | 12.8 / 4 | 292 / 210 us | **3.7 + 0.8** |
| `mul_mat_vec_f` | 272 | 7.4 us | 2.0 |
| `mul_mat_vec_k` (Q6_K output head, verify + drafts) | 4 | 417 us | 1.65 |
| `mul_mat_f` | 24 | 46 us | 1.1 |
| `k_bin_bcast` | 406 | 2.5 us | 1.0 |
| `rope_multi` | 58 | 17 us | 1.0 |
| `quantize_q8_1` | 668 | 1.3 us | 0.9 |
| `rms_norm_f32` | 289 | 2.8 us | 0.8 |
| `k_get_rows_float` / `k_get_rows` | 62 / 12 | 11.6 / 59.5 us | 0.7 + 0.7 |
| `copyBufferRectAligned` / `copyBuffer` | 250 / 238 | 2.7 / 1.4 us | 0.7 + 0.3 |
| `top_k_radix_select` (indexer) | 48 | 9.7 us | 0.5 |
| `scale_f32` / `unary_op` / `unary_gated_op` | 408 / 354 / 106 | ~1 us | 0.45 / 0.37 / 0.13 |
| `topk_moe`, `gated_delta_net`, `cpy_scalar_transpose`, `dsv4_hc_post/pre`, `concat_non_cont`, `get_rows_float_vec` | | | ~0.2-0.36 each |

**R9V rank 0 per step:**

| kernel | calls | avg | ms/step |
|---|---:|---:|---:|
| `tiered_moe_vec_exact_reuse3v2` | 96 | 197 us | **19.0** |
| `dense_mmvq_q8_reuse5` | 192 | 20.4 us | 3.9 |
| `copy_planned_expert_cache` + `plan_expert_cache_lru_fill` + `publish_expert_cache_lru` (LRU cache) | 48 each | | 2.4 + 0.9 + 0.3 |
| `vllm::cross_device_reduce_1stage` (custom AR) | 97 | **17 us** | **1.7** |
| `wvSplitK_hf_big` / `_sml` (BF16 skinny GEMV) | 96 / 96 | 14.8 / 5.3 us | 1.4 + 0.5 |
| hipBLASLt GEMM (`Cijk_...`) | 48 | 22 us | 1.1 |
| `hc_q8_mmq45_exact` (fused hyper-connection mat-vec) | 97 | 10 us | 1.0 |
| `gdn_mtp_tp2_fp32_kernel` (fused GDN for verify) | 36 | 17.6 us | 0.6 |
| `dense_mmvq_q6_reuse5` (output head, once for 5 tokens) | 1 | 633 us | 0.6 |
| torch elementwise (`vectorized` / `manual_unroll`) | 394 / 217 | 1.2 / 1.8 us | 0.9 |
| `quantize_q8_1` | 292 | 1.3 us | 0.4 |
| `topkGating` | 48 | 7.6 us | 0.4 |
| **sparse attention**: `_qsa_mqa_paged_strided` + `topKPerRowDecode` + `_qsa_sparse_paged_gqa_splitk` (+ merge) | 12 each | 21 / 16 / 14 us | **~0.6** |
| `_hc_combine_norm`, `quantize_hc45`, `_causal_conv1d_update`, copies | | | ~0.1-0.2 each |

**R9V rank 1** spends **20 ms/step in the all-reduce** (97 x 206 us, waiting). Its 224-channel MoE share takes 4.7 ms, so it idles while rank 0 finishes. R9V's uneven split is not balanced; rank 0 is its critical path. Our even 320/320 split is balanced.

**Our verify pass alone** (n2 profile, 3 tokens):
- 38.5 ms wall, 25.3 ms busy;
- 4,096 dispatches with ~4,000 gaps of 2-5 us, **11.7 ms in total**: uniform per-dispatch overhead, not subgraph boundaries (only ~12 gaps > 20 us, 0.7 ms);
- per pass: `quantize_q8_1` 593, `mul_mat_vec_q` 497, `scale` 375, `bin_bcast` 361, `unary` 330, `mul_mat_vec_f` 288, `rms_norm` 256, copies 324, `unary_gated` 97, `hc_pre`/`hc_post` 96 each, AR 96, MoE 96.
- The draft passes are ~6-12 small evaluations per step (7-42 dispatches each, 0.4 ms wall): cheap.

**Per-layer pattern** (GDN layer, from the kernel sequence):
- HC pre: `rms_norm`, `quantize`, `mmvq(w_down)`, `scale`, `silu`, `quantize`, `mmvq(w_up)`, `hc_pre`.
- GDN block:
  - conv-state update: `concat_non_cont` + (`copyBufferRect` + `copyBuffer`) x3 (one pair per rollback slot);
  - `get_rows_vec`, `ssm_conv`, 2x (`rms_norm` + `scale`), `mmvf`, `bin_bcast`, `unary`, `bin_bcast`, `mmvf`, `unary`, `gated_delta_net`, `rms_norm`;
  - `quantize` + `mmvq`, `unary_gated`, `quantize` + `mmvq(out)`, AR.
- HC post: `mmvf`, `scale`, `sigmoid`, `scale`, `hc_post`.
- HC pre (ffn): the same 8 kernels.
- MoE: `mmvf(router)`, `topk_moe`, `quantize`, `mmvq_moe(gate+up)`, `quantize`, `mmvq_moe(down)`, `bin_bcast` x3, AR.
- Shared expert: ~13 kernels.
- HC post (ffn): ~9 kernels.
- Total ≈ 85 per layer. Several mat-vecs share an input (wq/wk/wv; wqkv/wqkv_gate; HC down/inject; the PLE key/value pair; shared up/gate), and each quantizes it again.

## 6. Interpretation: the culprits (ours vs R9V rank 0, per step)

1. **All-reduce + launch skew: ~6 ms, plus part of the idle time.**
   - The meta backend (`-sm tensor`) cuts every decode into ~100 host-launched subgraphs, with a host-issued all-reduce between them.
   - Each GPU's subgraph is launched in turn from one host thread, so GPU 0 waits for GPU 1 at every cut: 73 us per AR vs R9V's 17 us.
   - R9V runs one CUDA graph per rank per step with its custom AR inside.
2. **Sparse attention done densely: ~4 ms at 32k ctx, growing linearly with context.**
   - `build_attn_qsa` (qwen4exp.cpp) builds a full [n_kv x n_tokens] -inf mask (`ggml_fill` + `ggml_set_rows` of the top-k cells + `ggml_add` of the causal mask), then runs dense FA over all n_kv cells.
   - R9V gathers and attends only the indexer-selected blocks (2048 budget).
3. **Dense mat-vecs: ~4.7 ms.**
   - R9V: `dense_mmvq_*_reuse5` (weights read once for the 5 verify tokens), a fused HC mat-vec, BF16 skinny GEMVs, and the output head once per step.
   - Ours: 572 mmvq + 272 mmvf + 4 x 417 us output head per step.
4. **~2,000 more kernels per step, and their 2-5 us gaps:** ~6 ms or more.
   - Per-mat-vec input quantisation (668/step);
   - unfused elementwise HC / norm / scale chains;
   - GDN conv-state copies (2 x (n_rs_seq+1) + concat per GDN layer);
   - full-length mask building in the attention layers.
5. **MoE is NOT a culprit:** ours is 16.0 ms vs R9V rank 0's 22.5 ms (incl. cache management). There is still headroom from multi-token expert reuse (reuse3v2: decode each distinct expert once for all verify tokens). Cold experts sit in coherent (GPU-uncached) host memory, so duplicate reads of a cold expert cross PCIe again.
6. **Prefill** (1,053 at ub 1024 vs 1,727): mostly the chunk size (host experts re-streamed per ubatch) plus R9V's WMMA grouped MoE kernel. Not profiled per kernel yet.

## 7. Optimisation backlog (the user: "we will optimise everything you found")

Every item is measured with the same-configuration protocol:
- `step5.sh`-style warm flashnext on two loads, plus pooled ABBA for decode;
- `decode-profile.sh` + `torchtrace_anatomy.py rocprof` for the per-step anatomy;
- perplexity / `test-backend-ops` for correctness.

1. **Sparse QSA attention for small batches** (decode/verify, n_tokens ≤ 8).
   - Gather the top-k selected KV cells per query token, then run FA on the compact set (2048 instead of n_kv). No full-length mask.
   - Keep the dense masked path for prefill. Expected: ~4 ms/step at 32k, more at 64k+.
2. **One graph per GPU per decode (in-graph all-reduce).**
   - Make the P2P all-reduce a node inside each device's graph (the kernel is graph-safe, `allreduce.cu:987-993`) instead of host-issued between ~100 subgraphs.
   - Removes the per-cut host round trips and the GPU0/GPU1 launch skew. Expected: most of the 7.6 ms AR, plus idle time.
   - The biggest structural change (`ggml-backend-meta.cpp`).
3. **Kernel-count reduction** (each pattern is measured by kernels/step):
   - a. quantise once per shared mat-vec input (~230/pass);
   - b. GDN conv-state update in one kernel (concat + per-slot tail copies: ~7+ per GDN layer, x36);
   - c. fuse the HC elementwise chains: `scale→silu` (x2/layer), `scale→sigmoid→scale` (x2/layer);
   - d. norm + scale pairs;
   - e. MoE glue (`bin_bcast` x3 after the MoE);
   - f. the full-length QSA mask build, which disappears with item 1;
   - g. PLE constant weight `cont`/cast per pass (layer 1 only: small).
4. **Dense multi-token mat-vec efficiency:** weights read once for all verify columns, cheaper at 5 tokens (R9V `dense_mmvq_*_reuse5`); a fused HC mat-vec (R9V `hc_q8_mmq45`); the output head once per step.
5. **MoE multi-token reuse:** decode each distinct expert once and apply it to all matching verify tokens (R9V `reuse3v2`). Consider non-coherent (cacheable) mapped memory for cold experts (`GGML_CUDA_UVA_NONCOHERENT`).
6. **Then MTP depth 3/4 again,** plus the user's levers (p_min, MTP head A/B).
7. **Prefill:**
   - the ub size (a rung decision);
   - a per-kernel prefill anatomy of both engines (not done yet);
   - grouped WMMA MoE / cold-expert streaming if the anatomy says so.
8. Re-measure ROCm 7.14 warm; re-test the HIP knobs with repeated loads.
9. (Needs user approval: VRAM copies of host experts) the LRU expert cache on top of the tier table.
10. The hot set from our own route counts (`llama-imatrix` counts) vs R9V's catalog; greedy vs uniform (93.2 vs 92.7% predicted).

## 8. Tools (in git)

- **`scripts/fleet/`:**
  - `r9v_fetch.py` (package + image, SHA-256 checked);
  - `oci_unpack.py` (docker-save → rootfs, no docker);
  - `openai_bench.py` (one streaming request; `BENCH_SAVE_TEXT`; warmup 2 = warm);
  - `moe_hotset.py` (hot file from imatrix counts or R9V's catalog; `--match-uva N`, `--policy uniform|greedy`, `--eval heldout:`);
  - `multi-prompt.sh` (pooled 6 prompts; arms by position; `MP_ABBA=1`, `MP_ENV_A/B`, `MP_ARGS_B`, `MP_NO_RESTART=1`).
- **`validation/v19-scripts/`:**
  - `r9v-run.sh` (R9V from the rootfs; `R9V_TORCH_PROFILE_DIR`);
  - `mkprompt.sh`;
  - `flashnext.sh` (`FN_ENV`, `FN_TAG`, `FN_UB`, `FN_ARGS`, `FN_WARM`; leaves production stopped);
  - `step1.sh` .. `step5b.sh`, `recheck.sh`;
  - `decode-profile.sh` (`DP_ARGS`, `DP_PROMPT`, `DP_TOKENS`, `DP_TAG`);
  - `hostprof.sh` (gdb samples);
  - `envab.sh` (env A/B, `ENVAB_REPEAT`, `ENVAB_NOREV`);
  - `step_anatomy.py` (evaluations, gaps, kernels per verify pass);
  - `pass_sequence.py` (kernel sequence of one pass);
  - `torchtrace_anatomy.py` (per-step numbers from vLLM torch traces or our rocprof traces);
  - `r9v-profile.sh` (rocprof, loses the traces; superseded by `r9v-torchprof.sh`);
  - `build-rocm714.sh`.

## 9. Artefacts on the box

- `/mnt/gguf/r9v/`:
  - `package/` (our shards symlinked, R9V's MTP/mmproj/metadata);
  - `image/` (11 parts + joined tar);
  - `rootfs/` (23 GB unpacked runtime);
  - `ple/` (28.8 GB PLE table, hash `dd55c289...`);
  - `src/` (R9V source at aeee44f);
  - `cache/`;
  - `bench/` (all runs: results.jsonl, logs, `.mem` samples, answers, args files, hot files, traces `prof-decode-*`, `prof-torch-r9v`).
- Dev tree `/opt/llamacpp/tmp-v19` (+ symlink `llama-cpp-mine-v19dev`), built from the step 2 code (tiered + AR hybrid default ON). Remove both at the end.
- Production is stopped for the whole campaign. The unit files on disk point at v18 (edited by the user), but systemd still holds the v16 ones.

## 10. Chronological log

- Baseline (13:5x): v16 958/36.2, v18 979/37.2, R9V (partly cold) 1,328/61.0 at depth 4 (acceptance 44.1%).
- ROCm 7.14 build of v18: 482/37.9 (cold, suspect).
- Step 1: AR hybrid (bit-identical, +1% ABBA); ub 2048/4096 prefill; host profile.
- Step 2: tiered +22% pooled; perplexity identical; 105/105 tests.
- Step 3: pooled ABBA (hybrid, tiered, n3, n4); greedy hot set and tiered+ub2048 ran during a drift (inconclusive).
- Recheck: tiered 1,061/42.4, tiered+ub2048 1,308/44.4, UVA+ub2048 1,229/38.4.
- HIP knobs and variance: ±6% per load.
- Step 5: the same-configuration comparison and both traces (sections 0 and 5).
- Step 6: sparse QSA attention, -4.35 ms/step (section 11.1); the AllReduce wait traced to the 384/256 MoE split (11.3).

## 11. Optimisation phase (from 2026-09-26 evening)

Metric: **ms per speculative step** (one verify pass plus its drafts), from `step_ms.py`. Tokens/s moves with the
acceptance, which changes with the generated text from one load to the next (0.47-0.51 in these runs).
Same-config args (`fn-xl-tiered-r9vlike.args`: MTP 4, MTP drafts only, 32k prompt, 2,048 tokens), warm, A B B A over loads.

### 11.1 Sparse QSA attention (step 6)

- **What:** decode and verify batches (≤ 8 tokens per stream, `LLAMA_QSA_SPARSE`, 0 = dense) hand the indexer's top-k
  list of each query to flash attention (`ggml_flash_attn_ext_add_kv_idx`, src[8]). The kernel gathers the listed
  K/V rows and reads the ordinary KQ mask at each one; no n_kv-wide mask is built (fill + set_rows + add gone).
  - CUDA/HIP: upstream's sparse mode of the mma kernel (NVIDIA-only until now), fed from src[8] instead of a mask
    compaction, at `<256,256,ncols1=1,ncols2=16>` (one query per tile, the 12 GQA heads of a device in the 16
    columns; new template instance). q8_0 K/V read in place.
  - CPU reference, `test-backend-ops` (16 cases: f16/q8_0, GQA 4-24, 1-8 queries, 2 sequences, -1 entries),
    every other backend refuses src[8], the meta backend asserts the lists are mirrored.
- **Correctness:** perplexity at ub 8 and 8k context (so the indexer really drops cells): dense 1.3260 ± 0.0249,
  sparse 1.3269 ± 0.0251.
- **Speed:** dense 68.0 / 67.9 ms/step, sparse 64.0 / 63.2 → **-4.35 ms/step (-6.4%)** at 32k. The verify FA went
  from 3.7 to 0.97 ms/step (75 us per call).
- The AllReduce now runs on RCCL (`ncclDevKernel_Generic_4`, 105/step, 64 us): the hybrid P2P path became opt-in in
  `41bc74fc7`, built for the first time here.

### 11.2 Where the QSA layer time really went (step 5 trace, one layer of a verify pass)

| step | kernels | us per layer |
|---|---|---:|
| index K / Q projections (BF16, 5 columns) | 2 x `mul_mat_f` (4-16 blocks only) | 97 |
| dequantize every raw indexer key | `k_get_rows` q8_0 (16.8 MB f32) | 64 |
| mean of the 4 block members | 4 copies + 3 adds + scale | 77 |
| norm + rope over all 8,192 blocks | `rms_norm`, `rope_multi` (81 us: 3/4 of each block idle) | 106 |
| score GEMM, relu, head sum, bias | | ~35 |
| block score to cells and back | transpose, `get_rows_float`, transpose, cast, add | 82 |
| top-k over 32k cells | radix, 4 passes | 56 |
| dense FA + fixup (before 11.1) | | 333 |

About 850 us per layer, ~10 ms per step over 12 layers (R9V: ~0.6 ms). Host side: `set_input_qsa` is O(n_kv) per
ubatch (~0.9 ms at 33k per its own comment), not yet measured.

### 11.3 The AllReduce wait is an uneven MoE split, not launch skew

- Pairing the AllReduce kernels of both GPUs: the ends agree to ±5 us, the starts spread p10 -140 / p90 +157 us.
  The pure AllReduce costs 9 us; the rest is one GPU waiting for the other.
- By subgraph: MoE mean start skew 150 us, correlated **1.00** with the difference in GPU time; attention 32 us,
  GDN 10 us.
- Per layer, one GPU is always ~1.5x slower on the MoE, and the slow GPU alternates layer by layer.
- **Cause:** the FFN split unit is `lcm(block, 128)` = 128 (`get_split_granularity`, llama-model.cpp), so the expert
  and shared-expert width 640 = 5 units splits **384/256**, and the per-layer "rotation" swaps the sides.
- So an in-graph AllReduce (backlog item 2) would not remove this wait. The fix is the split.

### 11.4 Step 7: fused indexer, rope, BF16 mat-vec, even MoE split, MTP attention

- `ggml_qsa_pool` (gather + mean + RMS norm of the cached keys, quantized rows read once) and `ggml_qsa_expand`
  (block score to cells plus the mask) replace the generic indexer ops (`LLAMA_QSA_FUSED=0` restores them).
- `rope_multi` packs several short rows into one block.
- BF16 mat-vec up to 5 columns on RDNA4 (the indexer projections left `mul_mat_f`).
- FFN split unit halves while the width does not divide evenly: 640 → 320/320.
- MTP draft attention (1 query, 12 heads per K/V head, q8_0) on the mma kernel at (1, 16) instead of `vec`
  (210 us per call, 4 per step; `GGML_HIP_FA_GQA16=0` restores it).

- **Correctness:** test-backend-ops QSA_POOL 12/12, QSA_EXPAND 6/6, ROPE 454/454, FLASH_ATTN_EXT 4102/4102,
  MUL_MAT bf16 151/151. Perplexity (ub 8, 8k): generic indexer 1.3262, fused 1.3228 (± 0.025; rounding-level
  score changes move blocks across the top-k boundary).
- **Speed** (A = the step 6 build, B = step 7, A B B A): A 62.5 / 61.6, B 55.6 / 56.8 ms/step →
  **-5.9 ms/step (-9.4%)**; prefill 1,045 → 1,116 t/s (+6.8%: the even split and the fused indexer help ubatches too).
- Since the start of the optimisation phase: **68.0 → 56.2 ms/step (-17%)**. R9V: 41 ms/step.

### 11.5 The draft loop costs a quarter of the step (step 7 trace)

- Per step (traced): verify pass 49.1 ms; 9.3 small draft evaluations 9.9 ms; **GPU idle before each draft
  evaluation 0.7 ms, 6.6 ms per step**, plus 1.0 ms before the verify.
- The MTP draft sampler cannot run on the backend under -sm tensor (`llama_context::set_sampler` refuses it), so
  every draft token copies the logits and runs the generic CPU chain (logit bias + top_k(10) + dist), which builds
  a candidate array of the whole 248k vocabulary. Only the first candidate and its probability are used.

### 11.6 Step 8: kernel-count cuts (one build, env toggles)

- `quantize_q8_1` reuse (a 4-entry cache of q8_1 copies keyed by the activations' root tensor and row layout, so the
  routed and the shared experts share one copy): 668 → 549 quantizations per step.
- q8_0 mat-vec + mat-vec + GLU now fuse at 2-8 columns (the shared expert of a verify batch; the kernel supported
  it, the launcher did not): mul_mat_vec_q 572 → 524 per step.
- Conv-state rollback tails copied straight from the strided view (no cont first): 440 → 259 copies per step.
- The scale -> silu / sigmoid (-> scale) and rms norm -> scale fusions **did not run**: `ggml_cuda_can_fuse` is a
  whitelist and did not list them (fixed for step 9).
- So the A/B measured mostly the P2P AllReduce (B) against RCCL (A): A 54.9 / 55.6, B 54.6 / 56.3 ms/step, no gain;
  the trace shows the P2P kernel at 37.6 us per call against RCCL's 31 us. RCCL stays the default.
- Perplexity (ub 8, 8k): A 1.3280, B 1.3273. Kernels per step 4,594 → 4,194. Step: **~55 ms** (from 68.0).
- Found on the way: the fork's mmvk (RDNA4 K-quant mat-vec, batch 1) skipped the SWIGLU_CLAMP gate
  (MUL_MAT_VEC_FUSION q4_K glu_op=6 NMSE 9.5): fixed for step 9.

### 11.7 Step 9: fast MTP draft top-k, the fusions really on

- A = `LLAMA_SPEC_FAST_TOPK=0` and both elementwise fusions off, B = default (RCCL AllReduce in both).
- A 54.5 / 55.1, B 53.3 / 53.7 ms/step → **-1.3 ms/step**. test-backend-ops MUL_MAT_VEC_FUSION 1613/1613, ADD 228/228.
- Trace (B): 3,804 kernels per step (4,194 in step 8), traced wall 59.9 ms, busy 39.3, idle 20.5.
- Host profile (gdb samples at 32k): the main thread waits in `ggml_backend_cuda_synchronize` in 58 of 60
  samples. The GPU idle gaps (~4.4 ms per step) are wake-up and launch latency around each host sync plus the
  blocking input uploads, not host compute: the lever is fewer syncs per step (one per draft token today).
- Since the start of the optimisation phase: **68.0 → 53.5 ms/step (-21%)**. R9V: 41.

### 11.8 Where the remaining 12.5 ms are (step 8/9 traces vs R9V rank 0)

| per step | ours | R9V rank 0 |
|---|---:|---:|
| dense `mul_mat_vec_q` | 9.0 ms (524 calls) | 3.9 ms (192) |
| idle (launch gaps + host syncs) | 20.5 ms traced | 16.2 ms traced |
| AllReduce | 3.3 ms (RCCL) | 1.7 ms |
| output head (verify + 3 drafts, full vocabulary) | 1.66 ms | 0.6 ms (once; drafts use a coarse head) |
| attention | 1.5 ms | 0.6 ms |
| MoE (+ R9V's LRU cache) | 15.6 ms | 22.6 ms |

- Typical step (median, traced): verify 45.1 ms, 4 draft passes 8.1 ms (~2 ms each, 0.42 ms of it the head),
  host-sync gaps 4.4 ms.
- The hyper-connection down projection (10240 -> 320, 96 per step) runs at ~240 GB/s: RDNA4 verify batches get
  one warp per row, which starves a matrix with few rows and a long K.

### 11.9 Step 10: tall-K verify mat-vecs, MoE sharing

- `GGML_CUDA_MMVQ_TALL_K` (default on): on RDNA4 a verify batch (2-8 columns) through a q8_0 matrix with fewer than
  2,048 rows and at least 128 blocks of K gets 4 warps per row instead of one (the hyper-connection down projection
  10240 -> 320 ran at ~240 GB/s). test-backend-ops MUL_MAT q8_0 59/59.
- A (off) 53.9 / 54.6, B 50.9 / 52.3 ms/step → **-2.6 ms/step (-4.8%)**. Since the start: **68.0 → 51.6 ms/step**.
- `GGML_CUDA_MOE_STATS=1` (a debug counter; it synchronizes): per MoE call of a verify batch, **49.9 (token, slot)
  pairs but 33.7 distinct experts (32% duplicates); cold 11.2 pairs, 7.3 distinct (35% duplicates)**. Our kernel reads
  an expert once per pair, and cold experts sit in uncached host memory, so every duplicate crosses PCIe again: the
  case for a kernel that reads each distinct expert once per batch (R9V's reuse3v2).

### 11.10 Status at the mid-way write-up (2026-09-26 night)

- 68.0 → **51.6 ms/step** (-24%), prefill 1,045 → ~1,120 t/s, perplexity unchanged. R9V: 41 ms/step.
- The reusable lessons are in `PLAYBOOK-rdna4-tp-decode-optimisation.md` (methodology, traps, design rules, the gap
  table) and `PLAYBOOK-hip-kernel-optimisation.md` (HIP / RDNA4 kernel work).
- In progress: MoE expert reuse (`mul_mat_vec_q_moe_reuse`, R9V reuse3v2 structure, `GGML_CUDA_MOE_REUSE`).
- Next: the draft loop (needs a split-aware top-k in the meta backend for an unrolled draft graph or a coarse head),
  fewer and wider dense mat-vecs, the sparse FA config (75 us vs R9V's 14 us), the MoE weighted-reduction fusion that
  does not match on this graph, the GDN gate chains.

### 11.11 Steps 11-27 (2026-09-26 night to 2026-09-27)

Same-config ms per speculative step (step_ms.py, 2 loads per arm, today's hot set of 36.7 GiB unless noted):

| step | change | result |
|---|---|---|
| 11-12 | MoE reuse kernel (one block per distinct expert) | lost (MoE 15.5 -> 22.4 ms; VRAM spill, serialized loads), dropped |
| 12 | meta backend allocation dependencies | the fused MoE weighted sum runs in decode; costs nothing |
| 13 | ROCm 7.14 runtime / full build; HIP runtime knobs | decode +1-2 ms, prefill +3% (7.14); knobs: no effect |
| 14 | deterministic one-kernel top-k | 52.6 -> 49.4; **decode deterministic** (same text every load) |
| 15 | staged uploads (no blocking copy per input) | 49.3 -> 47.6, text byte-identical |
| 15 | F32 mat-vec tune (rows per block, wide blocks) | kernels -0.8 ms/step, step time unchanged (host-bound) |
| 16-18 | graph packet capture knob, KV pad 2048 | no gain (re-captures are cheap bursts every 256 tokens) |
| 19-24 | q8_1 written by producers (norm, gates, HC mix, output gate), copy runs | 47.6 -> 46.4, 3,477 -> 3,115 kernels/step |
| 21 | hot set 44 GiB (user's decision) | **40.8 ms/step, decode 71 t/s, prefill 1,283**, 31.6 of 32.6 GB per card |
| 24 | FA RDNA config sweep (256/256/16 cols) | baseline best; a dedicated sparse decode kernel parked |
| 25 | ubatch 2048 / 4096 (user's decision) | prefill 1,386 (+20%) at 30.75 GB; 4096 OOM |
| 26-27 | prefill staging of cold experts, one area | expert matmuls 10.5 -> 4.7 s per GPU, but copies (~26 GB/s) not hidden: prefill 1,152 -> 1,030 |
| 28 | staging with 2 and 3 areas | 1,030 / 1,027 vs 1,152 (off): more areas change nothing; staging stays off (see below) |
| 29 | `GGML_ALLOC_PEAK` probe | ub 4096 asked 10.1 GiB/GPU: 7 GiB = 14 device copies of the KQ mask, 2 GiB indexer score + relu |
| 30 | `ggml_qsa_expand_heads`, mask as is, shared zeros | compute buffer ub 1024 2,533 -> 909 MiB, ub 4096 10,130 -> 3,633 MiB; ub 1024 46.4 -> 46.0 ms/step, prefill 1,178, text identical; **ub 4096 fits: prefill 1,542 at 30.1 GB** |
| 31 | mm_ids_helper 8 loads ahead; F32 GEMM library; sparse prefill attention | 1,178 -> 1,193, text identical; rocBLAS Tensile (`ROCBLAS_USE_HIPBLASLT=0`) +1% (text changes); `LLAMA_QSA_SPARSE=1024` +4.5% (text changes) |
| 32-33 | own F32 GEMM (sgemm.cu) | router 1,034 -> 287 us, indexer 6,601 -> 1,795 us, HC 245 -> 123 us; **prefill 1,190 -> 1,280**; new reference text md5 0c8460b131f6183795627bea73282ee2 |

Every step since 14 checks the generated text md5 (reference 8a0ec2a9ccd2816c6250db15e16181eb up to step 31; from step 32, with our F32 GEMM, 0c8460b131f6183795627bea73282ee2: both texts are coherent and diverge 868 characters in).

Prefill anatomy (32k prompt, ub 1024, per GPU of ~24.7 s busy): mul_mat_q 12.3 s (MoE experts 10.2 s, bound by the
cold experts' PCIe reads), hipBLAS F32 GEMMs 2.9 s (router 1.0 ms/call at 2.7 TFLOPS on an 8x8-tile kernel, hc_inject
216 us, gated delta net alpha/beta, indexer scores), dense FA 1.9 s, gated delta net 1.2 s, RCCL 0.95 s, mm_ids_helper
0.94 s (3 calls per layer with the same ids; each of its 512 blocks scans every route).

Why the prefill staging loses at any area count (step 27 trace, `stage_trace.py`, per GPU): in the middle of the
prefill every layer's copies (7.3 ms) start right after the previous down projection and finish before they are
needed, with no idle gap above 0.3 ms. The loss is elsewhere. Per ubatch the GPU idles 160 ms (12 ms with staging off)
in sub-millisecond gaps spread over every layer, and the small kernels run slower (quantize_mmq_q8_1 0.48 -> 1.25 s,
rms_norm 0.64 -> 0.98 s, RCCL 0.95 -> 1.78 s). The DMA fills the PCIe link for about half of each layer, and everything
else that crosses it waits: the dispatch packets and kernel arguments (host memory) and the GPU-to-GPU all-reduce.
It also moves more bytes than the in-place reads: 12.3 s of copies per GPU for all cold experts, against a 5.8 s
in-place penalty for the routed ones only. Upper bound even if the side effects vanished: the 5.8 s. Staging stays in
the tree with `GGML_CUDA_MOE_STAGE` default off. Levers left for prefill: fewer passes over the cold experts (larger
ubatch, R9V prefills in 4096-token chunks) and fewer cold bytes (hot set), both limited by VRAM, hence step 29.

The compute buffer (steps 29-30). `GGML_ALLOC_PEAK=N` (ggml-alloc.c) logs the N largest tensors alive when each
buffer reached its size. At ub 4096 (65k ctx) the 10.1 GiB per GPU were:
- 7 GiB: 14 copies of the [65536 x 4096] f16 KQ mask. Each QSA layer passed `ggml_reshape_3d(kq_mask)` to
  qsa_expand; each reshape is a new tensor, the scheduler uploads a device copy per tensor, and its copies are graph
  outputs (never freed). The mask was also uploaded 14 times per ubatch.
- 2 GiB: the indexer score [16384, 4 heads x 4096] f32 and its relu, both alive, then the head sum and the bias add.
- 9-13 x 32 MiB: one zero source per layer for the dense mask's set_rows, each a leaf alive from the graph start.

`ggml_qsa_expand_heads` computes, per cell, ((relu(s0) + relu(s1)) + relu(s2)) + relu(s3) + bias + mask in the
unfused graph's order (HIP builds without fast-math, so the sums round the same), and takes the 4D mask as it is.
Result: ub 1024 compute buffer 2,533 -> 909 MiB, peak VRAM 27.8 -> 26.2 GB per card; ub 2048 1,817 MiB (27.5 GB
peak, prefill 1,422); ub 4096 3,633 MiB (30.1 GB peak, prefill 1,542, was OOM). What is left at ub 4096: the score
mul_mat (1 GiB), the expanded per-cell scores (1 GiB, read only by the top-k), the mask copy (512 MiB), the block
bias (256 MiB).

Prefill F32 GEMMs (step 31, `stage_trace.py --kernels`): every one runs a hipBLASLt kernel with an 8x8 or 16x16 macro
tile. The router (M 512, N 1024, K 2560) takes 1.0 ms per call at 2.7 TFLOPS, 1.49 s per GPU per 32k prefill; the
skinny hyper-connection GEMMs (M <= 8 and M = 24, long K) take 216 and 66 us per call, 0.8 s per GPU. rocBLAS's own
Tensile kernels are no better (+1%), so these need our own kernels.


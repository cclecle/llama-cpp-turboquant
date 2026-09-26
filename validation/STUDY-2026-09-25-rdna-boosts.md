# v17: upstream 84e76d8a2 + harvest of stew675/llama-cpp-rdna-boosts (2026-09-25)

## What v17 is

- **Base.** The 60 fork commits of v16, rebased from upstream `718f7b417` onto `84e76d8a2`. That is 246 upstream commits, and the base the rdna-boosts patch set was written for; master is 17 commits further on.
- **Harvest.** Selected changes from rdna-boosts (<https://github.com/stew675/llama-cpp-rdna-boosts>). That repository is not a git fork: it is 16 `git am` patch blocks. The author develops on 3x R9700 (gfx1201, the same GPU as ours).
- **Build.** Same cmake options as v16. The CMakeCache is identical in a full diff.

## Rebase: 5 conflicts

- **`allreduce.cu`.** Upstream #27825 now has its own ROCm AllReduce (host-staged, with a copy-engine path for large tensors). We keep both. `GGML_CUDA_AR_HIP_P2P` selects the internal HIP path:
  - `1` (default, v16 behaviour): our direct-P2P for small reductions, meta butterfly for large ones.
  - `0`: upstream only.
  - `2`: our P2P for small, upstream staged for large.

  Production uses RCCL (the Linux default), which stays the fastest; see the measurements.
- **`llama-context.cpp`.** Kept both our slot-state loader and upstream's `llama_batch_compat`.
- **`qwen4exp.cpp`.** Upstream now loads the hyper-connection norms as `[n_embd, hc]` with `TENSOR_ALLOW_RESHAPE`. Our MTP-draft `flags` are combined with that, and `nextn.hc_head_norm` follows the new shape. Upstream still has no qwen4exp MTP head, so our `cc5d688cb` stays.
- **`test-backend-ops.cpp`, `test-llama-archs.cpp`.** Kept both sides.
- **No SESSION/STATE_SEQ bump and no CLI flag removal.** The slot cache and the presets carry over unchanged.

## Harvested

| Change | Source block | Kill switch |
|---|---|---|
| HIP graphs re-instantiated instead of `hipGraphExecUpdate`, which leaks a kernarg slot per update (ROCm#10713). Capture skipped above 8 tokens; the fork skips every multi-token batch, which would lose MTP-verify replay | 11 | `GGML_HIP_GRAPH_FORCE_UPDATE=1`, `GGML_CUDA_GRAPH_MAX_TOKENS` |
| FA KV split chosen as if n_q = 1 for every n_q <= 8 | 00 | - |
| FA tile_Q `__syncthreads` race (np == 1), wider ncols2 under `-sm tensor`, head-256 ncols-32 row | 04 | - |
| MUL_MAT_ID stays on mmvq for 1..8 tokens on RDNA4 (q6_K 8 tokens 153 -> 105 us) | 13 | - |
| MoE-only wide VDR vec_dots; row (d, m) hoist in the q8_1 MMA vec_dot | 10 | - |
| Native q8_0 K/V in the MMA FA kernel for n_q <= 8, and the **RDNA4 GQA band**: head 256 with GQA 5..8 runs decode/verify on WMMA with the group in ncols2 = 8 and a round-robin KV split over P = nsm blocks. Extended here to F16 caches from n_q = 2 | 15 (+ ours) | `GGML_HIP_FA_BAND_WMMA=0`, `GGML_HIP_FA_BAND_WMMA_F16=0`, `GGML_CUDA_FA_KV_NATIVE=0` |
| `--spec-type draft-mtp-adaptive` (adaptive MTP depth, upstream PR #27210). Opt-in | 01 | not selected by any preset |

### Not taken

- **Band-uniform `nwarps = 1` for RDNA4 mmvq (blocks 08/13).** It costs 8% plain decode on Qwen3.6-35B-A3B: `llama-bench` tg128 77.4 -> 71.0. Upstream's per-type table stays.
- **Their q6_K MMQ hoist.** Same fix as our `3fa5e4929`.
- **Derived KQ mask.** It uses FLASH_ATTN_EXT src[5..7], which collides with our `-nckvc` log-sum-exp in src[5].
- **Others:**
  - fused MoE gate+up MMQ (upstream PR #28702 open) and chunked GDN (#29353/#28447 open)
  - bf16 KV
  - their AllReduce (RCCL beats the internal paths here)
  - qwen4exp block 14 (upstream has qwen4exp)

## Correctness

- **test-backend-ops -o FLASH_ATTN_EXT / MUL_MAT_ID / ADD:** clean. This includes new head-256 GQA-6 f16 and q8_0 cases at kv 512/4096/16384 and n_q 1/3/8.
- **One MUL_MAT q5_1 m=16 n=1 k=32 flake** (ERR 5.6e-4 vs a 5e-4 limit). It passes on 3 re-runs.
- **Full test-backend-ops: 12 reachable failures vs 4 on v16.** All are `MUL_MAT_VEC_FUSION(type=q4_K, glu_op=6 SWIGLU_CLAMP, k=256)`.
  - The dense variant (4 cases) already fails on v16.
  - The MoE variant (`use_id=1`, 8 cases) now fails too.
  - SWIGLU_CLAMP is used only by bailingmoe3, deepseek4, dflash, hy-v4, maple and step35; no fleet model. **Open item.**
- **fork-regress.sh on Qwen3.8-27B-UD-Q5_K_M:** perplexity identical to v16 (3.3698 tensor / 3.3836 layer; -nckvl 8 is 3.3836). The remaining FAILs (`-nkvo` perplexity did not run, checkpoints server start) are the same on v16, so the suite needs refreshing.
  - **Trap:** run it on a full build tree. Its perplexity corpus is `$BUILD/../README.md`, and without it the fallback text gives ppl ~1.0, which looks exactly like an attention mask leak.

## Performance

All tg numbers are pooled over 6 prompts (scripts/fleet/multi-prompt.sh), with each rung's exact router child args and the same GPU. "Deep" means MP_CHARS=120000, about 35k tokens of context.

| Rung | Context | v16 | v17 | Delta |
|---|---|---|---|---|
| Qwen3.8-27B:S:SHQ (f16 KV, MTP n5) | 2.5k | 44.9 | 44.4 | noise |
| Qwen3.8-27B:S:SHQ | deep | 39.9 | 43.3 | **+8.5%** (band off: 39.9) |
| Qwen3.8-27B:S (q8_0 KV, MTP n5) | 2.5k | 48.2 | 49.0 | +1.7% |
| Qwen3.8-27B:S | deep | 39.9 | 44.2 | **+10.8%** (band off: 40.1) |
| Qwen3.8-27B-TurboFable:HQ:M (Q8_0, 2-GPU tensor) | 2.5k | 58.2 | 58.0 | noise |
| Qwen3.8-27B-TurboFable:HQ:M | deep | 54.1 | 56.1 | +3.7% |
| Qwen3.8-Flash-Next:M (2-GPU tensor, UVA) | 2.5k | 36.5 / 37.6 | 39.6 / 40.4 | **+7.4..8.5%** |
| Qwen3.8-Flash-Next:M | deep | 34.3 | 36.0 | +5.0% |
| Qwen3.6-35B-A3B:S:SHQ (CPU experts) | 2.5k | 74.3 | 74.8 | noise |

- **Op level, gfx1201, head 256 GQA 6, kv 16k (us).**

  | n_q | q8_0 tile/vec (v16 path) | q8_0 band | f16 tile/vec | f16 band |
  |---|---|---|---|---|
  | 1 | 226 | 126 | 92 | 128 (not used at n_q = 1) |
  | 3 | 423 | 127 | 213 | 130 |
  | 8 | 551 | 194 | 348 | 187 |

- **Adaptive MTP on 27B:S:SHQ, cap 5 vs fixed n_max 5:** 44.8 -> 46.3 t/s (+3.3%), acceptance 65% -> 75%. Cap 8 OOMs building the MTP context on this rung. **Not applied to presets: user decision.**
- **2-GPU tensor AllReduce on 27B Q8_0, pp4096 / tg64:**

  | AllReduce | pp4096 | tg64 |
  |---|---|---|
  | RCCL (default) | 2210 | 30.1 |
  | internal P2P=1 | 1768 | 30.3 |
  | internal P2P=0 | 1881 | 30.2 |
  | internal P2P=2 | 2037 | 30.2 |

  RCCL stays.

## Tools added (scripts/fleet)

`multi-prompt.sh` now takes:

- `MP_NO_RESTART=1`: run a series without restarting production each time.
- `MP_ENV_A` / `MP_ENV_B`: per-arm environment.
- `MP_CHARS`: prompt size.
- `MP_ARGS_B`: a second configuration on the same build.

## Full rung sweep (2026-09-26)

Every rung of both preset stores ran once through a scratch router that mirrors its unit (`scripts/fleet/rungsweep.sh`):

- The prompt filled half the per-slot context (8k to 142k tokens).
- Each run generated 256 greedy tokens.
- VRAM was read after generation.
- The text went through a gibberish check.
- SINGLEGPU ran on both cards at once.

| Store | Rungs | OK |
|---|---|---|
| DUALGPU | 101 | 101 |
| SINGLEGPU | 219 | 204 |

The 15 SINGLEGPU non-OK rungs were all Devstral (see below). Raw lines: `/root/work-20260926/all.txt` on the box.

### Found and fixed

- **`-nckvc` under `-sm tensor` aborted** (`GGML_ASSERT ret.axis != UNKNOWN`) as soon as a prompt crossed into the host KV cells.
  - Hit on Mistral-Small-4:S at 32768 tokens per slot. It would hit Qwen3.8-27B-TurboFable:HQ at 172k.
  - **v16 has the same bug** (Qwen3.8-27B `-sm tensor --n-cpu-kv-cells 4096 -c 8192` aborts on v16).
  - Cause 1: the merge's log-sum-exp was a standalone leaf, so the tensor-split backend mirrored it while the FA output was split by head.
  - Cause 2: host KV copies were always split by head, even for a model whose attention is mirrored (the EAGLE draft of an MLA target).
  - Fix `fe49693b0`:
    - lse = `cont(permute(sum_rows(q)))`, which inherits q's split, and FA overwrites it;
    - host KV copies ask the model callback first;
    - split states of unallocated tensors are memoized per query, since the recursion over them was exponential.
  - PPL layer / layer+nckvc / tensor / tensor+nckvc: 1.4423 / 1.4438 / 1.4435 / 1.4427.
  - `fork-regress.sh` now checks the crossing under both split modes.
- **gemma-4 MTP assistant abort** in `llama_hparams::is_recr`. It was introduced by the fix above: the assistant reads the target KV, whose cache names carry target layer indices. The callback now leaves layers the model lacks to the caller (`2788adcda`).
- **Devstral-Small-2: garbage on every rung**, byte-identical on v16. Its 256k window comes from its own YaRN (factor 48 over an 8192 original context). The SINGLEGPU `[*]` `rope-scaling = none` stripped it.
  - Fixed in the preset repo (`/opt/llamacpp-config`, commit `0ffa8c9`, `rope-scaling = yarn` on its 16 rungs). Re-checked: all OK.
  - `scripts/fleet/ctxcheck.py` flags any rung whose per-slot context exceeds the usable context. Only Devstral, plus the deliberate Muse-Glimmer `override-kv`.

## WMMA above head 256 (rejected)

The head 320/512/576 instances were compiled but kept off AMD by a kernel guard. Enabling them with the fork's gfx1201 rows (`pp512` t/s):

| Model | d0, v17 | d0, WMMA | d32768, v17 | d32768, WMMA |
|---|---|---|---|---|
| gemma-4-31B, 1 GPU | 1227 | 1174 | 401 | 353 |
| gemma-4-31B, 2 GPU tensor | 2003 | 1957 | 701 | 663 |
| gemma-4-26B-A4B | 4054 | 3985 | 1444 | 1356 |
| GLM-4.7-Flash | 3401 | 3404 | 864 | 862 |
| Mistral-Small-4 (head 320) | | failed to bench | | |

- FLASH_ATTN_EXT was 3846/3846 with it on.
- The default cap is 256 again. `GGML_CUDA_FA_WMMA_MAX_HEAD` opts in.

## r9 (fork release r9): typed MMA K/V store

- Upstream 1884824fd (the FA swizzle refactor) is in this base and not in v16. It made the generic MMA K/V loader store through a `char *` even without swizzling, so HIP splits the 16-byte shared store. The fork measured 2-6% prefill, and 14-40% on head-256 MMA prefill.
- Ported in `d6d8dbfc9`, **reverted in `93759983c`**: on this tree it is slower everywhere.
- `r9.sh`, pp4096 t/s at d0 / d32768, v17 vs r9 (the only head-256 difference is the store):

| Model | v17 d0 | r9 d0 | v17 d32768 | r9 d32768 |
|---|---|---|---|---|
| Qwen3.8-27B, q8_0 KV | 1130.3 | 1113.5 | 874.6 | 874.5 |
| Qwen3.8-27B, f16 KV | 1134.2 | 1131.5 | 908.9 | 882.8 |
| Qwen3.6-35B-A3B | 2041.3 | 2039.6 | 1724.6 | 1711.1 |
| Qwen3.5-9B | 4214.3 | 4187.6 | 3072.2 | 3057.0 |
| gemma-4-31B | 1064.4 | 1054.4 | 395.1 | 393.8 |
| GLM-4.7-Flash | 2994.0 | 2992.0 | 838.1 | 836.4 |
| Devstral-24B | 1241.1 | 1238.4 | 835.3 | 833.9 |
| TurboFable-HQ, 2 GPU tensor | 2196.4 | 2179.2 | 1731.4 | 1661.2 |
| Mistral-Small-4, 2 GPU tensor (pp512) | 152.3 | 149.2 | 115.0 | 113.8 |

- FLASH_ATTN_EXT: 4052/4052 on both builds.

### r5 f16 band settings (op level)

`test-backend-ops perf -o FLASH_ATTN_EXT`, head 256, GQA 6, kv 16384, us/run (lower is better). The f16 n_q = 1 row
is not in the band (vec kernel).

| n_q | f16 ncols1=4 (default) | f16 ncols1=2 | f16 ncols1=2, P=48 | q8_0 ncols1=4 (default) | q8_0 ncols1=2 | q8_0 ncols1=2, P=48 |
|---|---|---|---|---|---|---|
| 1 | 94.6 | 94.3 | 94.2 | 134.1 | 83.9 | 74.9 |
| 3 | 134.5 | 122.2 | 122.8 | 135.7 | 141.8 | 125.1 |
| 5 | 189.4 | 141.0 | 168.2 | 205.9 | 168.4 | 169.7 |
| 8 | 192.6 | 180.9 | 203.7 | 208.6 | 221.5 | 219.4 |

ncols1 = 2 wins almost everywhere; q8_0 single-token decode drops 134 -> 75 us per attention layer. Taken up as a
per-width choice below.

## Attention memory: derived kq mask, native q8_0 prefill, band retune (2026-09-26)

Scratch tree `/opt/llamacpp/tmp-attn`, runner `validation/v17-scripts/attn.sh`.

- **Derived kq mask** (`29b12c106`, block-15 V3 extended to M-RoPE): on by default, `LLAMA_KQ_MASK_DERIVED=0` off.
  Applies to single-stream caches (np = 1 rungs; `kv-unified = 0` with np > 1 keeps the packed mask).
- **Native q8_0 prefill** (`0e54762e4`): opt-in, `GGML_CUDA_FA_KV_NATIVE_PREFILL=1`.
- **Band retune** (`6aeda8cad`): ncols1 = 2 for all widths, P = 3/4 nsm for q8_0.

### Correctness
- FLASH_ATTN_EXT 4082/4082 on ROCm0 (32 new derived cases), also with native prefill on.
- Perplexity (4 chunks of 4096, one sequence per batch) identical to the last digit, derived off vs on:
  27B q8_0 3.2225, 27B f16 3.2234, gemma-4-31B 37.4026, 35B-A3B 3.7238, Devstral 3.7896, 27B `-nckvc 2048` 3.2201,
  27B `-sm tensor` 3.2217, gemma-4-31B `-sm tensor` 37.5796. Native prefill: 27B 3.2225.

### Memory (server load, compute buffer MiB)
| Load | GPU | host (pinned) | VRAM per GPU |
|---|---|---|---|
| 27B c131072 ub1024 q8_0, packed | 928 | 296 | 24.78 GB |
| + derived | 673 | 41 | 24.52 GB |
| + native prefill | 245 | 41 | 24.07 GB |
| gemma-4-31B c131072 ub1024, packed / derived | 1455 / 1196 | 303 / 44 | 25.74 / 25.47 GB |
| 27B `-sm tensor` c262144 ub1536 f16, packed / derived | | 828 / 62 | 19.77 / 18.97 GB each |

### Speed
Prefill pp4096 t/s, d0 / d32768 (same build, env A vs B; same-config run-to-run noise ~0.5-1%):
| | off | derived | derived + native prefill |
|---|---|---|---|
| 27B q8_0 | 1168 / 914 | 1152 / 916 | 1148 / 881 |
| 27B f16 | 1152 / 920 | 1154 / 920 | |
| gemma-4-31B | 1093 / 399.6 | 1093 / 398.7 | |
| 35B-A3B | 2919 / 2340 | 2927 / 2349 | |
| 27B `-sm tensor` | 1902 / 1521 | 1889 / 1525 | |

The derived mask is speed-neutral. Native prefill costs 3.7% at d32768: a VRAM-for-speed trade per rung, so opt-in.

Decode tg64 t/s, v17 -> attn (band retune), d0 / d16384 / d65536:
- 27B q8_0: 25.61 / 24.26 / 21.63 -> 25.81 / 24.90 / 22.50 (+2.6% / +4.0% deep)
- 27B f16: even (n_q = 1 f16 is not in the band); 35B-A3B: even.

Tooling fix found here: `benchab.sh` picked the env by build name, so an A/B of one build against itself ran A twice.

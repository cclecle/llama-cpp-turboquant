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
- Fixed in `d6d8dbfc9`.
- Results: see below once `r9.sh` finishes.

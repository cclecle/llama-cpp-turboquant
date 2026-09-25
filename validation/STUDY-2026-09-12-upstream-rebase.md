# Rebase study - upstream 8b4b3558f -> 718f7b417

**Date:** 2026-09-12 - **Fork branch:** `rebase/upstream-20260904` at `118ec2b8d` (v15, in production)
**Base:** upstream `8b4b3558f` (2026-09-04) - **Target:** upstream `718f7b417` (2026-09-12)
**Gap:** 115 upstream commits in 8 days, 64 fork commits to replay, 48 files touched by both sides.

---

## Verdict

Feasible and worth doing. Only 4 files conflict at the git level and every resolution is
mechanical. What needs care is not the merge but the cutover: three things change fleet
behaviour even though they merge clean, and one of them stops every preset from loading.

1. `--mmap` / `--mlock` / `-dio` are REMOVED, not deprecated. Both production `[*]` blocks
   carry `mmap = 1` and `mlock = 0`. The router loads presets with `ignore_unknown_keys=false`,
   so llama-server will throw at startup. Preset change required: `load-mode = mmap`.
2. Flash-Next slot files change layout with no version bump (indexer V section gone). The new
   build reads old files, the old build cannot read new ones. One-way, safe on this box because
   all three instances cut over together.
3. Gated-delta-net normalisation changes from `max` to `rsqrt`. Small output change for every
   Qwen 3.5+ family on the fleet. Ladders need a re-run, not a re-tune.

Plus one decision, now measured (section 9): upstream's native flash attention for gfx1201
beats the fork's rocWMMA path at every depth on both head sizes. DROP rocWMMA in this rebase.
That deletes six fork commits, one build flag, two of the four conflicts and 36 preset lines.

---

## 1. Must do before cutover

| Item | Commit | What breaks | Fix |
|---|---|---|---|
| Preset keys removed | `14a9d09f7` | `mmap = 1`, `mlock = 0` in `SINGLEGPU/main.ini:15-16` and `DUALGPU/main.ini:28-29` -> "option not recognized" at startup | replace both lines with `load-mode = mmap` |
| Flash-Next file layout | `311d4211b` | old binary cannot read files written by the new one | cut all 3 units over together (already the practice) |
| GDN numerics | `5fdfa6282` | output differs slightly on qwen35 / qwen35moe / qwen4exp | re-run ladders for Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B, AgentWorld, Flash-Next |

`load-mode` default is `auto` = mmap unless a listed device lacks mmap support. The iGPU is
excluded from the device list when dGPUs exist, and the units pin `HIP_VISIBLE_DEVICES` to the
dGPUs, so `auto` resolves to mmap here. Writing `load-mode = mmap` explicitly makes it immune
to a future `-dev` change, which is what the user wants.

## 2. Slot-file impact per family

| Family | Impact | Why |
|---|---|---|
| Flash-Next (qwen4exp) | new reads old, old cannot read new | indexer cache no longer has V tensors; `state_write_data` skips null V, so the section disappears from the end of the payload. Server locates the v5 trailer through the footer, not through `nread`, so the leftover bytes in an old file are ignored |
| GLM-4.7-Flash (deepseek2) | none | `5cdd3d1da` changes the NextN KV filter, but the fleet GGUF has no `nextn_predict_layers` key (verified, all 25 `deepseek2.*` keys listed). `n_layer_nextn == 0`, no change |
| every other family | none | no version constant moved, `llama-kv-cache.cpp` untouched upstream |

Neither `LLAMA_SESSION_VERSION` (10) nor `LLAMA_STATE_SEQ_VERSION` (3) moves. The 28 GB on
`/mnt/cache` stays valid.

## 3. Git-level conflicts

Four files. Everything else in the 48-file intersection auto-merges.

| File | Fork side | Upstream side | Resolution |
|---|---|---|---|
| `ggml/src/ggml-cuda/fattn.cu` ~645 | `f90f330d8` comment "do NOT raise the cap to 256" | `16378d93f` raises the cap to 256 and fixes the prune the comment cites | DROP `f90f330d8` from the replay |
| `ggml/src/ggml-cuda/fattn.cu` ~688 | MLA_DECODE / MLA_PREFILL cases in `get_alloc_size` | `5a4d0feca` rewrites the adjacent VEC case | keep both |
| `ggml/src/ggml-cuda/fattn-common.cuh` ~1135 | `if (stream_k_use)` LSE routing | `16378d93f` replaces the stream-k heuristic just below | `if (stream_k_use) { auto should_use_stream_k = ...; }` |
| `common/speculative.cpp` ~2467 | 7-line comment + `if (!params_spec.devices.empty())` guard | `415e909d8` same guard + single-device LAYER block | take upstream's block, keep the fork's comment |

Two more land inside fork-modified functions but merge clean; read them after the rebase:
`311d4211b` in the indexer ctor lambda (`llama-memory-hybrid-idx.cpp`, 6 lines above our
`n_cpu_kv_*` args) and `5d806aa25` in `create_checkpoint()` (10 lines above our `update_dft`).

## 4. Fork commit to drop

`f90f330d8 cuda(hip): warn against raising the AMD WMMA head-size cap`. Six lines of comment
saying the DKQ=256 mma instance has no device code on gfx1201. Upstream `16378d93f` gives it
device code (`DKQ > 128` -> `DKQ > 256` in `fattn-mma-f16.cuh`). The warning becomes false.

## 5. Behaviour changes that merge clean

Ranked by fleet impact.

1. `16378d93f` FA tuning for gfx1201. MEASURED 2026-09-12, see section 9: upstream's native
   path beats rocWMMA at every depth on both head sizes. Decision: DROP rocWMMA. That removes six
   fork commits (`561fcf75f`, `c91ad66c2`, `542a58b07`, `615748367`, `f90f330d8`, `e459a61f0`),
   the `GGML_HIP_ROCWMMA_FATTN=ON` build flag, two of the four git conflicts in section 3, and
   the `rocwmma-fattn = off` lines in the Magistral and Devstral presets (36 lines; they become
   unknown keys once the flag is gone, same failure as `mmap`).
2. `82d6bb284` router child lifecycle rewritten: one monitor thread, poll-based, line-framed
   stdout, deadline force-kill. Same HTTP surface, same protocol. Soak test load / unload /
   sleep / wake with `--models-max 1`.
3. `d4abd573f` MoE MMQ J-tile chosen from `ceil(ne12 * n_expert_used / ne02)` on RDNA4. Hits
   the moe-hybrid batched path too (`n_expert_used = 1`). Re-run MoE prefill validation.
4. `5d806aa25` checkpoint spacing eviction only when the list is nearly full. Restored
   checkpoints have `id_task = -1` and used to be evicted on the next create; they now survive.
   Net gain for hybrid / SWA resume after a slot restore.
5. `73ab7599b` branchless Q4_K / Q5_K unpack in `vec_dot`, unguarded, so it runs on gfx1201 for
   batch>1 Q4_K and all-batch Q5_K mmvq. Bit-identical; spot-check throughput.
6. `5a4d0feca` `GGML_CUDA_FA_ALL_QUANTS` is deprecated but still honoured (warns each
   configure). `GGML_CUDA_FA_QUANTS=all` is the same 49 instances. `scripts/validate.sh:107` and
   `scripts/fork-regress.sh:129` grep the OLD cache key; update them if the flag is migrated.
7. `992cb503c` fewer scheduler splits in multi-backend graphs. Helps HIP graphs.
8. `415e909d8` `-mmdev` defaults to the first `--device`; a single `-devd` forces LAYER split.
   No preset sets either key, so no effect today.
9. `718f7b417` httplib 0.56: rejects a request carrying both `Transfer-Encoding` and a
   non-zero `Content-Length`. No new OpenSSL requirement.
10. `3bcfeb700` PCH + unity build for `src/models/*.cpp`. Never touches ggml-cuda / ggml-hip.
    No duplicate file-scope symbols in the fork's model files.

## 6. Verified not affected

- iGPU (gfx1036): `f3f1a8f27` flips lazy mode off when a non-mmap device is in the list, and
  `d4389a4dd` hard-codes `integrated = false` on HIP. Neither reaches a unit that pins
  `HIP_VISIBLE_DEVICES=0`, `=1`, `=0,1`.
- The two most recent fork commits (`27bc05592` indexer restore fix, `118ec2b8d` temp name):
  upstream did not touch `llama-kv-cache.cpp` or `server-slot-io.cpp`. `311d4211b` composes
  with the restore fix because `state_read_data` already skips null V.
- qwen4exp MTP head: not landed upstream. `947e85a52` stays.
- `304665fe7` IQ MoE is SYCL only. `9dcf84e5a` Kimi-K3 does not touch recurrent serialisation.
- `895c045fd` chat parser split: fork touches none of those files.
- `-nckvc` refusal for qwen4exp, `kv_cells_unsupported`, MSA / DSA: untouched upstream.

## 7. Re-validation after rebase

- Build: same configure line as v15 (byte-identical CMakeCache diff), plus the mmap preset change.
- `test-llama-archs` row count (not exit code), qwen4exp save/load roundtrip 0.00e+00.
- Ladders: the five GDN families (section 1), one round each, expecting a small drift not a
  regression.
- FA decision: rocWMMA on vs off on Qwen3.6-27B and Mistral-Small-4 (section 5.1).
- MoE prefill on Qwen3.6-35B-A3B and AgentWorld (section 5.3).
- Slot state: save on Flash-Next `:S`, restore into `:M` and `:L`, non-zero slot; then one dense
  family round trip. Both must show `restored draft state` + `loaded N context checkpoints`.
- Router soak: 20 load / unload cycles with `--models-max 1`, one forced sleep and wake.

## 8. Plan

1. Pin `refs/rebase-target/20260912` = `718f7b417`. Backup branch `backup/pre-upstream-rebase-20260912`.
2. Branch `rebase/upstream-20260912`, replay the 64 commits with the gate refs as before,
   dropping `f90f330d8`.
3. Resolve the four conflicts in section 3. Re-read the two clean-but-adjacent hunks.
4. Preset change in both `main.ini` `[*]` blocks. Deploy the ini BEFORE the binary is started
   (the old binary accepts `load-mode` already, so the order is safe either way).
5. Build v16 on the box from the same configure line. Full CMakeCache diff against v15.
6. Validation list in section 7. Cut over all three units together.

## 9. Measured: native WMMA vs rocWMMA on gfx1201 (2026-09-12)

Setup: v15 tree plus the three upstream FA commits (`b74f590ea`, `5a4d0feca`, `16378d93f`)
cherry-picked, built in `/tmp/fa-bench-20260912` with the exact v15 configure line. Same
binary, flipped only by `--rocwmma-fattn 0|1`. The v15 production `llama-bench` ran the same
matrix as the old-native baseline. GPU 0 only (R9700 32 GB), units stopped, `-fa 1`, q8_0 KV,
`ub 1024 b 4096 t 7`, 4 reps. Run-to-run noise measured by rocWMMA-on on both binaries
(unchanged code): 0.2 to 2.1 percent.

Qwen3.6-27B Q5_K_S, head size 256, prefill tokens/s. tg64 identical in every cell (vec kernel).

| depth | rocWMMA (prod today) | v15 native (TILE) | new native (tuned) |
|---|---|---|---|
| pp4096 @ 0 | 1079 | 1031 (-4.5%) | 1090 (+0.9%) |
| pp4096 @ 16384 | 868 | 676 (-22%) | 947 (+9.1%) |
| pp4096 @ 32768 | 721 | 504 (-30%) | 835 (+15.8%) |

Magistral-Small Q6_K, head size 128, prod already runs `rocwmma-fattn = off`.

| depth | v15 native (prod today) | new native (tuned) | rocWMMA |
|---|---|---|---|
| pp4096 @ 0 | 1209 | 1228 (+1.5%) | 1176 (-2.7%) |
| pp4096 @ 16384 | 864 | 906 (+4.9%) | 694 (-20%) |
| pp4096 @ 32768 | 662 | 719 (+8.7%) | 476 (-28%) |

Reading: on v15 rocWMMA was genuinely needed for D=256 (native TILE was 30% behind at 32k,
which is the +36% the fork recorded). Upstream's tuning moves native D=256 from 30% behind to
16% ahead. D=128 gains 5 to 9% at depth from the same commit with no preset change. Nothing
on this fleet still needs rocWMMA.

Not covered by this bench, add to section 7: Mistral-Small-4-119B (MLA D=320, rocWMMA already
rejected that shape, so TILE either way, but the fork's "unprune TILE 320/256" commit goes with
rocWMMA), gemma-4 (D=128, SWA, expected to track Magistral), Qwen3.8-27B (D=256, expected to
track Qwen3.6-27B).

## 10. Rebase executed (2026-09-12)

Branch `rebase/upstream-20260912` on `718f7b417`, pinned as `refs/rebase-target/20260912`.
Backup `backup/pre-upstream-rebase-20260912` = `118ec2b8d`. 58 of 64 commits replayed by
cherry-pick in order; the six rocWMMA commits dropped. Five conflicts, all mechanical:

| Commit | File | Resolution |
|---|---|---|
| `9606670aa` spec draft inherits --device | `common/speculative.cpp` | upstream guard + single-device LAYER block, fork comment kept |
| `23a432009` MLA decode kernel | `fattn.cu` get_alloc_size | fork MLA_DECODE case + upstream VEC case |
| `01fb7c068` MLA prefill kernel | `fattn.cu` get_alloc_size | fork MLA_PREFILL case + upstream VEC case |
| `69bbc33d1` --op-offload-min-batch | `common/arg.cpp` | drop the `--rocwmma-fattn` option it was stacked on |
| `cf749a2bc` LSE output | `fattn-common.cuh`, `fattn-wmma-f16.cu` | `if (stream_k_use)` around upstream's lambda; kernel file deleted |

Drift check, per-commit diffstat old vs new: only those three commits differ, each by exactly
the dropped rocWMMA lines. No fork reference to rocWMMA remains outside upstream's own CI line
and TODO. `scripts/validate.sh` T0.1 already required `GGML_HIP_ROCWMMA_FATTN` to be absent.

v16 built at `/opt/llamacpp/llama-cpp-mine-v16/build3`: 114 targets, 0 errors. CMakeCache diff
against v15 is exactly `GGML_HIP_ROCWMMA_FATTN:BOOL=ON` removed and `GGML_CUDA_FA_QUANTS`
added (forced to `all` by the deprecated flag). `--rocwmma-fattn` is gone from the binary.

Presets rewritten (pre-change copy in `/opt/backups/backup-pre-v16-20260912-132031/llamacpp-config/`): `mmap = 1` ->
`load-mode = mmap` and `mlock = 0` deleted in both `main.ini`; 37 `rocwmma-fattn = off` lines
deleted from Magistral-Small, Devstral-Small-2-24B and Mistral-Small-4-119B-A6B. Both stores
parse under v16 (205 + 96 sections, no unrecognised key) and still parse under v15, so a
rollback needs no preset change.

v15 indexer restore fix, checked in production before carrying it forward: since the
2026-09-06 restart, Flash-Next restored at `n_slots` 2 and 4 into slots 0 to 3 with draft
state and checkpoints, zero `n_stream mismatch`. The only failures are a missing file and a
state larger than the target rung.

## 11. v16 validation (2026-09-12)

**Fleet smoke, `fleetcheck-v16.sh`: 27 of 27 OK** across both stores (one mid rung per family).
VRAM per rung against the v14 reference log: largest increase +0.08 GB (Muse-Glimmer), most
within 0.05 GB, all inside the 300-500 MiB margin policy. The footprint concern for the kernel
change is closed at the mid-rung level; the full 148-rung fit check was not run.

**Decode: two apparent regressions, both chased to ground and both NOT real.**

The reference-log comparison showed gemma-4-31B:M at -11% tg and Qwen3.6-35B-A3B:HQ:M at
-10% tg, reproducible to 0.2 t/s across passes. Bisection:

| step | result |
|---|---|
| v15 + only the 3 FA commits (rocWMMA still on) | identical to v15 -> FA commits are not the cause |
| same tree, rocWMMA forced off | gemma reproduces (46.7), Qwen does not (136.1) -> two different sources |
| llama-bench, batch 1..128 at depth 2048, both shapes | no regression anywhere -> not kernel throughput |
| preset variant without MTP | no regression on either -> lives inside speculative decoding |
| draft KV split off | still regresses -> not the LSE/TILE path |
| rocprofv3, one request, shrunk context | v16 GPU time +1%, FA work cheaper -> and the gap vanished |
| context / device-bank sweep | tg tracks the draft acceptance counts, and generated token counts differ under greedy |

Under greedy decoding a numeric shift of a few ulps (the FA kernel for gemma, the GDN `rsqrt`
change for Qwen) picks a different text, and speculative acceptance depends on the text. One
fixed prompt cannot separate that from a regression, so six prompts were run at production
geometry on the exact child command lines the router builds:

| rung | v15 pooled | v16 pooled | per-prompt range |
|---|---|---|---|
| gemma-4-31B:M | 60.9 t/s, 67% acc | 60.1 t/s, 66% acc | -11% to +3% |
| Qwen3.6-35B-A3B:HQ:M | 135.5 t/s, 52% acc | 132.8 t/s, 51% acc | -11% to +9% |

Pooled differences of 1.3% and 2.0%, per-prompt swings of 20 points on the same binaries. The
fleet check's single 2k-token prompt landed on the unlucky side for both. Lesson for the
harness: a speculative-decoding tg number from one greedy prompt is a trajectory sample, not a
throughput measurement; pool several prompts before reading a difference under 10%.

**Prefill**: no regression anywhere in the same-session A/B; the reference-log drops
(gemma-4-12B -28%) were day-to-day swing, +1% when re-measured back to back.

**Artifacts on the box**: `/root/fleetcheck-v16.log`, `/root/ab-*.log`, `/root/geo-sweep.log`,
`/root/multi-prompt*.log`, `/root/prof/{v15,v16}/k_results.db`, `/tmp/fa-bench-20260912`
(v15 + FA commits, both `llama-bench` and `llama-server`), `/tmp/cfg-diag` (preset variants).
These were scratch locations and may be gone; the tools that produced them are versioned in
`scripts/fleet/`.

**Cutover 2026-09-12 13:20**: the 3 units repointed to v16 (full config snapshot in `/opt/backups/backup-pre-v16-20260912-132031/`: units, preset store, proxy json),
reloaded and restarted together. All listening, 0 errors, 205 / 205 / 96 presets parsed, a model
loaded and served through the router. Rollback: `cp -a` the three unit files from that snapshot, `daemon-reload`, restart;
the presets parse under v15 as well, so they stay as they are.

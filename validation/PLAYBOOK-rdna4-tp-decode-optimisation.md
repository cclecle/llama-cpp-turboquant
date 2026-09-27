# Playbook: optimising speculative decode on 2x RDNA4 with `-sm tensor`

Lessons from the v19 campaign (Qwen3.8-Flash-Next against R9V, 2026-09-26), written to be reused on other models
and later campaigns. The measurements and the chronology are in `STUDY-2026-09-26-flashnext-r9v.md` (section 11);
this file is the part that should outlive that study.

Result so far: same-config decode **68.0 → 47.6 ms per speculative step (-30%)** at step 16, prefill +10%. Decode is
deterministic since step 14 (same text on every load), and repeat loads now agree to ±0.1 ms. R9V (vLLM fork) does
the same step in 41 ms.

## 1. Measure the right thing

- **The decode metric is ms per speculative step, not tokens/s.**
  - `validation/v19-scripts/step_ms.py <label prefix>`
  - steps = generated tokens − accepted draft tokens.
  - Acceptance moves with the generated text, and the text changes from one load to the next (0.47-0.51 here). A real
    -6% step change once showed up as +1.3% t/s.
- **A B B A over separate loads,** at least 2 per arm.
  - One load swings about ±6%.
  - Use env toggles on one build for the A/B (every change here has one).
  - To compare two builds, copy the old `build3/bin` and run it with `LD_LIBRARY_PATH` (`flashnext.sh FN_LD=`): the
    RUNPATH is absolute and points into the tree it was built in.
- **Warm cache:** the 28.8 GB PLE table is memory-mapped, and a cold page cache halves prefill. Anything that reads big
  files evicts it (perplexity over the GGUF, large copies).
- **Perplexity for sparse-attention changes:** use a context longer than the selection budget. At 2k the QSA indexer
  keeps every visible cell, so dense and sparse cannot differ. Use ub 8 (the decode path) and 8k context.
- **Rounding:** fused kernels change rounding, and a top-k selection turns rounding-level score changes into different
  kept blocks. Expect perplexity to move by ~0.3% (well inside ±0.025) and treat that as noise, not as a bug.
- **Profilers add ~9-12 ms per step,** mostly to idle time. Compare traced with traced, and busy time with busy time.
- **Look at the individual loads before calling a regression.** Step 11 looked 3 ms slower: its first run followed a
  36 s cold load, and the next A/B showed no effect.
- **Check the text for determinism.** Since step 14, the same build generates the same text on every load. A change
  that should not alter the math must keep it byte for byte, and two loads of any build must agree.

## 2. Read a trace before believing a hypothesis

The tools, all in `validation/v19-scripts/`:

| Tool | What it answers |
|---|---|
| `decode-profile.sh` | rocprofv3 kernel trace of one 32k-prompt request |
| `torchtrace_anatomy.py rocprof <csv> <steps>` | per-step kernels, busy, idle, and the top kernels (also vLLM torch traces) |
| `kernel_ngrams.py <csv> 3` | the most frequent 1/2/3-kernel sequences per verify pass, i.e. the fusion targets |
| `hostprof.sh` (`HP_PROMPT`, `HP_WAIT`) | gdb stack samples of the server (the rig has no perf) |
| `GGML_CUDA_MOE_STATS=1` (+ `GGML_CUDA_DISABLE_GRAPHS=1`) | per MoE call: (token, slot) pairs, distinct experts, cold ones |

Analyses that paid off (the code is in the study log; they are short python over the kernel CSV):

- **Pair the AllReduce kernels of both GPUs by order.**
  - End skew about 0 with start skew wide means one GPU waits for the other.
  - Classify the preceding subgraph (MoE / attention / GDN) and correlate the skew with the difference in GPU time.
  - Here the correlation was 1.00 for the MoE, which means uneven work, not launch skew. An "in-graph AllReduce" would
    not have fixed it.
- **Per-layer kernel durations on both GPUs:** a constant 1.5x ratio that alternates sides layer by layer means an
  uneven tensor split (section 3).
- **Split each step into verify pass, draft passes and GPU-idle gaps** (evaluations separated by > 150 us idle). Use
  medians: a few 10-20 ms host hiccups dominate the means. Here the four MTP drafts plus their gaps were 21% of a step.
- **Host profile:** if the main thread sits in `ggml_backend_cuda_synchronize` in almost every sample, the GPU idle gaps
  are wake-up and launch latency around each host sync plus blocking input uploads, not host compute. The lever is
  fewer syncs per step, not faster host code.

## 3. Traps in this fork (each cost a measurement)

- **Tensor-split granularity makes uneven splits.**
  - `get_split_granularity` (llama-model.cpp) used `lcm(block, 128)` for FFN tensors, so a width of 640 = 5 units split
    384/256, and the per-layer rotation swapped the sides.
  - Fixed: with an even split, the unit halves while the width does not divide. Check any model whose FFN, expert or
    shared-expert width is not a multiple of 256.
- **`ggml_cuda_can_fuse` is a whitelist.**
  - A new `ggml_cuda_try_fuse` pattern must also be listed there, or it silently never fires.
  - `test-backend-ops` still passes, because the unfused path is correct. Verify with a trace (kernel counts).
- **RDNA4 mmvq launches one warp per row for 2-8 columns.**
  - A matrix with few rows and a long K runs at ~240 GB/s (hyper-connection down projection 10240 → 320).
  - Fixed: the `halve_iters` variant gives 4 warps, `GGML_CUDA_MMVQ_TALL_K`.
  - There is no batch invariance to protect: 1 column already uses 8 warps.
- **BF16 mat-vec stops at 3 columns on AMD,** so `mul_mat_f` took the 5-column verify. It launches one block per 32 rows:
  4 blocks for a 128-row matrix. Now 5 columns on RDNA4, like F16.
- **`rope_multi` launched one row per 256-thread block,** which wastes 3/4 of each block on 128-wide rows (81 us for
  8,192 rows). Rows are now packed; the kernel has no row bound check, so packed rows must divide the row count.
- **Fused GLU mat-vecs were one-column only for dense weights.** The kernel handled any width; the launcher and
  `ggml_cuda_should_fuse_mul_mat_vec_q` did not. Now q8_0 at 2-8 columns.
- **The fork's `mmvk` (RDNA4 K-quant mat-vec, batch 1) skipped the SWIGLU_CLAMP gate.** It predates upstream's op.
  Grep new GLU ops into every fused kernel.
- **Upstream's sparse flash-attention mode is NVIDIA-only.** Explicit K/V lists (`ggml_flash_attn_ext_add_kv_idx`,
  src[8]) now reach the mma kernel on RDNA WMMA at `<256,256,1,16>`. A new template instance is needed per shape
  (`generate_cu_files.py`).
- **The vec flash-attention kernel reads K/V once per Q head.** A single-query decode with 12 Q heads per K/V head read
  a 32k q8_0 cache 12 times (210 us). The mma kernel at (1, 16) reads it once (`GGML_HIP_FA_GQA16`).
- **Backend sampling is refused under `-sm tensor`** (`llama_context::set_sampler`). Every draft token then copies the
  logits and runs the generic CPU chain, which builds a candidate array of the whole vocabulary. For MTP drafts only
  the top candidate and its probability are used: `spec_top_k_logits` (`LLAMA_SPEC_FAST_TOPK`).
- **The meta backend cannot reduce along a split row.** `handle_per_row` asserts that axis 0 is not split, and the
  output head is split by vocabulary rows. So there is no in-graph argmax or top-k over logits under `-sm tensor`. That
  blocks both an unrolled multi-draft graph and R9V's coarse draft head (rank with 2 of 10 Q6_K blocks, rescore exactly)
  until the meta backend gains a cross-device top-k merge.
- **Under `-sm tensor` the scheduler called `graph_optimize` on the meta backend, which had none.** So the CUDA
  fusions that need allocation dependencies (the MoE weighted sum) were refused by the overlap check. The meta backend
  now asks each device backend (reg proc `ggml_backend_graph_add_alloc_deps`) on its own graph.
- **ROCm 7.14 (R9V's), warm:** decode 1-2 ms per step slower, prefill +3%. The earlier "7.14 halves prefill" was a
  cold page cache. `build-v19-rocm714.sh` builds with it through a symlink to R9V's image.
- **The HIP runtime knobs** `ROC_ACTIVE_WAIT_TIMEOUT` and `HIP_FORCE_DEV_KERNARG`: no effect. Non-coherent cold experts
  (`GGML_CUDA_UVA_NONCOHERENT`): no effect.
- **The P2P AllReduce hybrid (`GGML_CUDA_AR_HYBRID`) lost to RCCL** once the split was even: 37.6 vs 31 us per call.
  Keep RCCL unless re-measured.

## 4. Design rules that held

- **Per-mat-vec activation quantization (q8_1) can be shared** between mat-vecs on the same activations
  (`ggml_backend_cuda_context::q8_1_reuse`). The rules that make it safe:
  - key on the root tensor, the byte offset and the row layout, following pure views only (VIEW/RESHAPE/PERMUTE/
    TRANSPOSE; an in-place op's result is a view that holds new data);
  - valid for one graph evaluation, reset at the start of every evaluation pass (a capture retry re-runs the loop);
  - main stream only;
  - buffers only grow, and a replaced buffer is kept for the CUDA graphs captured with it;
  - several entries, so the routed and the shared experts still share across the MoE down quantization in between.
- **Replace chains of generic ops that touch every KV cell every pass with one op each:** `ggml_qsa_pool`,
  `ggml_qsa_expand`. Each new op needs:
  - enum and name tables (and the `GGML_OP_COUNT` static asserts);
  - a CPU reference;
  - CUDA dispatch and `supports_op`;
  - a meta-backend split rule (`handle_generic(scalar_only)` for mirrored inputs);
  - `test-backend-ops` cases.
- **A new optional FA input** (src[8]) must be refused by every other backend's `supports_op`; the lse guard (src[5])
  is the template. The meta backend must assert it mirrored.
- **Kernels at ~1-2 us are cheaper to delete than to optimise:** a GPU step here costs ~3 us of gap per kernel
  (3,800-4,700 kernels per step). Count kernels per pass with `kernel_ngrams.py` and fuse the most frequent chains.
- **MoE expert reuse** (R9V `reuse3v2`):
  - one block per (token, slot) route; the first route of an expert computes all of its routes;
  - weights decoded in registers once per block, with no LDS staging. R9V measured that the LDS barriers cost more than
    they save on VRAM experts.
  - Cold experts sit in uncached host memory, so every duplicate read crosses PCIe again. Verify batches here repeat 32%
    of their routes (35% of the cold ones).

- **The hot-expert set is the biggest decode lever once the kernels are fixed** (step 21).
  - Cold experts are read from host memory over PCIe and dominate the MoE.
  - 44 GiB instead of 36.7 GiB of expert bytes in VRAM: 46.7 → 40.8 ms per step and prefill +12%, at 31.6 of 32.6
    GB per card.
  - `moe_hotset.py --budget-gib` sizes it. The budget is a rung decision (the user's), and the text stays
    byte-identical since placement changes no math.

## 5. Process traps (remote rig from Windows)

- **Don't pipe Python through a bash heredoc to edit C++** (`\n` became a real newline in a string literal, and quoting
  broke another script). Write the script with an editor tool and run it, or use exact-match edits.
- **Staging hunks on Windows:** pipe patches as bytes, since text mode adds CRLF and `git apply` rejects them:
  `scripts/fleet/git_stage_hunks.py`.
- **Sync the dev tree by checksum:** md5 of the tracked files locally, `md5sum -c` on the rig, then tar only the files
  that differ. Their fresh mtimes make make rebuild just those. `tar --force-local` on Windows, and
  `--no-same-owner` on the rig.
- **Never chain a benchmark after a build without checking `errors: 0`:** a failed build leaves the previous binaries,
  and the A/B then measures the wrong code.
- **Check `systemctl is-active` before assuming a run finished:** a poll timeout is not completion.
- Long jobs run under `systemd-run --unit=... --collect`. Kill by PID, never `pkill -f` inside plink.

## 6. Where the remaining gap is (step 10, per step, traced, vs R9V rank 0)

| | ours | R9V | lever |
|---|---:|---:|---|
| MoE (+ R9V's LRU cache) | 15.6 ms | 22.6 ms | expert reuse (in progress), hot-set size (VRAM is equal at 65k; 4.8 GB/GPU free) |
| dense `mul_mat_vec_q` | ~7-8 ms | 3.9 ms | fewer, wider mat-vecs (qkv concat), fused hyper-connection mat-vecs |
| idle (launch gaps + host syncs) | ~20 ms | 16.2 ms | kernel count (fusions), fewer host syncs per step (drafts) |
| draft passes | 4 x ~1.6 ms | fused, one CUDA graph | full-vocabulary head 0.42 ms per draft; needs a split-aware top-k |
| AllReduce | 3.3 ms | 1.7 ms | balanced now; the residue is jitter |
| output head | 1.66 ms | 0.6 ms | draft head once per step or coarse (needs a split-aware top-k) |

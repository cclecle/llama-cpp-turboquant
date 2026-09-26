# Playbook: HIP / RDNA4 kernel optimisation in ggml-cuda

What the v19 campaign taught about writing and tuning kernels for RDNA4 (Radeon AI PRO R9700, gfx1201) inside
ggml's CUDA/HIP backend. Every rule below comes from a measurement or a code reading in this campaign. Where something
is only a hypothesis, it says so. The model-level lessons (tensor split, drafts, methodology) are in
`PLAYBOOK-rdna4-tp-decode-optimisation.md`.

## 1. The machine, as the numbers showed it

- **Compute:** 64 CUs, wave32 (ggml `WARP_SIZE` 32, physical warp 32), LDS 64 KB per workgroup.
- **VRAM:** peak ~640 GB/s.
  - A healthy mat-vec reaches 450-620 GB/s: Q8_0 `wqkv` 2560 → 5120 at 480 GB/s, the Q6_K output head at ~625 GB/s.
  - Anything far below that is a launch-shape problem, not the memory.
- **Host memory over PCIe Gen5 x16:**
  - Mapped, coherent host memory (the cold experts of `<dev>_TIERED`) reads at ~29 GB/s per GPU.
  - It is not cached in the GPU L2, so every repeated read of the same bytes crosses PCIe again.
- **Dispatch cost:**
  - Inside a HIP graph, consecutive kernels leave ~2.7-3.3 us of idle between them; outside one (small draft graphs),
    ~6.6 us.
  - A 1 us elementwise kernel therefore really costs ~4 us.
  - At 3,800-4,700 kernels per decode step, the gaps (~20 ms traced) outweigh most kernels.

## 2. Check the achieved bandwidth first

For every kernel in the per-step top list, compute bytes moved / duration and compare with ~640 GB/s. In this
campaign, each kernel far below that had a launch-shape problem:

| kernel | symptom | cause | fix |
|---|---|---|---|
| `mul_mat_vec_q` Q8_0 10240 → 320, 5 columns | 14.7 us, ~240 GB/s | RDNA4 table: 1 warp per row for 2-8 columns, 320 warps for the whole GPU (~5 per CU) | 4 warps per row when rows < 2048 and K >= 128 blocks (`halve_iters` variant): **-2.6 ms/step** |
| `mul_mat_f` BF16 2560 → 128 / 512, 5 columns | 48 us | one block per 32 rows: 4 / 16 blocks | route to `mul_mat_vec_f` up to 5 columns on RDNA4 |
| `rope_multi` 128-wide rows x 8,192 | 81 us | one row per 256-thread block: 3/4 of threads exit at once, 8,192 blocks | pack rows per block (`blockDim.x` = rows, must divide the row count): 6 us |
| `k_get_rows` q8_0 → f32 of every key | 64 us | materialises 16.8 MB of f32 to average 4 rows | fused gather + mean + norm reading q8_0 once (`ggml_qsa_pool`): 12 us |
| `k_get_rows_float` of a [5, n_blocks] transpose | 60 us | rows of 5 floats, plus two transposes | direct `score[blk[j]] + mask[j]` kernel (`ggml_qsa_expand`): ~3 us |
| `flash_attn_ext_vec` 1 query, 12 Q heads per K/V head, 32k q8_0 | 210 us | one block per Q head: K/V read 12 times | mma kernel with the 12 heads in the columns at (1, 16): ~90 us |

Rules that follow:
- **Give each CU at least 8-16 waves.** For a mat-vec, count output rows x warps per row against 64 CUs. With few rows,
  split K across warps (or blocks); with short rows, pack several rows per block.
- **Never materialise a dequantised or expanded copy of an O(n_kv) tensor to feed a reduction.** Fuse the gather into
  the consumer.
- **A reduction over a tiny inner dimension wants a dedicated kernel,** not a transpose + get_rows + transpose.

## 3. Launch configuration mechanics in ggml-cuda

- **mmvq's launch shape comes from constexpr tables** (`calc_nwarps`, `calc_rows_per_block`) keyed by device table and
  column count. They are evaluated both on the host and in the kernel (`get_device_table_id()`), so host and device
  must agree.
  - Add shape-dependent variants through the existing template flags (`small_k`, `halve_iters`).
  - Instantiate a variant only where it changes the shape (the `c_promoted` / `c_tall_ok` constexpr checks), or every
    type compiles a second, identical kernel.
- **Kernel arguments through `ggml_cuda_kernel_launch` (cudaLaunchKernelEx) cannot use C++ default parameters.** A new
  trailing parameter must be passed explicitly at every launch site (`rms_norm_f32` has 6).
- **`__launch_bounds__` must match what is launched.** Derive it from the same constexpr tables as the launch.
- **`ggml_cuda_pdl_sync` / `ggml_cuda_pdl_lc` are no-ops on HIP.** Keep them for CUDA symmetry.
- **Batch invariance:** RDNA4 decode (1 column: 8 warps) and verify (2-8 columns: 1 warp) already sum mat-vec partials
  differently. A launch change for verify batches does not break a guarantee that existed; the RDNA3.5 table is the
  one place that keeps it.

## 4. Flash attention on RDNA4 (the mma/WMMA kernel)

- **Kernel names in rocprof traces carry the template arguments** (`flash_attn_ext_f16<256, 256, 8, 4, ...>` =
  DKQ, DV, ncols1, ncols2). Read them before guessing which path ran.
- **GQA packing:** ncols2 Q heads of one K/V head share the tile columns. A K/V tile loaded once serves them all; the
  vec kernel (1 query) instead reads K/V once per head.
- **The tile loaders can gather rows through an index list** (`use_sparse`): K, V and mask are read per listed row, and
  -1 entries become zero rows with a -inf mask.
  - Upstream fed this only from an NVIDIA mask compaction.
  - With explicit lists (`ggml_flash_attn_ext_add_kv_idx`) it runs on WMMA at ncols1 = 1, one query per tile, so each
    query keeps its own list.
- **q8_0 K/V are read natively** by the mma kernel when `Q->ne[1] <= 8` (`kv_native`), which avoids converting the
  whole cache to f16. Keep new paths inside that window or budget the conversion.
- **Stream-k:** with few output tiles (5 queries), the KV range is split across blocks, then a fixup kernel adds
  10-25 us per call. A dedicated split-K sparse kernel (R9V's `_qsa_sparse_paged_gqa_splitk` is 14 us) would beat
  75 us.
  - The RDNA config for 16 columns is 64 threads with 32-cell KV batches: likely latency-bound.
  - Tuning target; perf cases are registered in `test-backend-ops -m perf`.
- **Every new (DKQ, DV, ncols1, ncols2) needs a template instance.** Edit `generate_cu_files.py` and regenerate: the
  instance files are generated, and an `extern` declaration without an instance only fails at link time.

## 5. Memory, allocation and HIP graphs

- **Host-side decisions made while a HIP graph is captured are baked into it.**
  - A cache that skips a kernel on a hit must be invalidated at the start of every evaluation pass:
    `ggml_cuda_graph_evaluate_and_capture` can re-run its loop, and a replayed capture holds only its own kernels.
- **Pool memory:**
  - The VMM pool frees in LIFO order only; the legacy pool (`GGML_HIP_NO_VMM=ON`, this build) does not care.
  - A buffer held across ops breaks LIFO when another op already holds temporaries.
  - For data that must outlive one op, use a context-owned buffer that only grows. Keep replaced buffers, because
    captured graphs still point at them.
- **`cudaMalloc` during capture** is allowed in the relaxed capture mode ggml uses.
- **Multiple streams** (the concurrent-events path) can race on any context-owned scratch. Restrict such caches to the
  main stream.
- **Generic pointers can point into LDS,** and ggml's `vec_dot_*` functions work on an LDS copy of quant blocks.
  - Mind the alignment: q8_0 blocks are 34 bytes, 2-byte aligned (`get_int_b2`).
  - Copy with the widest load the source alignment allows.

## 6. Reuse and fusion patterns that worked

- **Quantise shared activations once:** see `ggml_backend_cuda_context::q8_1_reuse` for the safety rules (root tensor,
  pure views, per pass, main stream, grow-only buffers). 668 → 549 quantize kernels per step.
- **Elementwise fusions for chains of ~1 us kernels** (`scale -> unary -> scale`, `rms_norm -> scale`): each removes a
  launch plus a ~3 us gap.
  - Register the pattern in `ggml_cuda_can_fuse` (a whitelist) *and* handle it in `ggml_cuda_try_fuse`.
  - Then confirm with a trace that the standalone kernels disappeared.
- **Multi-column fused GLU mat-vec:** the kernel already looped over columns. Removing the launcher restriction, for
  one type to limit compile time, fused the shared expert of every verify batch.
- **MoE expert reuse** (R9V `reuse3v2`):
  - one block per (token, slot) route, and the first route of an expert computes all of its routes;
  - no LDS staging; each lane decodes a weight block into registers and dots it with every matching activation.
  - R9V found that LDS staging with two barriers per K step cost more than it saved on VRAM experts.
  - Whether the compiler folds the repeated `vec_dot` weight loads is a **hypothesis to check in the ISA** (R9V wrote
    explicit per-type decoders instead).
- **Top-k over a large vocabulary on the CPU:** one pass in 64-entry chunks, skipping a chunk whose max cannot enter the
  current top-k. Almost every chunk is skipped: ~0.1 ms for 248k entries, instead of sorting a candidate array.

## 7. Cross-GPU synchronisation

- **A direct-P2P AllReduce kernel** (flags in peer VRAM, `__threadfence_system`, spin with `s_sleep`) costs ~9 us when
  both GPUs arrive together.
- **What was measured as "AllReduce time" was mostly one GPU waiting for the other.** Fix the load balance before
  touching the collective. Once balanced, RCCL (31 us per call including residual jitter) beat the P2P kernel
  (37.6 us).
- **To find the cause,** pair the collective kernels of both GPUs by order and compare their start times and the GPU
  time of the preceding work (the analysis is in the other playbook).

## 8. Compile time and binary size

- **`mmvq.cu` is one big translation unit;** every added template variant lengthens the build.
  - Scope new variants to the types that need them (`type == GGML_TYPE_Q8_0` in `if constexpr`).
- **The flash-attention instances are split in `template-instances/`,** so a new case compiles in parallel.
- **A change to `ggml.h` (a new op) or to `common.cuh` rebuilds the whole HIP backend** (~10-25 min on the rig). Batch
  such changes.

## 9. Tuning method

1. Take a trace of the real workload (`decode-profile.sh`) and its per-step bill (`torchtrace_anatomy.py`).
2. For the top kernels, compute the achieved bandwidth (or FLOP rate) and read the template arguments.
3. For the long tail, count chains (`kernel_ngrams.py`) and fuse the most frequent.
4. Microbenchmark in `test-backend-ops -m perf`: register the real shapes in `make_test_cases_perf`, as for the QSA
   cases.
5. Correctness:
   - `test-backend-ops` against the CPU reference, including odd sizes, strided views, several sequences and unused
     entries;
   - then perplexity on the real path;
   - then the text.
6. Speed: A B B A on the server metric (ms per step), with an env toggle per change.
7. Confirm in a new trace that the change did what was intended. Twice here it had not: a fusion missing from the
   whitelist, and a build that had failed.

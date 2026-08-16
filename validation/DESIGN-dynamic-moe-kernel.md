# Dynamic MoE placement kernel - design and current state

> ## STATUS 2026-08-16: promotion and the facade were TRIED, MEASURED, and REMOVED
>
> Sections 2 to 8 of this document describe block promotion and the facade store as live,
> deployed features. **They are not.** Both were removed from the code and from every ini file
> on **2026-08-16**, after measurement showed they lose to the per-expert pool everywhere.
> What ships is the pool - which appears below only as the "facade-off fallback".
>
> **Read section 9 first.** It records what was tried, the numbers, and why each mechanism was
> cut. Everything before it is preserved as the design record of the attempt, not as a
> description of current behaviour. In particular the section 6 and section 8 tables compare
> facade configs against v6 static-ot, never against a correctly configured pool - section 9
> explains why that comparison was misleading.

Date: 2026-08-14. Branch `feat/moe-hybrid-kernel` (committed base `967a9dac1` + the unified
resident store). Target hardware: 2x AMD R9700 (32 GiB each, gfx1201) + 12-core Zen4, 110 GB RAM.
Primary model: Mistral-Small-4-119B (36 blocks, 128 experts/block, top-4, 89.6 GiB of expert
weight = 96.7% of the model).

## 1. The problem this solves

The expert weights do not fit in VRAM, so part of them must live in host RAM and someone must
compute them there. Stock llama.cpp forces a static, load-time choice per tensor (the `-ot` list:
these blocks on GPU, those on CPU) and computes each operation on exactly one processor. That
wastes whichever processor is idle, ignores the fact that some experts are used far more than
others, and has to be re-tuned by hand whenever the context size or model changes.

This kernel makes placement dynamic and demand-driven: experts sit where the measured routing
says they earn their memory, both processors work on the same layer when that is profitable, and
the placement adapts at runtime instead of being baked into a config line.

## 2. The strategy in one view

Every expert is in exactly one of three classes, and the policy moves experts between classes as
demand is measured:

| class | weights live in | computed by | serves |
|---|---|---|---|
| 1 | VRAM | GPU | as much traffic as VRAM can hold |
| 2 | host RAM | GPU, reading over PCIe | a tuned share of what class 1 misses, decode only |
| 3 | host RAM | CPU | the remainder |

Two facts of the machine shape how this is implemented:

- **The scheduler places whole operations.** A layer's 128 experts are one tensor, and the graph
  runs that layer on one backend. So "this expert is in VRAM" can be expressed two ways: move the
  whole layer (then the GPU runs it natively, at full pipeline speed, zero coordination), or keep
  the layer on the CPU and have our kernel farm the resident experts out to the GPU per layer
  (flexible, but each layer pays a fixed coordination cost).
- **Prefill and decode are different workloads.** At prefill a layer carries ~512 tokens, so any
  per-layer cost is amortized and the GPU's batched matmul dominates. At decode a layer carries
  one token: the work is tiny and fixed costs dominate. Every policy in this kernel is therefore
  phase-dependent.

This yields the placement granularities in the kernel:

**The facade store (REMOVED 2026-08-16 - see section 9; it lost to the pool at every point
measured, and its recorded wins came from a mis-set flag).** With `moe-hybrid-facade on`, per-expert
facade slots sharded across both GPUs are the ONE resident store: decode reads them through
GPU-owned facade nodes (pointer tables, zero CPU involvement), prefill and verify batches read
the same shards from the CPU-owned hybrid path (per-device launches; down's partial sums join
over P2P). Class-1 residency, both phases, no duplication - see section 5, `moe-hybrid-facade`.

**Block promotion (REMOVED 2026-08-16 - see section 9; it swaps the wrong unit and each hole
converts 2.46 GiB of pool).** `--moe-hybrid-promote N` pre-allocates N pairs of
GPU "hole" tensors at load (the swap buffers that make moves possible). Once demand has
accumulated, the policy copies the chosen blocks' expert tensors into holes (~190 ms per tensor,
once) and flips the layer's weight pointers. From the next graph on, those blocks run natively on
the GPU - pipelined, no per-layer coordination, identical cost to a statically placed block, but
chosen at runtime and reversible. The holes are named to match the tensor-split patterns, so
under `--split-mode tensor` each promoted block is row-split across both GPUs like any
loader-placed expert tensor. The facade store beat it on this model; it remains for models where
whole-block residency wins.

**Per-expert pool slots (THIS IS WHAT SHIPS as of 2026-08-16 - the other two were removed).**
Without facades, CPU-owned layers get a
VRAM slot pool (sized by `--moe-hybrid-experts`, `auto` = free VRAM minus `--moe-hybrid-reserve`)
holding the hottest experts per tensor, and the layer's work is split three ways per the classes
above. Rows are disjoint in the output, so CPU and GPU never overlap and nothing is reduced.

## 3. What happens on each node, per phase

**Prefill (batched node, ~512 tokens, ~1390 expert-token pairs):**
1. The CPU groups the routing (which token goes to which expert) - this already existed.
2. The layer's activation block is uploaded to the GPU once (it is dense), and a small gather
   kernel builds the pair-ordered activation matrix on-device at VRAM speed. (The CPU used to do
   this gather at 5 GB/s - 4.5 ms per layer, more than the matmul itself.)
3. Resident experts run through the tiled quantized matmul (MMQ) with GPU-side bucketing - a real
   GEMM per expert, not a mat-vec.
4. Non-resident experts' rows are computed by the CPU concurrently. Class 2 is off at prefill
   (`--moe-hybrid-stream-prefill 0`): measured, PCIe-fed matmuls cost the GPU more time than they
   save the CPU.
5. The CPU spin-waits on a completion event (no syscall) and scatters results.

**Decode (1 token, ~3-12 pairs):**
1. Same routing grouping; the tiny activation rows are copied directly (1 us).
2. Resident experts run through the mat-vec, which is measured at the VRAM bandwidth roofline
   (60 us for ~30 MB of expert reads) - the GPU side is efficient.
3. Each non-resident expert is row-split: the CPU takes the bottom 70% of rows, the GPU reads the
   top 30% (`--moe-hybrid-stream 0.30`) straight from pinned host RAM over PCIe. 0.30 is a sharp
   measured peak - the limit is join latency, not PCIe bandwidth.
4. Promoted blocks skip all of this: they are ordinary GPU layers.

**The decode economics, plainly:** coordinating one CPU-owned layer with the GPU costs ~0.3 ms
(driver calls, launch, wait, copy back) regardless of how little work it carries. 36 layers means
~21 ms/token of pure coordination - a third of a token's budget - which is why per-expert slots
alone cannot beat the static baseline at decode even at 80% VRAM coverage, and why promotion
(zero coordination) exists.

## 4. The placement policy

- **Scoring:** every layer execution adds its routing counts to a per-expert window; every
  migration pass folds the window into a decayed rating (`rating = rating*0.98 + window`).
  Half-life ~8.8k tokens. A 3000-token routing trace showed demand is near-stationary - the
  long-run top-72 misses 9% of visits while a fast-reacting policy missed 17%, and even a perfect
  predictor (Belady bound) only reaches 5.5% while paying more in swap traffic than the misses
  cost. So the policy deliberately has long memory, and prediction was measured and rejected.
- **Migration:** the resident set is rewritten every `--moe-hybrid-migrate 256` tokens on a
  dedicated stream; slots go EMPTY -> LOADING -> READY and only READY slots serve, so decode never
  stalls on a refill. An incumbent expert defends its slot with a `--moe-hybrid-hysteresis 2.5`
  score multiplier: measured, this halves evictions and refill traffic at equal throughput.
- **Promotion:** currently single-shot - after ~96 graph builds the top-demand blocks (block-level
  demand is uniform on Mistral by construction, so effectively an arbitrary pick) are moved into
  the holes and stay there. Rotation/demotion is a latch away but has no demand signal to act on
  for this model.
- **Device balance:** slot tensors are assigned to whichever GPU holds the fewest bytes (naive
  round-robin gave one GPU every large tensor).

## 5. Configuration keys

Every key below is a CLI flag added on this branch, and through the preset system each one is also
a per-model ini key (same name, no leading dashes) and an environment variable (`LLAMA_ARG_` +
uppercased name). Recommended values are the measured ones for Mistral-119B on this box.

### Foundation

**`cpu-moe = true`** (upstream flag, no value). Places every expert tensor in host RAM at load.
This is what removes the static `-ot` list: with all experts starting in RAM, placement belongs
entirely to the policy below. Without it the kernel only manages whatever the `-ot` list happens
to leave on the CPU, which reintroduces a hand-tuned static split.

**`moe-hybrid = on|off`** (default off). Master switch for the whole kernel. Off means stock
llama.cpp behavior: CPU-resident experts are computed by the CPU alone. The backend registers its
hooks unconditionally at startup (the registry exists before arguments are parsed), and this
switch gates them lazily on the first node.

### Sizing the VRAM working set

**`moe-hybrid-experts = auto|N`** (default auto). How many pool slots each expert tensor gets -
the unit of class-1 placement **for facade-off configurations only**: with the facade on, every
expert tensor lives in the facade store, the pool is empty (census: `pool: 0.00 GiB, 0
admitted`), and `moe-hybrid-facade-slots` is the sizing knob instead. For facade-off runs: a slot
is a container for one expert; the policy binds and rebinds experts to slots as demand moves.
`auto` fills all free VRAM minus the reserve, measured at pool build time, which happens ~8
graphs after startup so the KV cache and compute buffers are already allocated and are never
fought over. An explicit `N` applies to every tensor uniformly; if it does not fit, the kernel
logs it and shrinks proportionally rather than failing.

**`moe-hybrid-reserve = MiB`** (default 512, recommended **1024**). Per-device VRAM left free when
`auto` sizes the slots. This is the static headroom for everything that loads *after* the pool:
the draft model and context growth. It is the answer to "leave space for draft and context" - a
per-model constant, not a runtime trim. Measured at 128k+EAGLE: 4096 left 8 GiB idle and cost 15
points of decode (coverage 64%, decode -20% vs baseline); 1024 gave 63 slots, 69% coverage,
decode -5%, and no OOM with the draft loaded. Raise it only if a model's draft or context
genuinely needs more.

**`moe-hybrid-promote = N`** (default 0, best measured **12**). True block residency: N pairs of
GPU hole tensors are pre-allocated at load (each pair = one block = 2.46 GiB on Mistral, row-split
across both GPUs under tensor split), and at runtime the policy moves N whole blocks into them -
data copied once (~190 ms/tensor), layer pointers flipped, the scheduler then runs those blocks
natively with zero per-layer coordination. This is the decode lever: promoted blocks cost exactly
what statically placed blocks cost. The holes claim their VRAM at load, so `auto` slots
automatically size around them - the two share the budget without manual arithmetic. Values >= 14
currently trigger the open decode collapse (section 7) - stay at 12 until that is fixed.

### The class-2 share (GPU reads host RAM over PCIe)

**`moe-hybrid-dma = on|off`** (default off, recommended **on**). Pins the host expert weights
(hipHostRegister, all 89.6 GiB on Mistral) so GPU kernels can read them in place. Required by the
stream shares below. Cost: pinned pages cannot be reclaimed, and the CPU reads them ~11% slower
(measured 31.8 -> 28.3 GB/s); class-2 traffic contends for the same DRAM controller on top. Both
costs are repaid by the GPU work class 2 contributes - net wash at 128k, clear win at 16k.

**`moe-hybrid-stream = FRAC`** (default 0, recommended **0.30**). At decode, the top FRAC of every
non-resident expert's output rows are computed by the GPU reading pinned host memory, while the
CPU computes the bottom 1-FRAC concurrently - both sides of the same expert, disjoint rows, no
reduction. 0.30 is a sharp measured peak (decode 12.7 / 15.6 / 11.8 / 8.6 at 0 / 0.30 / 0.45 /
0.60): past it the CPU ends up waiting on the GPU's PCIe reads, so the limit is join latency, not
link bandwidth. Do not raise it on bandwidth arithmetic - that reasoning was tried and measured
wrong.

**`moe-hybrid-stream-prefill = FRAC`** (default and recommended **0**). The same share for
batched (prefill) nodes. Measured repeatedly, including after the prefill-path rework: any
positive value loses (batched kernel time grows 3.8 -> 7.6 ms as the share rises to 0.5), because
at prefill the GPU is already saturated with the resident experts' matmuls and PCIe-fed matmuls
cost it more time than they save the CPU. Exists so the two phases can be steered independently.

**`moe-hybrid-stream-batch = N`** (default **32**). The boundary between "decode" and "batched"
for everything phase-dependent: nodes with at most N tokens use the decode policy (mat-vec
kernel, `stream` share) and larger nodes use the batched policy (MMQ kernel, `stream-prefill`
share). 32 deliberately puts speculative verify batches (~13 tokens) on the decode side - they
are still bandwidth-bound single-pass reads. There has been no reason to move it.

**`moe-hybrid-decode = on|off`** (default on; off is built but not yet measured). Whether the
per-expert slots serve decode-sized nodes at all. `off` makes non-promoted blocks pure CPU at
decode - no dispatch, no launch, no join, stock-kernel cost - while the slots still serve
prefill. Designed to pair with high `promote` counts once the high-N bug is fixed: promoted
blocks carry decode, slots carry prefill, and the ~0.3 ms/layer coordination cost disappears
entirely. Scoring still runs when off, so placement stays informed.

### The facade (REMOVED 2026-08-16 - keys deleted, see section 9)

**`moe-hybrid-facade = on`** + **`moe-hybrid-facade-slots = N`** (default off). The facade slots
are THE resident-expert store, for both phases. Decode-sized nodes (<= 4 tokens) are served by
GPU-owned facade twins of the expert tensors: the token never crosses to the CPU, deleting all
~74 per-token scheduler boundaries that made decode overhead-bound. A per-device pointer table
serves each expert from wherever it lives - the N hottest per tensor from the facade slots in
VRAM (filled once demand settles, sharded across both GPUs by the tensor split), the rest read in
place from mapped host RAM over both PCIe links. Prefill and verify batches stay CPU-owned but
read the SAME shards for class 1: gate_up (segmented row split) launches once per device with
disjoint dst rows; down (contraction split) computes per-device partials over its column slice,
device B's partial hops to the tensor's home device over P2P and a small add joins them before
the single D2H. The pool holds nothing for facade'd tensors - no duplication, and every read on
both phases is local to its shard at VRAM speed. Shard geometry is learned at runtime from
load-time row markers, never assumed. Requires `moe-hybrid-dma = on`; the down path additionally
needs direct peer access (probed at init; without it down falls back to the pool).

Because the store is shared, N is a pure SIZE knob, not a prefill/decode trade: both axes rise
monotonically with it all the way to the VRAM ceiling (16k no-spec, 12 reps: N=28 = 274/18.7,
48 = 304/23.0, 64 = 345/26.1, 72 = 371/29.4, 80 = 388/30.7). Set it as high as the target
context + draft still allow the load to fit.

### The placement policy

**`moe-hybrid-migrate = N`** (default 512, recommended **256**). Tokens between placement passes.
Demand is scored every node, but the resident set is only rewritten every N tokens, on a dedicated
stream, with slots gated EMPTY -> LOADING -> READY so decode never stalls on a refill. 256
measured better than 1024 (which converges to only 60% coverage inside a session) with no churn
penalty. Migration traffic is not the constraint; convergence speed after a load or topic change
is.

**`moe-hybrid-hysteresis = F`** (default 1.5, recommended **2.5**). A resident expert defends its
slot with an F-times score multiplier: a challenger must be clearly hotter, not marginally, before
the swap is paid for. Measured: evictions 1264 / 678 / 332 at 1.0 / 1.5 / 2.5 with equal or
better throughput - most churn is near-ties trading places, and this suppresses exactly that.

**Score decay** (env only, `GGML_MOE_HYBRID_DECAY`, default 0.98). Weight kept from previous
windows when folding demand. Deliberately long memory (half-life ~8.8k tokens): a routing trace
proved the demand near-stationary, and short memory (0.875 and below) measurably evicts long-run
hot experts on window noise. Not exposed as a flag because the axis measured flat between 0.75
and 0.995 on the benchmark - the trace analysis, not throughput, chose 0.98.

### Diagnostics (env only, not for production)

- `GGML_MOE_HYBRID_STATS=1` - the census: per-class visit shares, per-stage GPU timings
  (gather/H2D/kernel/D2H/join, split decode vs batched), CPU bytes and effective GB/s (counting
  only rows the CPU actually computed), migration/eviction/refill counters, per-tensor table.
- `GGML_MOE_HYBRID_TRACE=path` - dumps the routing trace (one line per single-token node) at
  shutdown, for offline policy analysis; this is what refuted the predictor.
- `GGML_MOE_HYBRID_VRAM` / `--moe-hybrid-vram` - legacy byte-budget sizing, superseded by
  `moe-hybrid-experts`; kept for compatibility, do not use in new configs.

### Recommended Mistral ini block - SUPERSEDED 2026-08-16, do not copy

The block below is the 2026-08-14 facade config, kept for the record. The facade keys no longer
exist; a server given them refuses to boot, because the preset parser validates every section.
**What actually ships is in section 9.**

```ini
cpu-moe                   = true
moe-hybrid                = on
moe-hybrid-experts        = auto
moe-hybrid-reserve        = 1024
moe-hybrid-facade         = on        ; REMOVED - key no longer exists
moe-hybrid-facade-slots   = auto      ; REMOVED - key no longer exists
moe-hybrid-dma            = on
moe-hybrid-stream         = 0
moe-hybrid-stream-prefill = 0
moe-hybrid-migrate        = 256
moe-hybrid-hysteresis     = 2.5
poll                      = 100
spec-draft-n-max          = 6
spec-draft-p-min          = 0.9
```

(Final-build retune, 12 reps at 128k + EAGLE: f64/stream0/n6 = 334.5/28.76 acc .465 beats the
n12 draft control (333.4/27.63 acc .482), the f56 slot control (310.3/26.73 acc .514) and, at
matched acceptance, the stream-0.30 control (338.6/29.60 acc .509 - the higher draw explains the
raw gap). `spec-draft-n-max 6` also bounds verify batches to 28 pairs.

`moe-hybrid-facade-slots = auto` restores the last piece of context-dynamism the pool used to
provide: at model load it takes measured free VRAM and subtracts the dense weights this load
will place, the KV cache (context length and cache types handed down from the common layer),
the draft model's file size when one is configured, a compute margin
(GGML_MOE_HYBRID_AUTO_COMPUTE_MB, default 1536), a 22 KB/token context-linear term for the
draft's own KV and depth-scaling scratch, and the reserve - then divides by the per-slot cost.
Validated landings on this box: 64 at 128k + EAGLE (the exact measured max-fit; 66 aborts the
load), 79 at 16k without a draft (empirical max 80-81). The estimate assumes tensor split;
under layer split prefer an explicit count. Change the context or the draft and the store
re-sizes itself - no ini edit.

`moe-hybrid-promote` is superseded by the facade and left out; the machinery remains for models
where whole-block residency wins. `moe-hybrid-experts` only matters with the facade off.)

## 6. Current performance (Mistral-119B, vs the production baseline: static -ot + tensor split)

All medians of 12 reps unless noted. Baseline = 18 blocks static VRAM / 18 blocks static CPU.
"Unified" = the single resident store (facade shards serve both phases, pool empty).

| regime | baseline | this kernel (final build) | verdict |
|---|---|---|---|
| **128k + EAGLE (production), ini config** | 277 / 19.9 (acc .606) | **334.5 / 28.8 (acc .465)** | prefill **+20%**, decode **+45% raw with the acceptance draw against it** |
| 16k, no spec, 80 slots (max fit) | 274 / 14.0 | **371.3 / 32.5** | prefill **+36%**, decode **+131%** |
| correctness | - | PPL 2.3965 vs 2.407 facade-off (150ch, inside error bars); temp-0 700-token decode byte-identical across restarts, crossing the fill and two re-fill passes; 6-topic basket crash-free | parity |

Intermediate sweep points from the pre-re-fill build (the shape of the curve, all 12-rep medians):
16k slots 28/48/64/72 = 274/18.7, 304/23.0, 345/26.1, 371/29.4 - both axes monotone in the one
knob; prod slots 48/56 = 298/22.0, 321/24.7.

The old prefill/decode trade (more facade = decode, more pool = prefill; best-of was 337/15.4 vs
271/21.4 on opposite ends) is GONE: one store serves both phases, so the slot count just sets how
much demand is VRAM-resident. The facade decode is med == max stable - the CPU is out of the
loop entirely, so run-to-run decode jitter disappears with it.

Structural wins beyond throughput: no hand-tuned `-ot` line per model/context; at 128k the slots
shrink automatically to make room for the KV cache and draft (the static split OOMs or needs
re-tuning); placement follows measured demand.

The former honest deficit - decode never beating the baseline - is resolved by the unified
store: production decode 24.7 vs 19.9 baseline with the acceptance draw against it, and the
promotion machinery is no longer needed for the decode win.

## 6b. Two crashes found by the mixed-prompt basket, both root-caused and FIXED

The single-prompt bench harness was structurally blind to both; a 6-class prompt basket (prose,
code gen, code refactor, poem, table, reasoning) exposed them within minutes. Keep using the
basket for any future validation of speculative or placement changes.

1. **MMQ buffer under-allocation (illegal memory access).** mmq.cu sized the tail padding of the
   quantized-activation buffer as `J_max(ne11)`; our synthetic one-expert-per-row launches (and
   any broadcast caller) pass ne11 = 1, which floors to ZERO padding, so the tile kernel's final
   partial column tile reads past the buffer. Whether the overread faults depends on what the
   pool happened to map after the buffer - hours of benches were silently overreading. Fixed by
   padding with the flattened column count the kernel actually tiles over
   (`J_max(max(ne_get_rows, 8))`). Upstream-relevant.
2. **HIP-graph capture collision ("operation not permitted when stream is capturing").** The
   build has GGML_HIP_GRAPHS=ON; when decode topology stays stable (for example p-min 0.95
   truncating drafts to ~1 token), upstream begins capturing the decode graph, and facade_run's
   lazy host-side work (geometry learn, pointer-table upload with a stream sync) is illegal
   inside a capture. Fixed by excluding graphs that contain facade nodes from capture
   eligibility (op_params magic check in ggml_cuda_graph_check_compability) - every measured
   number was effectively capture-free anyway.

## 6b-bis. Output corruption post-mortem: the facade gather dropped K-quant row tails (FIXED)

The worst bug of the campaign shipped to production and passed every gate. `moe_facade_gather`
copied expert rows into the decode staging buffer in 16-byte `int4` chunks, stopping when a full
chunk no longer fit. A q6_K row is a multiple of 210 bytes, which is a multiple of 16 only when
the column count is a multiple of 2048. Both production models fail that on exactly the down
projection: Qwen's down shards are 420-byte rows (last 4 bytes never copied), Mistral's are
840-byte rows (last 8 bytes never copied, 17 of 18 layers). The missing bytes are the final
super-block's scales and `d` - the multiplier for the last 256 weights of the row - so the row
computes with whatever scale the *previous* occupant of that staging slot left behind. The same
misalignment also made most of the `int4` loads themselves misaligned.

Why every gate was blind, recorded so the next campaign doesn't repeat it:

- **Perplexity only exercises prefill**, which reads the store in place through stock kernels
  (those assume only 2-byte alignment) and never touches the gather.
- **Temp-0 determinism proves determinism, not quality** - the stale-tail corruption is
  deterministic per routing history, so it passed restart-identity checks.
- **Short no-spec generations often look clean**: with stable top-4 routing the staging slot's
  previous occupant is frequently the *same* expert, so the stale bytes are accidentally
  correct. Speculation verify batches reshuffle the unique-expert list every step, making the
  tails nearly always wrong - hence token stutter under EAGLE/MTP ("can can can"), and constant
  '////' on Qwen, whose top-8-of-256 routing reshuffles even at one token.
- **Degenerate text self-inflates speculative acceptance** (Qwen showed 0.89 vs the historical
  ~0.58), so decode t/s went *up* while output quality collapsed. Every spec-on decode tuning
  measured on the broken build was invalidated.

The fix: take the vector path only when the row size and both endpoints are 16-byte clean,
otherwise copy 2-byte units (every K-quant block type is 2-byte aligned). The lasting rules:
**a validation run must read the generated text** (a content gate, not just metrics), an
acceptance rate far from its historical value is a red flag not a win, and no hand-rolled device
copy may assume K-quant rows are 16-byte aligned (q6_K = 210 B, q3_K = 110 B, q2_K = 84 B
blocks misalign; q4_K/q5_K/q8_0 rows happen to be 16-multiples).

Two more defects hid underneath, found once the tails were fixed (2026-08-15):

- **CUDA op fusion bypassed the facade hook.** The fused `{MUL_MAT_ID, MUL_MAT_ID, GLU}`
  pattern in `ggml_cuda_try_fuse` sent Qwen's gate/up facade nodes straight to
  `ggml_cuda_mul_mat_vec_q`, which read the twins' RAW allocation (a pointer-table store,
  ne02 = slot count) with router expert ids up to n_expert - out of bounds, dead routed-FFN,
  and the model *coasts on the shared expert* for dozens of plausible tokens before visibly
  collapsing. Fusion eligibility legally varies between structurally-identical graphs (node
  order, use counts, allocator placement), which made it look nondeterministic. Fixed by an
  early bail in `ggml_cuda_try_fuse` plus a magic check in `ggml_cuda_should_fuse_mul_mat`
  (mirrors the CUDA-graph capture veto). Mistral's fused-gate_up layout matches no fusion
  pattern and was never affected. Lesson: **every dispatch shortcut in the backend (fusion,
  capture, future fast paths) must be audited against tensors whose raw allocation is not
  their semantic content** - grep for consumers of MUL_MAT_ID whenever one is added.
- **Multi-token facade graphs stutter (root cause open).** With both fixes in, speculation-off
  output is pristine, but 2-4-token verify batches served by facade graph nodes produce token
  stutter ("scatteringcattering"). Mitigated by restricting both builder picks to
  `n_tokens == 1` (qwen35moe, deepseek2); verify batches ride the hybrid dispatch instead.
  The open suspect is facade_run's synthetic descriptor math for n_tokens > 1 against the
  stock ids-path contract. Revisit before re-widening the pick.

## 6c. The facade re-fill: dynamic placement on the unified store

The fill is no longer one-shot. On the migration cadence (`moe-hybrid-migrate` ubatches, default
256), llama re-asks the kernel for each tensor's hottest experts and swaps facade slots whose
occupant has been clearly overtaken. `top_experts` applies the hysteresis bonus to incumbents,
so a stationary workload converges to zero swaps, while a genuine demand shift moves the
residents. Each swap is ~20 MB of host-to-GPU copy and goes: unmap the old expert (it falls back
to its host weights - always correct), overwrite the slot bytes, map the newcomer.

Decode demand is now visible to the scorer: the facade prep kernel counts expert visits in a
per-tensor device array (device 0's shard only; both devices see the same ids), and the ranking
folds those counts into the window before it sorts. Without this, the facade decode path -
which owns all decode traffic - was invisible to placement, and the re-fill could only see
prefills.

Both regimes are measured on this build. Production damping (decay 0.98, hysteresis 2.5): zero
swaps across a 6-topic prompt basket - short topics are window noise by design, per the trace
analysis that chose the decay. Reactive settings (decay 0, hysteresis 1.0, migrate 32): 48-62
swaps per pass, ~1 GB of experts re-homed every 32 tokens, decode still 26.7 t/s while it
happens, byte-deterministic across restarts. The knobs really steer it; the invariant - experts
sit where measured demand says they earn their memory, at all times - holds again.

## 6d. Qwen3.5-122B: the same kernel, the split expert layout (2026-08-14 evening)

The kernel needed NO changes for Qwen beyond one constant: the facade staging bound rose from 16
to 32 unique experts (4 tokens x top-8; the builder pick caps by the PRODUCT of tokens and
routed experts, so a flat token cap can never overflow it again). Everything else was llama-side
plumbing: facade twins for the separate gate/up tensors, layer detection accepting either
layout, the fill/refresh iterating whatever exists, and the graph pick in qwen35moe (guarded to
quants without scale sidecars). Three real bugs were fixed on the way: the promotion-holes log
dereferenced the fused tensor unconditionally (segfaulted every split-layout load), the loader's
tensor budget was a flat +128 (Qwen needs 144 facades; now n_layer x 4 + 128), and facade-slots
auto learned to charge recurrent-state memory (delta-net r/s buffers, ~10 GiB on Qwen at 16k;
the term reads n_embd_r/s which are zero for pure-attention models, so Mistral is unaffected).

Validated (16k + MTP/ngram spec, same harness as every reference): PPL 1.8131 (facade, auto=91
of 256 slots) vs 1.8137 (facade off) - identical inside error bars; census "144 tensors served
from the facade store"; Mistral regression byte-identical on the same build. Performance, 4
reps: static baseline 257.1 prefill / 21.16 decode; unified store 250.5 / 131.8 median decode
(reps 62-201 - MTP/ngram acceptance swings widely; treat the decode multiple as directional
until a 12-rep characterization, but the direction is ~6x). The old "Qwen regresses under the
hybrid" verdict is obsolete: that was the CPU-path overhead, and the facade deleted it.

### Qwen3.5-122B deployment ini - SUPERSEDED 2026-08-16, do not copy

Kept for the record; the two facade keys were removed. **What actually ships is in section 9.**

```ini
cpu-moe                   = true
moe-hybrid                = on
moe-hybrid-experts        = auto
moe-hybrid-reserve        = 1024
moe-hybrid-facade         = on        ; REMOVED - key no longer exists
moe-hybrid-facade-slots   = auto      ; REMOVED - key no longer exists
moe-hybrid-dma            = on
moe-hybrid-stream         = 0.30      ; NOT 0: at 36% residency the PCIe help pays (+20% decode)
moe-hybrid-stream-prefill = 0
moe-hybrid-migrate        = 256
moe-hybrid-hysteresis     = 2.5
poll                      = 100
spec-draft-n-max          = 12        ; the old fleet default was right for Qwen
spec-draft-p-min          = 0.8       ; p axis measured flat 0.0-0.9; keep the default
```

Tuned numbers (12 reps, vs the same-harness static baseline 257.1 / 21.16 @ the same spec):
**285.9 prefill / 173.4 decode at acceptance 0.892 - prefill +11%, decode 8.2x.** The grid:
draft p_min is flat on Qwen (unlike Mistral's EAGLE, which collapses below 0.9); n12 beats n6 on
decode; and `stream 0.30` adds +20% decode over stream 0 at identical acceptance - the opposite
of Mistral's answer, because Qwen's 256 experts at 91 resident leave the GPU real PCIe work at
verify batches. Basket: all 6 classes clean both rounds, per-class decode 46-156 t/s.

## 7. Next steps, in order - SUPERSEDED 2026-08-16 by section 9

(Items 2 and 3 below concern promotion and facade machinery that no longer exists. Kept to show
what was planned before the measurement campaign closed those paths.)

1. **Qwen (old note, superseded by the tune above):** do not deploy there yet - its 3-tensors-per-block layout doubles per-layer
   coordination (147 nodes/token) and regresses; the fix is fusing gate/up launches, a separate
   work item. The unified store transfers as-is once that lands (nothing in it is
   Mistral-specific: geometry is learned, not assumed).
2. **Promotion leftovers** (machinery kept for models where whole-block residency wins, but
   superseded here): the high-N collapse at >= 14 blocks, and the N-2 silent skip in
   `moe_promote_step`. Only worth chasing on a model that actually wants promotion.
3. Cleanups: rewrite or remove the broken `--moe-hybrid-stream auto` controller; rotation latch
   if a block-skewed model ever needs it; a periodic facade re-fill if a workload with drifting
   expert demand ever shows up (the fill is currently one-shot at the 96-ubatch latch).

## 8. 2026-08-15 bottom-up re-verification (trust-recovery campaign) - the numbers that stand

Every layer re-verified on clean single-tenant GPUs after the three corruption fixes
(f9725cede gather tails, 778b6bc28 fusion bypass, f18899e04 ids-view stride). Full gate
artifacts on the box under /tmp/verify/. Superseded claims: the "decode 8.2x / 173.4 t/s @
acc 0.892" Qwen figure above was a corrupt-acceptance artifact and is VOID; the tables below
are 12-rep medians on the fixed build, measured against v6 baselines re-measured the same day
on the same harness (8k unique prefill, 200 gen, temp 0, warm store, prod 131k configs).

Kernel gates: store oracle byte-exact (288+144 tensor-device checks, 0 mismatches); 0
garbage-hunter hits under full spec; collapse prompt fluent spec-on and spec-off; restart
determinism byte-identical across boots (both models); KV save/restore/save file-identical;
150-chunk PPL 2.3968 +/- 0.025 (2.40 family, store prefill path exercised); 6-class basket
content read by hand, clean on both models in both spec states.

Estimator fix (this campaign): facade-slots=auto overcharged KV context-linearly on
non-MLA models - delta-net layers falsely charged (~75% of the term) and head-split KV
charged as if mirrored. Fixed in load_tensors (is_recr skip + /n_gpu unless is_mla);
Mistral arithmetic untouched (64 @131k = measured max, before and after). Qwen auto:
58 -> 125 @131k, validated under full load (no OOM, fill + 8k prefill + spec). Note: the
auto compute margin assumes ub <= 512; ub=1024 OOMs the EAGLE draft buffer regardless.

Gate table (pp/tg medians; v6 = /opt/backup_10082026/modelsBOTH.ini on v6 binaries):

| model | config | pp | tg | notes |
|---|---|---|---|---|
| Qwen3.5-122B | v6 static-ot TP (today) | 255 | 31.9 | acc .55-.71, 3 reps |
| Qwen3.5-122B | v7 pool (facade off) | ~230 | ~20 | 3 reps, L2 arm |
| Qwen3.5-122B | v7 facade 96 (old prod) | ~320 | ~21 | 3 reps, L2 arm |
| Qwen3.5-122B | **v7 facade auto=125 (new prod)** | **360** | **27.0** | 12 reps, max 53.4 |
| Mistral-119B | v6 static-ot layer (today) | 447 | 20.8 | acc .48-1.0, 3 reps |
| Mistral-119B | v7 pool (facade off) | ~180 | ~14 | 3 reps, L2 arm |
| Mistral-119B | v7 facade 48/n6 (old prod) | ~340 | ~19 | 3 reps, L2 arm |
| Mistral-119B | **v7 facade auto=64/n12 (new prod)** | **391** | **22.3** | 12 reps |

Honest verdict vs the plan's beat-v6-on-both gate: NOT a clean sweep. Qwen v7 wins prefill
+41% and loses decode -15% (median; -18% acceptance-matched) - the gap is the per-layer
facade cost vs v6's 28 fully-native static layers, the next kernel lever. Mistral v7 wins
decode +7% and loses prefill -12% (v6 ran a larger default ubatch; v7 must pin ub=512 for
the EAGLE draft buffer). Both new-prod configs strictly beat the OLD v7 prod on both axes,
so they ship; the v6 gaps are recorded here, not papered over.

Spec mechanism matrix (Qwen, same config/prompts, 3 reps each): spec-off 18.3 (+-0.1);
MTP-only 15.2 (net NEGATIVE alone - each draft costs a model pass); ngram-only 18.7;
full stack ~21-27 - the mechanisms are synergistic, keep the stack. The recurring 42-53 t/s
outliers are acceptance windfalls (acc .73-.80 when the ngram map warms on repeated topics)
- the "50 t/s mystery" of 2026-08-15 morning, reproduced and closed.

Wattage signature (1 Hz rocm-smi, phase-aligned): cold store-fill = ~55-65 W (the "50 W"
observation - one-time per load, ~110-130 pp first request); warm prefill ~90-115 W at
54-65% kernel duty (the rest is per-node activation h2d + output d2h - the top remaining
prefill lever, est. pp 360 -> ~500 if overlapped); facade decode 110-165 W. v6 averages
LOWER (64-66 W) while decoding faster: its speed is less work per token (native static
layers), not higher utilization - wattage is not a health metric here beyond the cold-fill
pathology.

Open items carried: test-backend-ops strided-ids MUL_MAT_ID coverage gap; verify-batch
dispatch cost (hybrid path serves 5-32-token verify: kernel 287 us/node, wait 9.5%);
prefill h2d/d2h overlap; the Qwen decode gap vs v6 (block-native path).

## 9. 2026-08-16: what was tried, what we found, and why promotion and the facade were removed

**Timestamp: 2026-08-16.** Removed on branch `rebase/upstream-20260810`. The code deletion is part
of the squashed commit `moe : hybrid CPU and GPU expert placement for MoE models`; the three CLI
keys `--moe-hybrid-facade`, `--moe-hybrid-facade-slots` and `--moe-hybrid-promote` are gone from
`common/arg.cpp`, and the corresponding lines were stripped from `/opt/modelsBOTH.ini`.

This section supersedes sections 2, 5, 6, 7 and 8 wherever they disagree with it.

### 9.1 Block promotion (`--moe-hybrid-promote N`) - REMOVED

**Tried:** pre-allocate N pairs of GPU "hole" tensors at load, copy the hottest blocks' expert
tensors into them once demand settles, then flip the layer weight pointers so those blocks run
natively on the GPU.

**Found:** it lost at every N. Two independent reasons, either one fatal.

- **It swaps the wrong unit.** Block-level demand is close to flat. The skew that makes dynamic
  placement pay at all is *within* a block - a few experts out of 128 carry the traffic - not
  between blocks. Promoting a whole block therefore buys native execution for a block that was
  not especially hot, and pays a whole block of VRAM for it.
- **Every hole converts pool.** A hole costs one block of VRAM up front, drawn from the same
  budget the per-expert pool uses. One hole = **2.46 GiB**, so a single hole removes far more
  per-expert residency than the promoted block gives back.

Secondary defects found and deliberately not fixed, because the mechanism was being cut: a high-N
collapse at N >= 14 blocks, and an N-2 silent skip in `moe_promote_step`. The N-2 waste traced to
the UD dynamic quant - `blk.4/28/32/35` carry `down_exps` as q8_0 while the prototype block is
q6_K, so those holes do not fit the tensors they were sized against.

### 9.2 The facade store (`--moe-hybrid-facade`, `--moe-hybrid-facade-slots`) - REMOVED

**Tried:** make per-expert facade slots the single resident store for both phases, and serve
decode-sized nodes through GPU-owned facade tensors so the token never crosses to the CPU.

**Found:** the plain per-expert pool beats it everywhere measured.

| model | facade on | pool (facade off) |
|---|---|---|
| Mistral-119B, 2 GPU, pp@8k | 402 | **505.7** |
| Qwen3.5-122B, 2 GPU, pp@8k | 356 | **410.7** |
| Qwen3.6-35B, mono-GPU, pp | 1505 | **1516** |

Two measurement faults had hidden this for weeks, and both are worth remembering:

- **The recorded facade win came from a mis-set flag.** `--moe-hybrid-facade off` does **not**
  free the store - the slot count is a *separate* env var. Every historical "facade off" arm
  therefore ran with the store still allocated and the pool at 0.00 GiB. Those arms were never
  pool measurements at all. Freed properly, the pool ties or wins. This is why the section 6 and
  section 8 tables look favourable: they compare the facade against v6 static-ot, never against a
  correctly configured pool.
- **The facade blocked the ubatch knob it needed.** Its fixed compute estimate prevents
  `ubatch-size 1536`, which is worth **+25.8% prefill** by itself. The facade was partly paying
  for itself with a setting it was also disabling.

### 9.3 Also measured dead - do not re-try

- **Any placement-ranking change**: LFRU, imatrix seeding, persisting counters across runs,
  noise-scaled hysteresis. A **35x** range in eviction count moved the hit rate by **0.0 points**.
  The ranking is not the lever; the capacity is.
- **More residency**: reaching 90% of visits needs roughly **10 GiB more VRAM than the box has**.

### 9.4 What ships instead

The per-expert pool, with the facade and promotion machinery deleted. Production config for both
big MoE models is the foundation keys (`moe-hybrid`, `-experts auto`, `-reserve`, `-dma`,
`-stream`, `-migrate 256`, `-hysteresis 2.5`) plus **`ubatch-size 1536` / `batch-size 4096`**.

Verified 2026-08-16 on the v8 build (`rebase/upstream-20260810`), real `/opt/modelsBOTH.ini`:

| model | measured pp@8k | target | delta |
|---|---|---|---|
| Mistral-Small-4-119B | 506.5 | 505.7 | +0.2% |
| Qwen3.5-122B | 410.0 (9 reps) | 410.7 | -0.2% |

Gates: `moe-nan` 0 and facade mentions 0 on both; class-1 residency 83% (Mistral) and 91.7%
(Qwen). Note that Qwen3.5-122B needs **>= 9 reps** to measure - a 5-rep sample read 376.3 (-8.4%)
purely from one slow rep.

### 9.5 The one idea still open

**True (block, expert) residency**: one copy of each expert, *moved* rather than duplicated, and
visible to the scheduler. Measured prize **2.82x** (native 3452.7 vs pool 1224.5 pp on
Qwen3.6-35B at identical residency).

The blocker is structural: ggml graphs are static while expert placement is data-dependent -
`ggml_top_k` runs *inside* the graph, so the partition cannot be known at graph build time. The
cheapest untested escape is a managed-memory buffer type (`hipMallocManaged`): one allocation,
driver-migrated pages, scheduler-native, no coordinator. Try that before attempting graph surgery.

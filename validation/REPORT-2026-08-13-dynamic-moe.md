# Dynamic MoE expert placement - design, measurements, deployment

Date: 2026-08-13. Branch `feat/moe-hybrid-kernel` (uncommitted worktree on top of `0567c33b8`).
Box tree `/opt/llamacpp/moe-hybrid`, build3. Production v6 untouched throughout.

## 1. What this is

A three-tier dynamic compute positioner for MoE expert tensors. Every expert tensor lives in host
RAM (`--cpu-moe`, no static `-ot` list); a score-driven placement policy decides, per expert, which
of three tiers serves it:

| tier | weights live in | computed by | when it pays |
|---|---|---|---|
| 1 | VRAM slot | GPU | always - VRAM reads are ~10x host reads |
| 2 | host RAM, pinned | GPU, over PCIe | decode only - adds PCIe bandwidth beside the CPU |
| 3 | host RAM | CPU | whatever the first two do not take |

- **Slots.** The unit of placement is a GPU expert slot per tensor (`--moe-hybrid-experts N`, or
  `auto` = fill free VRAM minus `--moe-hybrid-reserve`). The policy binds experts to slots and
  rebinds them as demand moves. There is no balance point: more slots = more throughput, so auto
  fills the cards (~28 GiB used per 32 GiB device on Mistral at 16k context).
- **Score.** Per-expert EWMA of routing hits, folded every migration pass
  (`rating = rating*decay + window`, decay 0.875 ~= half-life 5 passes). A resident expert defends
  its slot with a hysteresis multiplier so near-ties do not churn.
- **Evaluation period.** The resident set is rewritten every `--moe-hybrid-migrate` tokens
  (tuned: 256) on a dedicated stream; decode never stalls on a refill.
- **Two phases, two policies.** Prefill and decode differ in both *kernel* and *share*:
  batched nodes run the resident experts through the tiled MMQ matmul, decode nodes through the
  mat-vec; tier 2 takes `--moe-hybrid-stream` of each non-resident expert's rows at decode and
  `--moe-hybrid-stream-prefill` (default 0) at prefill, split at `--moe-hybrid-stream-batch`
  (default 32) tokens - so speculative verify batches count as decode.
- **Long context degrades gracefully.** The slots size themselves after the KV cache and draft
  model allocate, so at c=131072 the pool shrank 72 -> 55 slots per tensor on its own. No OOM, no
  hand tuning. The static `-ot` split needs manual re-tuning to survive the same change.

Under `--split-mode tensor` the CPU expert op is mirrored per device; both GPUs hold slots
(device assignment balances bytes, not tensor counts) and both serve tier 2 over their own PCIe
links.

## 2. Why the old measurements said "no gain"

Until today the VRAM pool was capped at 256 MiB per device = **5.7% coverage** on Mistral, a regime
where the mechanism cannot produce more than ~3% while this box swings +-20-30% run to run. Every
earlier positive or negative pool result was noise. Removing the cap and sizing in slots was the
unlock; everything below is measured with the proper architecture.

Second unlock: the GPU side originally fed *all* nodes through the decode mat-vec. At prefill that
ground 512-token batches one row at a time - fully dynamic prefill measured 167.8 vs 269.7 baseline
with the CPU waiting on the GPU 45.4% of node time. Routing batched nodes through MMQ turned the
same configuration from -38% to +18% prefill.

## 3. Results

Baseline everywhere = the production ini placement (static `-ot`) + tensor split, same context and
speculation settings as the arm it is compared against. All numbers are medians of 5 runs on the
natural-text corpus; this box's decode swings ~20-30% between server restarts, so single-run decode
deltas under ~20% are not conclusive. Prefill is stable to ~3%.

### Mistral-Small-4-119B (72 expert tensors / 36 blocks / 89.6 GiB experts, top-4 of 128)

**c=16384, speculation off** (interleaved x2):

| config | prefill med | decode med |
|---|---|---|
| baseline | 274.0 / 274.0 | 16.06 / 13.33 |
| fully dynamic, 72/128 slots | **319.8 / 328.0 (+18%)** | 15.48 / 12.89 (parity) |

Decode parity is expected at this context: VRAM coverage is ~equal (72.0% dynamic vs ~74%
effective for the static mix), and decode's residual cost is per-node dispatch, not placement.

**c=131072 + EAGLE speculation (production faith), tuned config:**

| config | prefill med | decode med | decode max | acceptance |
|---|---|---|---|---|
| baseline | 273.1 | 14.72 | 17.96 | 0.608 |
| tuned dynamic | 266.9 | **19.86 (+35%)** | **26.74 (+49%)** | 0.718 |

Acceptance-normalized decode is ~+14%; the rest of the +35% is the higher acceptance of that run
(acceptance varies between restarts). Tier 2 at 0.30 is what closed the gap - verify batches are
decode-sized and ride the shared CPU+PCIe path. No OOM; slots auto-shrank to 55/54 per tensor.

**Correctness:** PPL 2.7159 +/- 0.088 (dynamic) vs 2.7616 +/- 0.091 (baseline), 4 chunks - parity.
All tiers show non-zero work in the census.

### Qwen3.5-122B (147 expert tensors / 49 blocks / 80.9 GiB experts, top-8 of 256)

**c=16384 + MTP speculation:**

| config | prefill med | decode med | acceptance |
|---|---|---|---|
| baseline | 258.8 | 20.37 | 0.582 |
| tuned dynamic, 124/256 slots | 203.0 (-22%) | 13.83 (-32%) | 0.579 |

Acceptance equal, so the regression is real, and the census isolates it: coverage is *higher* than
baseline (75.8% of visits vs ~57% static), yet throughput is lower. Qwen's gate/up/down are three
separate tensors per block (147 hybrid nodes per token vs Mistral's 72), so per-node dispatch/join
overhead roughly doubles and eats the placement gain. **Do not deploy on Qwen** until the per-node
overhead is addressed (fuse per-block launches, batch pairs across tensors). Qwen3.6-35B shares the
3-tensor layout and the same expectation applies.

### Tuning grid (Mistral, one factor at a time)

| knob | tested | pick | evidence |
|---|---|---|---|
| tier-2 decode share | 0 / 0.15 / **0.30** / 0.45 / 0.60 | 0.30 | decode 12.7 / - / 15.6 / 11.8 / 8.6; sharp peak. The limit is join latency, not PCIe bandwidth - do not raise it |
| tier-2 prefill share | 0 / 0.15 / 0.30 | 0 | prefill 373 / 328 / 300 - the GPU has no spare capacity at prefill |
| evaluation period | **256** / 512 / 1024 | 256 | 1024 converges to only 60.5% coverage inside a session; 256 reaches 72.8% at equal churn |
| score decay | 0.75 / **0.875** / 0.9375 | 0.875 | flat axis, keep default |
| hysteresis | 1.0 / 1.5 / **2.5** | 2.5 | evictions 1264 / 678 / 332 at equal-or-better throughput; halves refill traffic |
| DMA registration | on/off | on | free: prefill 373.4 registered vs 354.7 not; the old ~11% tax does not reproduce |

## 4. Prepared ini changes (Mistral only)

In the Mistral section of `/opt/modelsBOTH.ini`: **remove** the `ot = blk\.(...)=CPU` line and add:

```ini
cpu-moe                  = true
moe-hybrid               = on
moe-hybrid-experts       = auto
moe-hybrid-reserve       = 1024
moe-hybrid-dma           = on
moe-hybrid-stream        = 0.30
moe-hybrid-stream-prefill = 0
moe-hybrid-migrate       = 256
moe-hybrid-hysteresis    = 2.5
```

(`moe-hybrid-reserve = 1024` is measured: at 128k with the EAGLE draft loaded it leaves 63 slots
and no OOM; 4096 wasted 8 GiB and cost 15 points of decode. The DMA registration tax on CPU reads
is real (~11%: 31.8 -> 28.3 GB/s) and class-2 DRAM contention costs another ~30% of CPU rate, but
class 2 pays it back in GPU work - net wash at 128k, clear win at 16k - so both stay on.)

Every key maps 1:1 to a CLI flag added on this branch (all visible in `llama-server --help`).
Before deploying, run the `scripts/validate.sh` preset-parse tier - `cpu-moe` is a no-value flag
and its ini spelling should be confirmed once. Qwen sections stay unchanged.

## 5. Optimization pass (2026-08-14)

Four axes were investigated on top of the committed architecture; two shipped, two were settled by
measurement:

1. **R9700 stage profile** (new census lines, `GGML_MOE_HYBRID_STATS`): per decode node the class-1
   kernel is 60.6 us for ~2.9 experts - the VRAM bandwidth roofline, nothing to recover there. Per
   batched node the dominant cost was the CPU-side activation gather: 4498 us of single-threaded
   memcpy against 3714 us of MMQ kernel.
2. **GPU-side gather** (`moe_gather_rows`): batched nodes upload the dense activation block once
   and gather on device. Gather 4498 -> 3.1 us, join wait 1581 -> 26 us, **prefill 311.6 -> 380.9
   med = +39% over the production baseline** (was +18%). PPL identical (2.7159). An event-spin
   join replaced the per-node stream sync (class-2 join wait 141 -> 31 us).
3. **Predictor axis refuted by data.** A 3000-token routing trace (216k records) simulated at
   72/128 slots: static long-run top-72 misses 9.0% of visits, the reactive EWMA 17.4%, LRU 13.8%
   at 40 swaps/token, and the Belady perfect-lookahead bound 5.5% at 15.9 swaps/token (~158
   MB/token of migration - costlier than the misses it saves). Token persistence 14.6%; next-layer
   predictability 37% top-4. The routing is near-stationary: the win is longer memory, not
   prediction. **Score decay default raised 0.875 -> 0.98** (half-life ~8.8k tokens); churn drops,
   no downside measured.
4. **iGPU dropped on evidence**: the integrated GPU reports ~1 CU, its arch is not in the fat
   binary (kernels for gfx1201 only), and routing class-2 work to it segfaults. Isolation via
   `--device ROCm0,ROCm1` works (split and pools stay on the R9700s) but there is no compute to
   harvest.

### The decode investigation (12-rep converged runs, acceptance controlled)

Early gates contradicted each other (+35%, then -13/+31%, then -20% on decode). Decomposed, the
variance had three causes, each now closed:

1. **The acceptance lottery.** Under EAGLE, decode t/s multiplies by draft acceptance, which
   ranged 0.539-0.732 across runs of identical configs (temperature sampling -> divergent text ->
   divergent draft luck). Decode verdicts are now taken with speculation OFF only.
2. **A lying census.** `hybrid_stats_record` counted full expert bytes regardless of the row split
   (its GB/s exceeded the hardware ceiling - the tell). Fixed: bytes pro-rated by `cpu_row_end`.
3. **Reserve overshoot.** 4096 MiB/device left 8 GiB idle at 128k while coverage fell to 64%.
   At reserve 1024: 63 slots, 68.9% coverage, no OOM even with the draft loaded. At production
   faith this alone took decode from -20% to -5% and prefill from +6.5% to +12.8% (acc 0.613 vs
   0.606 - no lottery).

The clean acceptance-free verdict at c=131072, 12 reps, speculation off:

| | prefill med | decode med | coverage |
|---|---|---|---|
| baseline | 273.1 | 14.03 | 50% (18 of 36 blocks) |
| dynamic, reserve 1024 | **364.1 (+33%)** | 11.75 (-16%) | **80.1%** |

**Decode is per-node-overhead-bound, not coverage-bound.** At 80% coverage the CPU reads only
~0.27 GiB/token (4 ms) vs the baseline's 0.70 GiB (11 ms) - dynamic should win by ~7 ms/token and
instead loses ~14. The difference, ~21 ms/token, is thread 0's serialized CUDA sequence per hybrid
node (copies + launch + join + scatter, ~290 us x 72 nodes), where the baseline's 18 GPU blocks
run pipelined in the native graph with no joins. More slots cannot fix this; only fewer hybrid
nodes can.

**Consequence - the next build is per-phase placement granularity ("block promotion"):** prefill
keeps per-expert slots (MMQ amortizes all overhead: +33% at 128k), while decode wants whole blocks
either fully GPU-resident and run natively by the scheduler (zero hybrid involvement) or fully on
CPU. Demand picks which blocks promote; slow rotation keeps it adaptive; converged fully-dynamic
then equals the static split of the hot blocks plus adaptivity - greater than or equal to static
by construction. Under tensor split the promoted tensors must interoperate with the meta backend
(join the split, or run single-device against a mirrored src1) - a design spike, not a patch.

Also settled: the prefill class-2 share stays 0 even after the gather fix (MMQ-over-PCIe costs
the GPU more than it saves the CPU: batched kernel 3.76 -> 7.64 ms as the share rises to 0.5).

## 6. True block residency (--moe-hybrid-promote, 2026-08-14 night)

The copy-based class 1 was retired as the decode answer, per the operator's standing spec: an
expert lives in exactly one place and is MOVED between classes, never duplicated. Since the machine
schedules whole nodes and a layer's 128 experts share one tensor, residency is real at BLOCK
granularity:

- `--moe-hybrid-promote N` pre-allocates N pairs of GPU hole tensors at load (the swap buffers of
  the spec), named to match the tensor-split patterns so the meta backend row-splits them across
  both GPUs like loader-placed expert weights.
- At runtime (after demand accumulates), the policy picks blocks; each block's expert tensors are
  copied into holes (~190 ms/tensor, once) and the layer pointers flip. The next graph build runs
  those blocks NATIVELY - pipelined GPU execution, zero per-node dispatch, exactly the baseline's
  static-block cost, but demand-chosen. A flip forces one graph rebuild (reuse would serve the old
  topology).
- Verified: N=14 = 34.45 GiB of holes, split evenly; promotions logged; promoted blocks leave the
  hybrid census; PPL 2.2256 +/- 0.047 over 8 chunks with the flip mid-run (later chunks entirely
  on promoted blocks).
- Traps burned: hole names must parse as REAL block ids (the split callback indexes hparams by the
  id - blk.900 aborts); default server verbosity suppresses all load-time INFO (use -lv 5);
  the loader's ggml contexts have an exact tensor budget (raised by 128 for holes).

Per-phase granularity in full: prefill keeps per-expert slots (MMQ, +33-39%), decode gets whole
promoted blocks (native, overhead-free). The promote-count sweep (VRAM split between the two) is
the current measurement. Rotation/demotion is a latch away (single-shot promotion today).

## 7. Open items

1. **Per-node overhead on 3-tensor-per-block models** (Qwen 122B/35B): fuse per-block launches or
   batch pairs across tensors. This is the remaining kernel-level lever, and it also pays on
   Mistral decode (18% GPU wait at 72 slots with tier 2 off).
2. **Decode acceptance repeats at 128k**: the +35% headline carries an acceptance tailwind
   (0.718 vs 0.608); repeat runs would pin the acceptance-normalized number (~+14%).
3. **`--moe-hybrid-stream auto`** (the adaptive tier-2 controller) mis-steers (15.09 vs 21.78 for
   fixed 0.30 predecessor test) - fixed shares only until rewritten.
4. **"One place only"**: resident experts still have their host pages in the file mapping. With
   mmap those pages are reclaimable cache, and this box has RAM headroom (110 GB vs 92.6 GiB
   model), so nothing is gained today; on a RAM-constrained box, `madvise(MADV_DONTNEED)` on
   fully-resident experts is the follow-up. Note the conflict: tier-2 DMA pins 89.6 GiB, and
   pinned pages cannot be reclaimed - the two features trade against each other on small-RAM hosts.
5. **iGPU as tier-2 server**: implemented, measured a net loss at single-token decode, off by
   default (`HIP_VISIBLE_DEVICES=0,1` keeps it out entirely).
6. Nothing is committed yet; the branch carries the kernel, the knobs, and this report.

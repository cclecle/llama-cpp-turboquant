# The MoE placement engine: from static splits to one demand-driven store

**Date:** 2026-08-14 · **Branch:** `feat/moe-hybrid-kernel` · **Model:** Mistral-Small-4-119B
(Q6_K, 36 blocks × 128 experts, top-4) · **Hardware:** 2× AMD R9700 32 GiB + 12-core Zen4,
110 GB RAM

## The problem

Mistral-Small-4-119B carries **89.6 GiB of expert weights — 96.7% of the model — against 64 GiB
of total VRAM.** Most of the experts must live in host RAM, and someone has to compute them
there. Stock llama.cpp forces a static, load-time answer: a hand-written `-ot` list pinning some
blocks to GPU and some to CPU. That answer is wrong three ways: it idles whichever processor
isn't assigned to the current layer, it ignores the fact that some experts are used far more
than others, and it has to be re-tuned by hand every time the context size or model changes.

## What was built

A placement engine inside the CUDA backend that makes expert residency **dynamic and
demand-driven**. Every expert lives in exactly one of three tiers, and measured routing moves
them between tiers at runtime:

1. **VRAM-resident** — the hottest experts, held in slots sharded across both GPUs, computed at
   full VRAM bandwidth.
2. **GPU-computed from host RAM** — missing experts read in place over PCIe (decode-sized work
   only, where the GPU would otherwise idle).
3. **CPU-computed** — the remainder, on the processor that already owns the memory.

Demand is scored on every layer execution; the resident set follows it. No `-ot` line, no manual
re-tuning: at 128k context the store automatically sizes itself smaller to leave room for the KV
cache and the speculative draft model, exactly where the static split OOMs or needs re-tuning.

## The decisive step: one store for both phases

The path here had two big intermediate results and one unification that made everything click:

- **Prefill** was solved by per-expert VRAM slots + a real tiled matmul (MMQ) with the
  activation gather done on-device, and both processors working the same layer on disjoint
  output rows. This alone reached ~+39% prefill — but decode stayed stubbornly at or below the
  static baseline, because coordinating a CPU-owned layer with the GPU costs ~0.3 ms/layer and
  36 layers of that is a third of every token's budget.
- **Decode** was solved by the *facade*: GPU-owned twin tensors of the experts, picked by the
  graph builder for decode-sized batches, served through per-device pointer tables — a resident
  expert resolves to VRAM, a missing one to mapped host RAM read over both PCIe links. The token
  never crosses to the CPU at all. Decode jitter disappears with it (median == max across reps).
- **The problem that remained:** decode's store (facade shards) and prefill's store (the slot
  pool) were *separate VRAM allocations that couldn't read each other*. Every gigabyte given to
  one phase was invisible to the other, so the facade slot count became a dial trading prefill
  against decode: 12 slots = 337 prefill / 15.4 decode, 48 slots = 271 / 21.4. You could have
  one, not both.

**The unification deleted the second store.** The facade shards are now the *only* resident
store, and the prefill/verify path learned to read them:

- `gate_up` shards are row-splits, so each GPU computes its own disjoint slice of every
  resident expert's output — two launches instead of one, no reduction, and each expert's
  matmul now runs on both GPUs in parallel.
- `down` shards are column-splits (contraction), so each GPU produces a partial sum over its
  half of the input; the second GPU's partial hops directly to the first over P2P, a small add
  kernel joins them, and one copy returns the result. No CPU involvement.
- Shard geometry is never assumed: it is learned at runtime from marker values stamped into the
  facade rows at load, which is what makes the same code correct for the fused/segmented
  `gate_up` split, the `down` contraction split, and any future model's layout.

The old pool code remains for facade-off configurations, but on Mistral it now allocates
**zero bytes** — the census reads `pool: 0.00 GiB, 0 admitted`, with per-device launch counts
exactly balanced.

## What it delivers

All numbers are medians of 12 repetitions on the production harness, against the tuned static
`-ot` + tensor-split baseline.

**16k context, no speculation** — the slot count is now a pure size knob; both axes rise
monotonically until VRAM runs out:

| facade slots | prefill t/s | decode t/s |
|---|---|---|
| baseline (static) | 274 | 14.0 |
| 28 | 274 | 18.7 |
| 48 | 304 | 23.0 |
| 64 | 345 | 26.1 |
| 72 | 371 | 29.4 |
| **80 (max fit, final build)** | **371 (+36%)** | **32.5 (+131%)** |

For contrast, the *best* the pre-unification code could do was 337 prefill (at 15.4 decode) or
21.4 decode (at 271 prefill) — mutually exclusive. The unified store beats both ends of that
trade simultaneously, and decode is now more than **double** the static baseline.

**Production configuration (128k context + EAGLE speculation):**

| config | prefill t/s | decode t/s | acceptance |
|---|---|---|---|
| static baseline | 277 | 19.9 | 0.606 |
| old code's best (facade-28) | 278 | 21.9 | 0.524 |
| unified, 56 slots | 321 (+16%) | 24.7 (+24%) | 0.428 |
| unified, 64 slots | 329 (+18%) | 27.5 (+38%) | 0.528 |
| **final build (64 slots, stream 0, draft 6/0.9)** | **334.5 (+20%)** | **28.8 (+45%)** | 0.465 |

The final-build row carries the acceptance draw *against* it (0.465 vs the baseline's 0.606 —
acceptance swings by prompt mix and multiplies decode), so the +45% decode is an understatement
of the mechanism. The draft settings were re-searched on this structure: chain length 6 beats
the old 12, confirming the common recommendation, while the commonly recommended p-min 0.75
measured strictly worse than 0.9 on this EAGLE-1 draft at every chain length.

## Late additions on the same day

- **Dynamic placement runs continuously again.** The store re-fills on the migration cadence:
  demand from decode (counted on-device by the facade kernel) and prefill both feed the ranking,
  hysteresis keeps near-ties from churning, and topic shifts move the residents. Measured both
  ways: production damping = zero swaps on stable demand; reactive settings = ~50 experts
  re-homed per pass with decode still at 26.7 t/s.
- **Two crashes found by a 6-class prompt basket and fixed**: a stock MMQ buffer under-allocation
  (zero tail padding for our launch shape; the final tile read past the buffer) and a collision
  between HIP-graph capture and the facade's lazy table upload. The single-prompt bench harness
  could not see either; the basket is now the standard validation harness.

## Why the numbers are trustworthy

- **Determinism:** temperature-0, 700-token generations are byte-identical across full server
  restarts — the gate that exercises the facade fill crossing mid-generation.
- **Perplexity:** 150-chunk A/B runs (unified store vs facade-off) agree within error bars
  (2.398 vs 2.407 ± 0.025). 150 chunks matters: the store fills after 96 ubatches, so shorter
  PPL runs never touch the new path.
- **Census accounting:** every expert visit lands in exactly one tier; the shares sum to 100%,
  pool admissions for unified tensors are zero, and per-device launches are balanced to within
  a fraction of a percent.
- **Same harness, same prompts, same rep counts** as every recorded baseline on this box.

## Deployment

```ini
cpu-moe                   = true
moe-hybrid                = on
moe-hybrid-experts        = auto      ; facade-off fallback only
moe-hybrid-reserve        = 1024
moe-hybrid-facade         = on
moe-hybrid-facade-slots   = 64        ; sized for 128k + EAGLE; raise until load stops fitting
moe-hybrid-dma            = on
moe-hybrid-stream         = 0
moe-hybrid-stream-prefill = 0
moe-hybrid-migrate        = 256       ; also the re-fill cadence
moe-hybrid-hysteresis     = 2.5       ; incumbent defense for pool AND re-fill
poll                      = 100
spec-draft-n-max          = 6
spec-draft-p-min          = 0.9
```

## What's next

- **Qwen:** its 3-tensors-per-block layout doubles per-layer coordination and regresses under
  the CPU hybrid; the fix (fusing gate/up launches) is a separate work item. The unified store
  itself transfers as-is — nothing in it is Mistral-specific, since geometry is learned, not
  assumed.
- **Draft tuning:** the EAGLE speculation parameters predate this work and are being re-searched
  on the new structure.

# Qwen3.8-Flash-Next: v16 vs v18 vs R9V (2026-09-26)

**Question:** is our Flash-Next underperforming? R9V (github.com/Dyluhn/R9V v0.4.4, a vLLM fork with gfx1201 kernels)
was run against our v16 (production) and v18 on the same GGUF and the same rig.

**Decision rule:** a gap above 10%, judged separately for prefill and for decode, means v19 ports R9V's approach.

## Setup

- **Model:** the same Unsloth UD-IQ4_XS (3 shards) for all three. Our shards hash-match R9V's package manifest.
- **Rig:** 2x R9700 32 GB (PCI 03:00, 07:00), 110 GB RAM, a Proxmox container. Production was stopped.
  - Locked memory is capped at 8 MiB for everything, production units included.
  - The page cache cannot be dropped between backends.
- **Workload:** one run per backend:
  - 65,536 ctx;
  - a 32,222-token prompt (llama.cpp docs plus a long-answer task, `validation/v19-scripts/mkprompt.sh`);
  - max 2,048 tokens, temperature 0, no ignore_eos;
  - a short warmup request first.
  - The client is the same for all three: `scripts/fleet/openai_bench.py`.
- **v16 / v18 config:** the production `Qwen3.8-Flash-Next:XL` rung args with `--ctx-size 65536`:
  - `-sm tensor`, ub 1024;
  - experts of layers 0-15 on `ROCm0_UVA`;
  - PLE on the CPU;
  - q8_0 KV;
  - shared Q8_0 MTP head at n_max 2, plus the ngram-mod and ngram-map-k4v drafts.
  - ROCm 7.2.4.
- **R9V config:** profile `qwen38-mtp4`, unchanged except `--max-model-len 65536`:
  - image `sha256:2dac17a2...` (WMMA grouped MoE prefill, 4096-token chunks);
  - MTP depth 4 with a block-FP8 MTP checkpoint;
  - expert TP split 416/224;
  - CED off (CED is an approximate prefill).
  - It ran as a plain host process from the unpacked image (`scripts/fleet/oci_unpack.py`, `validation/v19-scripts/r9v-run.sh`), with no docker. ROCm 7.14 userspace, torch 2.11.
  - Not run: R9V's preflight doctor and its setup-time qualification.

## Results

| | prefill t/s (prompt / TTFT) | decode t/s | draft acceptance | tokens / step | ms / step | peak VRAM GPU0 / GPU1 | host RAM peak | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v16 | 958 | 36.2 | 61.5% (1239/2015) | ~2.5 | ~66 | 26.2 / 26.5 GB | 23.1 GB | 11 s |
| v18 | 979 (+2.1%) | 37.2 (+2.8%) | 62.9% (1215/1933) | ~2.5 | ~66 | 26.2 / 26.5 GB | 23.2 GB | 8 s |
| **R9V** | **1,328 (+36% vs v18)** | **61.0 (+64% vs v18)** | 44.1% (1306/2964, depth 4) | ~2.8 | ~45 | 27.0 / 29.0 GB | 49.4 GB | 236 s (first start, compiles) |

- Every run generated the full 2,048 tokens.
- The generated text was similar in length (7.8-8.5k chars). The text itself was not saved, so there is no quality comparison.
- R9V's own reference on its host is 1,641 t/s prefill at 32K, so on our rig it reached 81% of that.
- R9V's quoted 67-76 t/s decode was measured on short prompts.

**Verdict: both gaps are well above the 10% threshold. v19 goes after both prefill and decode.**

## Where R9V's advantage comes from (first read, from its log and profile)

- **Decode: an expert cache in VRAM, not a fixed per-layer placement.** R9V's own log:
  - rank 1 keeps 16.4 GiB of experts resident, with 3.0 GiB cold in host memory;
  - rank 0 keeps 3.2 GiB resident and 32.8 GiB cold, behind an **11.3 GiB LRU cache of 160 expert slots** in VRAM.
  - Each expert's channels are split 416/224 across the two cards.
  - A decode step goes to PCIe only on a cache miss. Our `:XL` rung reads the routed experts of all 16 offloaded layers from host memory on every step.
  - Its per-step time is 45 ms for 5 verified tokens against our 66 ms for 3.
- **Prefill:** a WMMA grouped MoE kernel on 4096-token chunks (group 32). Larger chunks amortise streaming the cold experts. We run ub 1024.
## ROCm version ruled out

The same v18 source was built against ROCm 7.14, a copy of R9V's `core-7.14` in `/opt/rocm-7.14` (clang 23, `validation/v19-scripts/build-rocm714.sh`). It ran on the 7.14 runtime (`libamdhip64.so.7.14.60850` mapped) with the same benchmark:

| | prefill t/s | decode t/s | acceptance |
|---|---:|---:|---:|
| v18, ROCm 7.2.4 | 979 | 37.2 | 62.9% |
| v18, ROCm 7.14 | **482 (-51%)** | 37.9 | 68.9% |

- ~~The newer ROCm halves our prefill.~~ **SUSPECT, to re-measure (found 2026-09-26 during v19 step 1).**
  - The same v18 code with a plain RCCL all-reduce also prefilled at 503 t/s once: that was the first run after two perplexity runs had read the whole GGUF, and the next run gave 978.
  - The `:XL` rung keeps the 28.8 GB PLE table memory-mapped on the CPU, and prefill reads its rows for every prompt token. A cold page cache sends those reads to the NVMe and halves prefill.
  - The ROCm 7.14 run came right after copying 19 GB into `/opt`, which also flushes the cache. The runner now measures a second, warm repetition (`openai_bench.py` warmup 2, `cache_prompt: false`).
  - ROCm 7.14 must be re-measured warm before any conclusion.
- Decode is unchanged, so ROCm is not R9V's decode lever.
- Stay on 7.2.4 for now. The decode gap is R9V's design: its expert cache and placement. Its prefill runs may also have been measured partly cold; re-measure R9V warm too.

## Artefacts

On the box, `/mnt/gguf/r9v/`:
- `bench/`: `results.jsonl`, the logs, the per-second VRAM/RAM samples;
- `rootfs/`: the unpacked runtime;
- `ple/`: the 28.8 GB PLE table, hash-matched to R9V's pin.

## v19 step 1 (branch `v19/flashnext`, tree `/opt/llamacpp/tmp-v19`, `validation/v19-scripts/step1.sh`)

- **1.1 AllReduce hybrid** (`5a9fdf8c9`): small FP32 reductions go through the direct-P2P kernel next to RCCL.
  - Cap: 32,768 elements, below RCCL's own BF16 threshold, so the numerics are unchanged. `GGML_CUDA_AR_HYBRID=0` turns it off.
  - Perplexity at ub 8, where every reduction is decode-sized: bit-identical, 1.4516 off and on.
  - Pooled 6-prompt decode (`:XL` args, 65k): **39.7 → 43.4 t/s**. At matched acceptance the gain is +7.1% to +9.7% (p0 201/294: 40.2 → 44.1; p2 181/262: 41.9 → 45.2).
  - Prefill: unchanged (978 vs 979). Prefill reductions exceed the cap and stay on RCCL.
  - Note: the P2P kernel's GPU time equals NCCL's (61 us/call, both mostly waiting for the peer GPU). The gain is RCCL's host-side cost.
- **1.4 prefill ubatch** (config only, not applied), warm runs:

  | ub | prefill t/s | decode t/s | peak VRAM GPU0/GPU1 (65k ctx) |
  |---|---:|---:|---|
  | 1024 (rung) | 978 | 37.9 | 26.1 / 26.4 GB |
  | 2048 | **1,233** | 37.9 | 27.7 / 28.0 GB |
  | 4096 | **1,387** (above R9V's 1,328) | 37.9 | 30.8 / 31.0 GB |

  The prefill gap to R9V is mostly the ubatch: host experts are streamed once per ubatch. The production `:XL` rung runs at 262k ctx and is sized tighter, so fitting ub 2048/4096 there is the user's rung decision.
- **Host profile** (gdb stack samples during decode): the server's compute thread sits in `ggml_backend_cuda_synchronize` in 38 of 40 samples. CPU sampling (1/40) and graph rebuilds (1/40) do not matter.
  - The ~23 ms of GPU idle per step is the gaps between ~4,400 small kernels per token, not host work.
  - **Steps 1.2 and 1.3 are dropped** as measured not worth it. The idle lever is kernel fusion (launch count).
- **Methodology fix:** runs are now warm (see the ROCm 7.14 correction above): `openai_bench.py` warmup 2, `cache_prompt: false`.

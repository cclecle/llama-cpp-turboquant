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
- **To rule out first:** the ROCm version (7.14 vs our 7.2.4). Next step: v18 built against R9V's unpacked ROCm 7.14, same benchmark.

## Artefacts

On the box, `/mnt/gguf/r9v/`:
- `bench/`: `results.jsonl`, the logs, the per-second VRAM/RAM samples;
- `rootfs/`: the unpacked runtime;
- `ple/`: the 28.8 GB PLE table, hash-matched to R9V's pin.

# v19 campaign runners (2026-09-26)

Qwen3.8-Flash-Next UD-IQ4_XS on three backends, one benchmark each (65,536 ctx, 32k-token prompt, up to 2,048
generated tokens): v16 (production), v18 (v17 + the rdna-boosts harvest), and R9V (github.com/Dyluhn/R9V, a vLLM
fork with gfx1201 kernels). If R9V is more than 10% ahead on prefill or on decode, v19 ports its kernel.
Everything lives on the box under `/mnt/gguf/r9v/` (removing that directory undoes it).

| script | what it does |
|---|---|
| `r9v-run.sh [max len]` | R9V profile `qwen38-mtp4` as a plain host process: the image's `/opt/r9v` venv + ROCm 7.14 unpacked by `scripts/fleet/oci_unpack.py`, env and `vllm serve` args transcribed from R9V `scripts/launch.sh`, container paths mapped to host ones |
| `mkprompt.sh <out> [tokens]` | the benchmark prompt: llama.cpp docs cut to the target with the model's HF tokenizer, then a long-answer task |
| `flashnext.sh [backends]` | the benchmark: production stopped, each backend loaded, warmed up, measured with `scripts/fleet/openai_bench.py`, VRAM/RAM sampled |

R9V setup, in order: `scripts/fleet/r9v_fetch.py` (package + image parts, SHA-256 checked, our shards linked),
`scripts/fleet/oci_unpack.py` (rootfs), R9V `tools/prepare_ple.py` (the 28.8 GB PLE table from our GGUF; its hash
matched R9V's pin `dd55c289...`).

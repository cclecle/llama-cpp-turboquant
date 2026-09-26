#!/bin/bash
# mkprompt.sh <out file> [target tokens, default 32000] : the Flash-Next benchmark prompt. Real text (the llama.cpp
# docs of the v18 tree, a fixed file order) cut to the target size with the model's own tokenizer, then a task that
# asks for a long structured answer (no ignore_eos: the length comes from the model).
set -eu
out=$1; target=${2:-32000}
T=/opt/llamacpp/llama-cpp-mine-v18
# the model's HF tokenizer (R9V package metadata), run by the unpacked R9V venv: llama-tokenize cannot build a
# qwen4exp context ("requires ctx_other")
TOK=/mnt/gguf/r9v/package/metadata/tokenizer.json
PY=/mnt/gguf/r9v/rootfs/opt/r9v/bin/python
task='

---
Task: using only the documentation above, write a complete, detailed technical handbook for an engineer who must deploy and tune this software on a two-GPU AMD workstation. Cover, each in its own numbered section with sub-sections: building for ROCm/HIP, every server option that matters for throughput and memory, KV-cache types and context sizing, speculative decoding, multi-GPU split modes, quantization choices, and troubleshooting. Be exhaustive and concrete, quote option names exactly, and give worked examples. Aim for well over 3000 words.'
count() { "$PY" -c 'import sys, tokenizers; t = tokenizers.Tokenizer.from_file(sys.argv[1]); print(len(t.encode(open(sys.argv[2], encoding="utf-8", errors="ignore").read()).ids))' "$TOK" "$1"; }
docs=$(cd "$T" && ls docs/*.md docs/*/*.md tools/server/README.md | sort)
(cd "$T" && for f in $docs; do printf '\n\n=== %s ===\n\n' "$f"; cat "$f"; done) > "$out.all"
chars=$((target * 3))
for i in 1 2 3 4 5; do
  head -c $chars "$out.all" > "$out"; printf '%s' "$task" >> "$out"
  n=$(count "$out"); echo "chars $chars -> tokens $n"
  [ $((n > target ? n - target : target - n)) -lt 200 ] && break
  chars=$((chars * target / n))
done
rm -f "$out.all"
echo "prompt $out: $(wc -c < "$out") bytes, $n tokens (model tokenizer, no chat template)"

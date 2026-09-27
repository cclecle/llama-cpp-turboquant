#!/bin/bash
# step42.sh : v19 decode, 4 waves per row for more verify mat-vecs, measured in the real decode (step 41's perf loop
# keeps an 8 MB matrix in the 64 MB infinity cache: attn_gate 13 us there, 25 us in the decode trace). D = default
# (tall-K below 2048 rows from 128 K blocks), W = every q8_0 verify mat-vec, M = below 8192 rows from 64 K blocks.
# The reduction order changes, so the text may change; ms/step is the metric. Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
n=0
for arm in D W M M W D; do
  n=$((n + 1))
  case $arm in D) E="X=0" ;; W) E="GGML_CUDA_MMVQ_TALL_K_ROWS=1000000 GGML_CUDA_MMVQ_TALL_K_KBLOCKS=0" ;; M) E="GGML_CUDA_MMVQ_TALL_K_ROWS=8192 GGML_CUDA_MMVQ_TALL_K_KBLOCKS=64" ;; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s42-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s42- results.jsonl
md5sum v19dev-s42-*.answer.txt
echo STEP42_DONE

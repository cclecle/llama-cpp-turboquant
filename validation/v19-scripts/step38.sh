#!/bin/bash
# step38.sh : v19 decode, the coarse draft head's ranking blocks and candidate count (step 37: blocks 0,5 lost 3%
# acceptance, blocks 2,7 kept it at 8192 candidates: 45.3 ms/step, 2.88 tokens/step). One load each
# (deterministic); every text must be the reference 15c95895521e9ee4fc6aa67dc23f6ff1.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
n=0
for cfg in 1,6:8192 3,8:8192 4,9:8192 2,7:4096 3,7:8192 2,7:8192; do
  n=$((n + 1)); b=${cfg%%:*}; r=${cfg##*:}
  FN_ENV="GGML_CUDA_COARSE_HEAD_BLOCKS=$b GGML_CUDA_COARSE_HEAD_ROWS=$r" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s38-b${b/,/}-r$r bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s38- results.jsonl
md5sum v19dev-s38-*.answer.txt
echo STEP38_DONE

#!/bin/bash
# step37.sh : v19 decode, why the coarse draft head lowered acceptance (step 36: 2.88 -> 2.79 tokens/step, text
# identical). A = every row a candidate (the exact pass for all: isolates the q8_1 input rounding of the exact pass
# from the ranking), B = 16384 candidates, C = 32768, D = 8192 with ranking blocks 2,7. One load each (deterministic).
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops ARGMAX (hinted Q6_K head)"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops -o ARGMAX -b ROCm0 -p 'type_a=q6_K' 2>&1 | grep -E 'OK|FAIL|passed' | tail -4)
echo "### throughput"
n=0
for arm in A B C D; do
  n=$((n + 1))
  case $arm in A) E="GGML_CUDA_COARSE_HEAD_ROWS=124160" ;; B) E="GGML_CUDA_COARSE_HEAD_ROWS=16384" ;; C) E="GGML_CUDA_COARSE_HEAD_ROWS=32768" ;; D) E="GGML_CUDA_COARSE_HEAD_BLOCKS=2,7" ;; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s37-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s37- results.jsonl
md5sum v19dev-s37-*.answer.txt
echo STEP37_DONE

#!/bin/bash
# step36.sh : v19 decode, two items from the step 35 anatomy.
#   a) the MTP draft head (4 x 418 us per step, the full Q6_K vocabulary half per device for a single-token greedy
#      draft): GGML_HINT_ARGMAX_ONLY -> rank every row with 2 of its 10 Q6_K blocks, exact logits for the best 8192
#      rows, -inf for the rest (R9V's coarse-to-exact draft head). The target's verify head stays exact, so only
#      draft tokens can change: the TEXT MUST STAY THE REFERENCE. GGML_CUDA_COARSE_HEAD=0: exact.
#   b) the MTP eh_proj [5120 -> 2560] on [5120, 4 streams, n_tokens] went through the batched mat-vec, reading the
#      weight once per token (12 ms per 1024-token ubatch, 72 us per verify): flattened to one matrix of 4*n_tokens
#      columns. Other rounding (mat-mul instead of mat-vec): the text may change. GGML_CUDA_MM_FLATTEN=0: batched.
# Arms: R = both off (must be the reference 15c95895521e9ee4fc6aa67dc23f6ff1), C = coarse head only (must equal R),
# F = both on. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p 'type_a=(q6_K|q8_0)' 2>&1 | grep -E 'FAIL|passed' | tail -6)

echo "### throughput"
n=0
for arm in R C F F C R; do
  n=$((n + 1))
  case $arm in R) E="GGML_CUDA_COARSE_HEAD=0 GGML_CUDA_MM_FLATTEN=0" ;; C) E="GGML_CUDA_MM_FLATTEN=0" ;; F) E="X=0" ;; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s36-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s36- results.jsonl
echo "### generated text per load (R and C must be 15c95895521e9ee4fc6aa67dc23f6ff1)"
md5sum v19dev-s36-*.answer.txt
echo STEP36_DONE

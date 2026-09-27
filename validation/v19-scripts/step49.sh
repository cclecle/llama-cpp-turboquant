#!/bin/bash
# step49.sh : v19 prefill, one q8_1 copy per activation for the MMQs that share it (q/k/v, qkv/gate, the
# hyper-connection down/inject): step 45 counted 5,817 quantize_mmq_q8_1 calls, 0.76 s per GPU per 32k prefill at
# ubatch 4096. Same bytes quantized once: the texts must stay the reference (ub 1024 e78f5272a0921e576d01dec29e164795,
# ub 4096 899f4090eb578cda2e83135d33041213). GGML_CUDA_MMQ_Q8_1_REUSE=0: quantize for each. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops MUL_MAT (q8_0 / q4_K / iq4_xs)"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 2400 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p 'type_a=(q8_0|q4_K|iq4_xs)' 2>&1 | grep -E 'FAIL|passed' | tail -6)
n=0
for arm in R O O R R4 O4; do
  n=$((n + 1)); E="X=0"; [[ $arm == O* ]] && E="GGML_CUDA_MMQ_Q8_1_REUSE=0"
  UB=1024; [[ $arm == *4 ]] && UB=4096
  FN_UB=$UB FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s49-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s49- results.jsonl
md5sum v19dev-s49-*.answer.txt
echo STEP49_DONE

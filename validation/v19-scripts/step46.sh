#!/bin/bash
# step46.sh : v19 prefill, mm_ids_helper with 8 waves per expert over contiguous token ranges (step 45: 660 us per
# call at ubatch 4096, 0.86 s per GPU per 32k prefill). The ranges are merged in order: same outputs, so the text
# must stay the reference 15c95895521e9ee4fc6aa67dc23f6ff1 at ubatch 1024. GGML_CUDA_MMID_WAVES=0: one wave.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops MUL_MAT_ID"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 2400 $B/test-backend-ops -o MUL_MAT_ID -b ROCm0 2>&1 | grep -E 'FAIL|passed' | tail -6)
echo "### throughput: W = 8 waves, O = one wave (ubatch 1024), then W4 / O4 at ubatch 4096"
n=0
for arm in W O O W W4 O4; do
  n=$((n + 1)); E="X=0"; [[ $arm == O* ]] && E="GGML_CUDA_MMID_WAVES=0"
  UB=1024; [[ $arm == *4 ]] && UB=4096
  FN_UB=$UB FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s46-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s46- results.jsonl
md5sum v19dev-s46-*.answer.txt
echo STEP46_DONE

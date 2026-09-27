#!/bin/bash
# step48.sh : v19 prefill, the gated delta net on 4 lanes per state column for batches of >= 32 tokens (32 rows per
# lane: 2 shuffle steps per sum instead of 5, float4 rows); step 45: 2.7 ms per call at ubatch 4096, 1.26 s per GPU.
# Other summation order: the text may change. GGML_CUDA_GDN_COLS=0: one wave per column. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops GATED_DELTA_NET"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o GATED_DELTA_NET -b ROCm0 2>&1 | grep -E 'FAIL|passed' | tail -6)
n=0
for arm in C O O C C4 O4; do
  n=$((n + 1)); E="X=0"; [[ $arm == O* ]] && E="GGML_CUDA_GDN_COLS=0"
  UB=1024; [[ $arm == *4 ]] && UB=4096
  FN_UB=$UB FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s48-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s48- results.jsonl
md5sum v19dev-s48-*.answer.txt
echo STEP48_DONE

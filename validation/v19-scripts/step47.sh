#!/bin/bash
# step47.sh : v19 prefill, the gated delta net loads token t+1's inputs while token t is computed (step 45: 2.7 ms per
# call at ubatch 4096, 1.26 s per GPU per 32k prefill, bound by the per-token load latency). Same arithmetic: the
# texts must be step 46's (ub 1024 15c95895521e9ee4fc6aa67dc23f6ff1, ub 4096 3c8d0479affd5b5b79d8d97fe44534a9);
# baseline prefill 1,372 / 1,924 t/s. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops GATED_DELTA_NET"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o GATED_DELTA_NET -b ROCm0 2>&1 | grep -E 'FAIL|passed' | tail -5)
n=0
for arm in P1 P1 P4; do
  n=$((n + 1)); UB=1024; [ $arm = P4 ] && UB=4096
  FN_UB=$UB FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s47-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s47- results.jsonl
md5sum v19dev-s47-*.answer.txt
echo STEP47_DONE

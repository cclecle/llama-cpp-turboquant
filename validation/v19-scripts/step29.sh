#!/bin/bash
# step29.sh : v19, what fills the compute buffer (ubatch 4096 asks 10.1 GiB per GPU and does not fit; R9V prefills in
# 4096-token chunks). GGML_ALLOC_PEAK=30 makes each reservation log the largest tensors alive at the buffer's peak.
# Load only, at ubatch 1024 and 4096 (the 4096 load fails after the reservation has logged). Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
grep -c GGML_ALLOC_PEAK $T/build3/bin/libggml-base.so

for UB in 1024 4096; do
  FN_LOAD_ONLY=1 FN_UB=$UB FN_ENV="GGML_ALLOC_PEAK=30" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s29-u$UB bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
  echo "### ub $UB: reservations (largest first)"
  grep -E 'alloc peak|MiB  ' v19dev-s29-u$UB.log | cut -c1-190 | head -80
  grep -E 'compute buffer|failed' v19dev-s29-u$UB.log | cut -c1-160 | head -8
done
echo STEP29_DONE

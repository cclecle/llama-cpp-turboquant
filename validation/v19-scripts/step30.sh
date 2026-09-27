#!/bin/bash
# step30.sh : v19, the QSA indexer without its compute-buffer waste (step 29: ubatch 4096 asked 10.1 GiB per GPU, of
# which 7 GiB were 14 device copies of the KQ mask, one per layer's reshape of it, and 2 GiB the indexer scores and
# their relu). ggml_qsa_expand_heads does the relu, the head sum, the block bias and the mask add in one op, in the
# order of the unfused graph, and takes the mask as it is (one copy); the dense path shares one zero source per width.
# The math is unchanged: every text must be the reference (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped.
#   1) build, 2) test-backend-ops QSA_EXPAND, 3) compute-buffer peaks at ubatch 1024 / 2048 / 4096 (load only),
#   4) throughput at ubatch 1024 (2 loads, against step 28's A arms: 1,153 / 1,150 t/s, 46.4-46.5 ms/step), then one
#      load each at 2048 and 4096 if they fit
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops QSA_EXPAND"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops -o QSA_EXPAND -b ROCm0 2>&1 | grep -E 'OK|FAIL|passed' | tail -30)

for UB in 1024 2048 4096; do
  FN_LOAD_ONLY=1 FN_UB=$UB FN_ENV="GGML_ALLOC_PEAK=12" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s30-peak$UB bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
  echo "### ub $UB: the pp reservation"
  grep -E 'alloc peak Meta' -A12 v19dev-s30-peak$UB.log | cut -c1-170 | head -14
done

echo "### throughput"
for tag in U1a U1b U2 U4; do
  case $tag in U1*) UB=1024 ;; U2) UB=2048 ;; U4) UB=4096 ;; esac
  FN_UB=$UB FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s30-$tag bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s30-U results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s30-U*.answer.txt
echo STEP30_DONE

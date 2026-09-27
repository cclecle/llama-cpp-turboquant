#!/bin/bash
# step26.sh : v19, prefill staging of the cold experts (approved by the user 2026-09-27): for MUL_MAT_IDs of at least
# 256 tokens, the next layer's cold experts are DMA'd on a copy stream into one VRAM staging area while the current
# layer computes, and the expert matmuls of that layer read them there (the experts stay in host memory; decode
# unchanged). GGML_CUDA_MOE_STAGE=0: read in place. The staging copies the same bytes: every text must be the
# reference (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped.
#   1) build, 2) test-moe-tiered on both GPUs, 3) A B B A (prefill t/s is the point) + md5 + the staging log line
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-moe-tiered"
for d in ROCm0 ROCm1; do
  (cd /tmp && HIP_VISIBLE_DEVICES=0,1 timeout 900 $B/test-moe-tiered $d > /mnt/gguf/r9v/bench/tmt-s26-$d.log 2>&1); echo "$d: $(tail -1 tmt-s26-$d.log) (fails: $(grep -c FAIL tmt-s26-$d.log))"
done

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_MOE_STAGE=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s26-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s26- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s26-*.answer.txt
grep -h 'prefill staging' v19dev-s26-B2.log | head -2 | cut -c1-200
echo STEP26_DONE

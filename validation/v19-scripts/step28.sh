#!/bin/bash
# step28.sh : v19, the prefill staging with several areas (GGML_CUDA_MOE_STAGE_AREAS): step 27 showed the expert
# matmuls 2.2x faster from VRAM (10.5 -> 4.7 s per GPU) but the copies (~26 GB/s per GPU) not hidden with one area:
# layer L+1's copy had to fit between the end of layer L's expert matmuls and the start of L+1's. With two areas the
# next layer but one is copied while the next one is read.
# A = staging off, B = 2 areas (default), C = 3 areas. Every text must be the reference
# (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped. 1) build, 2) A B C C B A (prefill t/s is the point)
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -4)
echo "### throughput, warm, A B C C B A"
n=0
for arm in A B C C B A; do
  n=$((n + 1))
  case $arm in A) E="GGML_CUDA_MOE_STAGE=0" ;; B) E="X=0" ;; C) E="GGML_CUDA_MOE_STAGE_AREAS=3" ;; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s28-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s28- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s28-*.answer.txt
echo STEP28_DONE

#!/bin/bash
# step20.sh : v19, kernel-count cuts, both bit-identical by construction:
#   - GGML_CUDA_FUSE_Q8_1 (step 19) now also for the hyper-connection norm, whose mat-vec reads it through a reshape
#     ([2560, 4, T] as [10240, T]: no row padding, so the q8_1 blocks are the same; the entry is keyed by the view)
#   - GGML_CUDA_FUSE_CPY_MULTI (default on): a run of up to 8 consecutive copies with one layout in one launch (the
#     rollback tails of each convolution state, one copy per speculative slot: 5 per gated delta net layer)
# A = both off, B = default. Every text must be the reference (8a0ec2a9ccd2816c6250db15e16181eb).
# Production stays stopped. 1) A B B A + md5, 2) trace of B: kernels per step
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### throughput, warm, A B B A"
n=0
for arm in A B B A; do
  n=$((n + 1)); E="X=0"; [ $arm = A ] && E="GGML_CUDA_FUSE_Q8_1=0 GGML_CUDA_FUSE_CPY_MULTI=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s20-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s20- results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s20-*.answer.txt

echo "### kernel trace: B"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s20 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s20
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -12
timeout 600 python3 kernel_ngrams.py $f 1 30 2>&1 | head -24
echo STEP20_DONE

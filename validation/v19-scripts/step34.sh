#!/bin/bash
# step34.sh : v19 prefill, sparse QSA attention on RDNA4. The dense QSA prefill runs FA over every cell with the
# combined top-k + causal mask; upstream's sparse FA (NVIDIA only) turns the mask into one list per 8 queries (the
# columns any of them can see) and gathers only those, reading the mask at each. Ported to HIP (wave32 ballot),
# enabled at the (8 queries, 8 heads) tile from 4096 cells. GGML_CUDA_FA_SPARSE=0: dense. The tiling of the online
# softmax changes, so the text changes; D (dense) must stay the reference 0c8460b131f6183795627bea73282ee2.
# Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)

echo "### test-backend-ops FLASH_ATTN_EXT head 256"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 2400 $B/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 -p 'hsk=256' 2>&1 | grep -E 'FAIL|passed' | tail -10)

echo "### throughput: P = sparse, D = GGML_CUDA_FA_SPARSE=0 (ubatch 1024), then ubatch 4096"
n=0
for arm in P D D P P4 D4; do
  n=$((n + 1)); E="X=0"; [[ $arm == D* ]] && E="GGML_CUDA_FA_SPARSE=0"
  UB=1024; [[ $arm == *4 ]] && UB=4096
  FN_UB=$UB FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s34-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
done
python3 step_ms.py v19dev-s34- results.jsonl
echo "### generated text per load (md5; D at ubatch 1024 must be 0c8460b131f6183795627bea73282ee2)"
md5sum v19dev-s34-*.answer.txt
echo STEP34_DONE

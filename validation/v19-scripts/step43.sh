#!/bin/bash
# step43.sh : v19 decode, flash-decoding over the QSA lists (fattn-sparse-decode.cu): a block per (chunk of 64 list
# entries, query, K/V head) that loads each listed K/V row once for the 12 Q heads, then an in-order combine. The
# mma kernel ran ~79 us + a 10 us fixup per call, 17 calls per step (R9V's decode kernel: ~14 us). Other rounding:
# the text may change. GGML_CUDA_FA_SPARSE_DECODE=0: the mma kernel. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops FLASH_ATTN_EXT (kv_idx lists)"
(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 -p 'n_sel' 2>&1 | grep -E 'FAIL|passed' | tail -8)
echo "### perf: new, then mma"
for E in X=0 GGML_CUDA_FA_SPARSE_DECODE=0; do
  echo "# $E"
  (cd /tmp && env $E HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops perf -o FLASH_ATTN_EXT -b ROCm0 -p 'n_sel' 2>&1 | grep -E 'FLASH_ATTN' | sed 's/  */ /g' | cut -c1-170)
done
echo "### throughput: N = new, M = mma"
n=0
for arm in N M M N; do
  n=$((n + 1)); E="X=0"; [ $arm = M ] && E="GGML_CUDA_FA_SPARSE_DECODE=0"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s43-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s43- results.jsonl
md5sum v19dev-s43-*.answer.txt
echo STEP43_DONE

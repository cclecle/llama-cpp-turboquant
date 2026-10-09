#!/bin/bash
# s1-cache.sh : v20 step 1, the expert cache (--moe-cache-slots) on Flash-Next, and that v20 without it is v19.
#   1. test-moe-tiered with the cache off and with 4 slots (bit-identical to the plain buffer, evictions included)
#   2. flashnext.sh on the r9v-like tiered args (ub 1024, 65k, 32k prompt): v19dev, v20dev cache off, 16 and 32 slots,
#      in an interleaved order; the texts must all be the v19 reference e78f5272a0921e576d01dec29e164795 (the cache
#      copies bytes, the matmuls read the same bytes) and the v20dev-off ms/step must equal v19dev's
#   3. one 16-slot run without graphs and with GGML_CUDA_MOE_CACHE_STATS: hit rates per layer
# Production stays stopped. Needs /opt/llamacpp/llama-cpp-mine-v20dev -> /opt/llamacpp/tmp-v20 (built).
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/tmp-v20/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### test-moe-tiered"
for s in 0 4; do
  echo "slots $s: $(cd /tmp && GGML_CUDA_MOE_CACHE_SLOTS=$s HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-moe-tiered 2>&1 | grep -cE 'FAIL|differ') failing lines; $(cd /tmp && GGML_CUDA_MOE_CACHE_SLOTS=$s HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-moe-tiered 2>&1 | tail -1)"
done
n=0
for arm in v19 off c16 c32 c16 off c32 v19; do
  n=$((n + 1)); b=v20dev; E="X=0"
  case $arm in v19) b=v19dev;; c16) E="GGML_CUDA_MOE_CACHE_SLOTS=16";; c32) E="GGML_CUDA_MOE_CACHE_SLOTS=32";; esac
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s1-$arm-$n bash flashnext.sh $b 2>&1 | grep -E 'load:|died|vram'
done
echo "### stats (no graphs)"
FN_ENV="GGML_CUDA_MOE_CACHE_SLOTS=16 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_CACHE_STATS=4800" FN_ARGS=fn-xl-tiered-r9vlike.args \
  FN_TAG=-s1-stats bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died'
grep -h "moe cache:" v20dev-s1-stats.log | tail -12
python3 step_ms.py v19dev-s1- results.jsonl; python3 step_ms.py v20dev-s1- results.jsonl
md5sum v19dev-s1-*.answer.txt v20dev-s1-*.answer.txt
echo S1_DONE

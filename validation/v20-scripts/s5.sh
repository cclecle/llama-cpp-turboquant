#!/bin/bash
# s5 : rebuild tmp-v20 (meta split-state fix), the :XS crash scenario on v20dev, then Flash-Next off / c32 md5 check
cd /mnt/gguf/r9v/bench
(cd /opt/llamacpp/tmp-v20 && bash validation/v17-scripts/build-scratch.sh /opt/llamacpp/tmp-v20 2>&1 | head -3)
S3_TAG=-fix bash s3-crash-bt.sh v20dev 2>&1 | grep -E 'request|crash|FAULT|S3_DONE' | head
for arm in off c32; do
  E="X=0"; [ $arm = c32 ] && E="GGML_CUDA_MOE_CACHE_SLOTS=32"
  echo "### fn-$arm loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s5-$arm bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v20dev-s5- results.jsonl
md5sum v20dev-s5-*.answer.txt
echo S5_DONE

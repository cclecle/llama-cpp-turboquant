#!/bin/bash
# step40.sh : v19 decode, which matrices the dense mat-vecs are (step 35: 524 calls, 8.3 ms per step vs R9V 192,
# 3.9 ms). GGML_CUDA_MMV_NAMES=1 logs each distinct mat-vec once: weight name, type, shape, columns. One short load.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
FN_WARM=1 FN_ENV="GGML_CUDA_MMV_NAMES=1" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s40 bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
grep -o 'mmv: .*' v19dev-s40.log | sort | uniq | grep -v '|ids' | sed 's/^mmv: //' | sort -t'|' -k4 | head -120
echo STEP40_DONE

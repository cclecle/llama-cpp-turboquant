#!/bin/bash
# step39.sh : v19 decode, draft sampling on the GPU under the tensor split. llama_context::set_sampler refuses the
# backend sampler with SPLIT_MODE_TENSOR, so every draft token copies its logits to the host and runs top-k(10) on the
# CPU. LLAMA_TP_BACKEND_SAMPLING=1 lets it through (the sampler reads the gathered logits): does it run, keep the text
# (the reference 15c95895521e9ee4fc6aa67dc23f6ff1) and save host time? Also the new coarse-head default (blocks 2,7).
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
n=0
for arm in D S S D; do
  n=$((n + 1)); E="X=0"; [ $arm = S ] && E="LLAMA_TP_BACKEND_SAMPLING=1"
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s39-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
  grep -h -E "backend sampling|backend offload|GGML_ASSERT|error" v19dev-s39-$arm$n.log | head -3
done
python3 step_ms.py v19dev-s39- results.jsonl
md5sum v19dev-s39-*.answer.txt
echo STEP39_DONE

#!/bin/bash
# step13.sh : v19, the ROCm version and the HIP runtime knobs, same source (step 12 minus the reuse kernel).
#   B = v19dev on ROCm 7.2.4 (the host's)
#   R = the same binary on the ROCm 7.14 runtime (R9V's; LD_LIBRARY_PATH first): the runtime and driver interface only
#   F = the source built with ROCm 7.14 (llama-cpp-mine-v19-rocm714, build-v19-rocm714.sh): compiler + runtime
#   K = B + ROC_ACTIVE_WAIT_TIMEOUT=10000 (the host spins up to 10 ms on a sync instead of sleeping on an interrupt)
#       + HIP_FORCE_DEV_KERNARG=1 (kernel arguments in device memory)
# Warm runs (the first measured 7.14 prefill was page-cache cold). A watcher logs which libamdhip64 every server maps.
# Production stays stopped. Throughput B R F K K F R B.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
L714=/opt/rocm-7.14/lib
( for i in $(seq 1 600); do
    for p in $(pgrep -x llama-server); do
      echo "$(readlink /proc/$p/exe) $(grep -m1 -o '/[^ ]*libamdhip64[^ ]*' /proc/$p/maps)"
    done
    sleep 3
  done ) > hiplibs-s13.raw 2>/dev/null &
W=$!

echo "### throughput, warm, B R F K K F R B"
n=0
for arm in B R F K K F R B; do
  n=$((n + 1))
  case $arm in
    B) FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s13-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:' ;;
    R) FN_LD="$L714:/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin" FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args \
         FN_TAG=-s13-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:' ;;
    F) FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s13-$arm$n bash flashnext.sh v19-rocm714 2>&1 | grep -E 'load:' ;;
    K) FN_ENV="ROC_ACTIVE_WAIT_TIMEOUT=10000 HIP_FORCE_DEV_KERNARG=1" FN_ARGS=fn-xl-tiered-r9vlike.args \
         FN_TAG=-s13-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:' ;;
  esac
done
kill $W 2>/dev/null
python3 step_ms.py v19dev-s13- results.jsonl
python3 step_ms.py v19-rocm714-s13- results.jsonl
echo "### libamdhip64 mapped (server exe, library)"
sort hiplibs-s13.raw | uniq -c
echo STEP13_DONE

#!/bin/bash
# step50.sh : v19, the hyper-connection combine (dsv4_hc_post, identity) fused into the next grouped RMS norm and its
# weight: the norm's first pass computes and writes the combined residual, so it is not read back (one 168 MB pass
# per HC at ubatch 4096) and one kernel per HC goes (~96 per decode pass). Same threads, sums and block size as the
# norm: the texts must stay ub 1024 e78f5272a0921e576d01dec29e164795, ub 4096 899f4090eb578cda2e83135d33041213.
# GGML_CUDA_FUSE_HC_NORM=0: separate kernels. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
B=$T/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
echo "### test-backend-ops"
for op in DSV4_HC_POST_NORM DSV4_HC_POST RMS_NORM; do
  echo "$op: $(cd /tmp && HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o $op -b ROCm0 2>&1 | grep -E 'tests passed|FAIL' | head -3 | tr '\n' ' ')"
done
n=0
for arm in F S S F F4 S4; do
  n=$((n + 1)); E="X=0"; [[ $arm == S* ]] && E="GGML_CUDA_FUSE_HC_NORM=0"
  UB=1024; [[ $arm == *4 ]] && UB=4096
  FN_UB=$UB FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s50-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
done
python3 step_ms.py v19dev-s50- results.jsonl
md5sum v19dev-s50-*.answer.txt
echo STEP50_DONE

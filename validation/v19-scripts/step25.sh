#!/bin/bash
# step25.sh : v19, prefill vs ubatch size (information for the user's rung decision; no preset is changed). The step 23
# trace: prefill GPU time is 50% mul_mat_q, mostly the cold experts, which every 1024-token ubatch reads again over
# PCIe from host memory (all 512 experts of every layer are routed to in a ubatch). A larger ubatch reads them once
# per more tokens, at the price of a larger compute buffer. U1 = ubatch 1024 (the rung), U2 = 2048, U4 = 4096 (batch
# raised to the ubatch where needed); today's hot set; 65k ctx. Production stays stopped. Order U1 U2 U4 U4 U2 U1.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### prefill / decode / VRAM vs ubatch"
n=0
for arm in U1 U2 U4 U4 U2 U1; do
  n=$((n + 1))
  case $arm in U1) UB=1024 ;; U2) UB=2048 ;; U4) UB=4096 ;; esac
  FN_UB=$UB FN_ENV="X=0" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s25-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died|timeout'
done
python3 step_ms.py v19dev-s25-U results.jsonl
echo STEP25_DONE

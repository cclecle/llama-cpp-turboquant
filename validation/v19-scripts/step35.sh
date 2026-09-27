#!/bin/bash
# step35.sh : v19 decode, a fresh per-step anatomy after steps 29-34 (decode 45.5 ms/step vs R9V 41 at today's hot
# set): kernel trace of a 32k-prompt, 512-token run, the per-step category table (torchtrace_anatomy.py) and the
# kernels with the most time per step (kernel_ngrams.py). Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s35 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s35
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
echo "### per-step anatomy ($steps steps)"
python3 torchtrace_anatomy.py rocprof $f $steps
echo "### kernel n-grams"
timeout 900 python3 kernel_ngrams.py $f 1 40 2>&1 | head -60
echo STEP35_DONE

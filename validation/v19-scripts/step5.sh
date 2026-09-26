#!/bin/bash
# step5.sh : same configuration as R9V, to find the core-performance culprit. Our tiered rung set up like R9V's
# qwen38-mtp4 profile: MTP drafts only (no ngram drafters), depth 4, no confidence cut-off, greedy. Same 32k prompt.
#   1) throughput, warm, two loads per engine (the box drifts about ±6% between loads)
#   2) kernel traces of both engines on the same request (512 tokens), compared per decode step
# Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
python3 - <<'PY'
a = open('fn-xl-tiered.args').read().split()
def setv(k, v):
    i = a.index(k); a[i + 1] = v
setv('--spec-type', 'draft-mtp')
setv('--spec-draft-n-max', '4')
open('fn-xl-tiered-r9vlike.args', 'w').writelines(x + chr(10) for x in a)
PY
echo "### throughput: ours (tiered, MTP 4, no ngram) x2, R9V x2"
for r in 1 2; do
  FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-r9vlike-$r bash flashnext.sh v19dev 2>&1 | grep -E '^\{|memory:|load:'
  FN_TAG=-$r bash flashnext.sh r9v 2>&1 | grep -E '^\{|memory:|load:'
done
echo "### kernel trace: ours"
DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-r9vlike bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
echo "### kernel trace: R9V"
bash r9v-profile.sh prompt-32k.txt 512
echo "### step anatomy: ours"
python3 step_anatomy.py $(find prof-decode-v19dev-r9vlike -name '*kernel_trace.csv' | head -1) 150
echo "### step anatomy: R9V (each worker process)"
for f in $(find prof-decode-r9v -name '*kernel_trace.csv'); do
  n=$(wc -l < $f); [ $n -lt 10000 ] && continue
  echo "== $f ($n dispatches)"; python3 step_anatomy.py $f 150
done
echo STEP5_DONE

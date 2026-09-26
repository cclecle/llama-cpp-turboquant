#!/bin/bash
# step5b.sh : the R9V half of step 5 again (its trace was lost to an early SIGKILL), then both anatomies.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
bash r9v-profile.sh prompt-32k.txt 512
for f in $(find prof-decode-r9v -name '*kernel_trace.csv'); do
  n=$(wc -l < $f); [ $n -lt 10000 ] && { echo "skip $f ($n lines)"; continue; }
  echo "== $f ($n dispatches)"; python3 step_anatomy.py $f 150
done
echo STEP5B_DONE

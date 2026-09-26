#!/bin/bash
cd /root/work-20260925
N=/opt/llamacpp/tmp-rdnab/build3/bin; O=/opt/llamacpp/llama-cpp-mine-v16/build3/bin
(HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$N timeout 7200 $N/test-backend-ops -b ROCm0 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > full_new.txt) &
(HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$O timeout 7200 $O/test-backend-ops -b ROCm0 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > full_old.txt) &
wait
for f in full_new full_old; do grep '^\[[A-Z_0-9]*\].*: FAIL$' $f.txt | grep -v 'prec=def' | sed -E 's/ERR = [0-9.]+/ERR/' > $f.fail; done
echo TBOFULL_DONE > tbofull.done

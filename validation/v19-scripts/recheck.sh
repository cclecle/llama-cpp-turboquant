#!/bin/bash
# re-check after the step 3 slowdown: tiered ub1024, tiered ub2048, UVA ub2048 (control)
cd /mnt/gguf/r9v/bench
FN_ARGS=fn-xl-tiered.args FN_TAG=-tiered-re bash flashnext.sh v19dev 2>&1 | grep -v "changed on disk"
FN_ARGS=fn-xl-tiered.args FN_UB=2048 FN_TAG=-tiered-ub2048-re bash flashnext.sh v19dev 2>&1 | grep -v "changed on disk"
FN_UB=2048 FN_TAG=-uva-ub2048-re bash flashnext.sh v19dev 2>&1 | grep -v "changed on disk"
echo RECHECK_DONE

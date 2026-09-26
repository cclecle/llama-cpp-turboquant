#!/bin/bash
# A/B series v16 vs rdnab2 on the Qwen rungs; production already stopped by the caller
cd /root/work-20260925
sed -i 's/^8-15$/0-7/' a27q.args
export MP_NO_RESTART=1
run() { echo "##### $1 dev=$2"; bash multi-prompt.sh $1 $2 v16 rdnab2 2>&1 | grep -E 'MEAN| p[0-9] '; }
run a27.args 0
run a27q.args 0
run a35.args 0
run afn.args 0,1
run atp.args 0,1
echo SERIES_DONE

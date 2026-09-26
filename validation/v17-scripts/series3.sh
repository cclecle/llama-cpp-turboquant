#!/bin/bash
# series 3: adaptive MTP depth vs fixed on 27B, deep-context TP and Flash-Next; production stopped by caller
cd /root/work-20260925
export MP_NO_RESTART=1
python3 - <<'PY'
a=open('a27.args').read().split('\n')
i=a.index('--spec-type'); a[i+1]=a[i+1].replace('draft-mtp,','draft-mtp-adaptive,')
j=a.index('--spec-draft-n-max'); a[j+1]='8'
open('a27ad.args','w').write('\n'.join(a))
PY
grep -A1 -E -- '--spec-type|--spec-draft-n-max' a27ad.args | tr '\n' ' '; echo
run() { echo "##### $*"; bash multi-prompt.sh "$@" 2>&1 | grep -E 'MEAN'; }
MP_ARGS_B=a27ad.args run a27.args 0 rdnab3 rdnab3b
MP_CHARS=120000 MP_ARGS_B=a27ad.args run a27.args 0 rdnab3 rdnab3b
MP_CHARS=120000 run atp.args 0,1 v16 rdnab3
MP_CHARS=120000 run afn.args 0,1 v16 rdnab3
echo SERIES_DONE

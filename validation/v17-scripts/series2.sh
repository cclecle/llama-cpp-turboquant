#!/bin/bash
# series 2: rdnab3 (upstream nwarps restored) vs v16, plus deep-context 27B band A/B; production stopped by caller
cd /root/work-20260925
export MP_NO_RESTART=1
run() { echo "##### $*"; bash multi-prompt.sh "$@" 2>&1 | grep -E 'MEAN'; }
run a35.args 0 v16 rdnab3
run afn.args 0,1 v16 rdnab3
export MP_CHARS=120000
run a27.args 0 v16 rdnab3
run a27q.args 0 v16 rdnab3
MP_ENV_B='GGML_HIP_FA_BAND_WMMA=0' run a27.args 0 rdnab3 rdnab3b
MP_ENV_B='GGML_HIP_FA_BAND_WMMA=0' run a27q.args 0 rdnab3 rdnab3b
echo SERIES_DONE

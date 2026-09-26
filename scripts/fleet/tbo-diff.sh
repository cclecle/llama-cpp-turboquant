#!/bin/bash
# tbo-diff.sh <release A> <release B> [op] : full test-backend-ops (or one op) for two builds in parallel, A on GPU 0
# and B on GPU 1, then the failure lists compared (FLASH_ATTN_EXT prec=def excluded: unreachable from llama.cpp).
# Production must be stopped by the caller. Output: tbo-<release>.txt / .fail in the cwd.
set -u
A=${1:?release A}; B=${2:?release B}; OP=${3:-}
run() { local v=$1 dev=$2 bin=/opt/llamacpp/llama-cpp-mine-$1/build3/bin
  HIP_VISIBLE_DEVICES=$dev LD_LIBRARY_PATH=$bin timeout 7200 $bin/test-backend-ops ${OP:+-o $OP} -b ROCm0 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > tbo-$v.txt
  grep '^\[[A-Z_0-9]*\].*: FAIL$' tbo-$v.txt | grep -v 'prec=def' | sed -E 's/ERR = [0-9.]+/ERR/' | sort > tbo-$v.fail; }
run $A 0 & run $B 1 & wait
for v in $A $B; do echo "$v: $(grep -c 'OK$' tbo-$v.txt) ok, $(wc -l < tbo-$v.fail) reachable fail"; done
echo "--- only in $B:"; comm -13 tbo-$A.fail tbo-$B.fail | cut -c1-220
echo "--- only in $A:"; comm -23 tbo-$A.fail tbo-$B.fail | cut -c1-220

#!/bin/bash
# fa_sweep.sh "nthreads occupancy nbatch_fa" ... : the RDNA mma flash-attention config for (DKQ 256, DV 256, 16
# columns), the one the qwen4exp sparse verify attention (1 query x 12 GQA heads per tile, lists of 2,051 cells)
# and the MTP draft attention (gqa16) run with, rebuilt per variant in the dev tree. Per variant: the build result,
# test-backend-ops on the sparse FA cases (n_sel=), and the FLASH_ATTN_EXT perf cases (make_test_cases_perf). The
# original entry is restored (and rebuilt) at the end. Needs an idle GPU 0; production must be stopped.
T=/opt/llamacpp/tmp-v19
H=$T/ggml/src/ggml-cuda/fattn-mma-f16.cuh
B=$T/build3/bin
cp $H /tmp/fattn-mma-f16.cuh.orig
set_cfg() { # <nthreads> <occupancy> <nbatch_fa>: rewrite the (256, 256, 16) entry of ggml_cuda_fattn_mma_get_config_rdna
  python3 - "$H" "$1" "$2" "$3" <<'PY'
import re, sys
p, nt, occ, nb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(p).read()
i = s.index('ggml_cuda_fattn_mma_get_config_rdna(')
j = s.index('return fattn_mma_config', i)
body = s[i:j]
pat = re.compile(r'GGML_CUDA_FATTN_MMA_CONFIG_CASE\(256, 256, 16, *\d+, *\d+, *\d+,')
assert len(pat.findall(body)) == 1, 'entry not found'
body = pat.sub('GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, %3s, %s, %3s,' % (nt, occ, nb), body)
open(p, 'w').write(s[:i] + body + s[j:])
PY
  grep -n 'CONFIG_CASE(256, 256, 16' $H | sed -n 3p
}
run_variant() {
  echo "=== variant: nthreads $1 occupancy $2 nbatch_fa $3"
  set_cfg "$1" "$2" "$3"
  (cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -3)
  echo "correctness: $(HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 -p n_sel= 2>&1 | grep -E 'tests passed|FAIL' | head -3 | tr '\n' ' ')"
  HIP_VISIBLE_DEVICES=0 timeout 900 $B/test-backend-ops perf -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -E 'us/run' | sed -E 's/^ *//' | cut -c1-230
}
for v in "$@"; do
  run_variant $v
done
cp /tmp/fattn-mma-f16.cuh.orig $H
echo "=== restored the original entry"
grep -n 'CONFIG_CASE(256, 256, 16' $H | sed -n 3p
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -2)
echo FA_SWEEP_DONE

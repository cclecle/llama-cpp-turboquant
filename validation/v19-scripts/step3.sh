#!/bin/bash
# step3.sh : v19 step 3 on the dev tree (tiered args from step2.sh). Production stays stopped.
#   a) pooled 6-prompt decode, A B B A (multi-prompt.sh MP_ABBA=1): AllReduce hybrid off vs on (UVA rung),
#      UVA vs tiered, then MTP depth on the tiered rung: spec-draft-n-max 2 vs 3, 2 vs 4
#   b) flashnext warm: tiered with the greedy hot set, tiered + ub 2048
cd /mnt/gguf/r9v/bench
MP=/opt/llamacpp/tmp-v19/scripts/fleet/multi-prompt.sh
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
sed "s|^/mnt/gguf/r9v/bench/hot-.*\.txt$|$(readlink -f hot-r9v-uva16-greedy.txt)|" fn-xl-tiered.args > fn-xl-tiered-greedy.args
for n in 3 4; do
  awk -v n=$n 'p { print n; p = 0; next } $0 == "--spec-draft-n-max" { p = 1 } { print }' fn-xl-tiered-65k.args > fn-xl-tiered-65k-n$n.args
done
pooled() { # <title> <env / args for multi-prompt>...
  echo "### pooled decode ABBA: $1"; shift
  env MP_NO_RESTART=1 MP_ABBA=1 "$@" 2>&1 | grep -E 'MEAN|p[0-9] '
}
pooled "AllReduce hybrid off (A) vs on (B), UVA rung" MP_ENV_A=GGML_CUDA_AR_HYBRID=0 MP_ENV_B=X=0 bash $MP fn-xl-65k.args 0,1 v19dev v19dev
pooled "UVA rung (A) vs tiered uniform (B)" MP_ARGS_B=fn-xl-tiered-65k.args bash $MP fn-xl-65k.args 0,1 v19dev v19dev
pooled "tiered: MTP n_max 2 (A) vs 3 (B)" MP_ARGS_B=fn-xl-tiered-65k-n3.args bash $MP fn-xl-tiered-65k.args 0,1 v19dev v19dev
pooled "tiered: MTP n_max 2 (A) vs 4 (B)" MP_ARGS_B=fn-xl-tiered-65k-n4.args bash $MP fn-xl-tiered-65k.args 0,1 v19dev v19dev

echo "### flashnext: tiered greedy hot set, tiered uniform + ub 2048"
FN_ARGS=fn-xl-tiered-greedy.args FN_TAG=-tiered-greedy bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
FN_ARGS=fn-xl-tiered.args FN_UB=2048 FN_TAG=-tiered-ub2048 bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
echo STEP3_DONE

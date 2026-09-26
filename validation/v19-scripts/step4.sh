#!/bin/bash
# step4.sh : MTP draft depth with a confidence cut-off (--spec-draft-p-min: the draft stops when its top token's
# probability falls below p), on the tiered rung. Pooled 6-prompt decode, A B B A, A = the rung as it is
# (n_max 2, p_min 0). Production stays stopped.
cd /mnt/gguf/r9v/bench
MP=/opt/llamacpp/tmp-v19/scripts/fleet/multi-prompt.sh
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
variant() { # <n_max> <p_min> -> args file name
  local f=fn-xl-tiered-65k-n$1-p$2.args
  awk -v n=$1 'p { print n; p = 0; next } $0 == "--spec-draft-n-max" { p = 1 } { print }' fn-xl-tiered-65k.args > $f
  printf -- '--spec-draft-p-min\n%s\n' $2 >> $f
  echo $f
}
for v in "4 0.7" "4 0.5" "4 0.85" "3 0.7" "2 0.7"; do
  set -- $v
  f=$(variant $1 $2)
  echo "### pooled decode ABBA, tiered: n_max 2 p_min 0 (A) vs n_max $1 p_min $2 (B)"
  MP_NO_RESTART=1 MP_ABBA=1 MP_ARGS_B=$f bash $MP fn-xl-tiered-65k.args 0,1 v19dev v19dev 2>&1 | grep -E 'MEAN'
done
echo STEP4_DONE

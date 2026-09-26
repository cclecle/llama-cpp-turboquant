#!/bin/bash
# step2.sh [hot file, default hot-r9v-uva16-uniform.txt] : v19 step 2, per-expert hot/cold placement (ROCmN_TIERED)
# on the dev tree. Same VRAM as the :XL rung (hot file sized with moe_hotset.py --match-uva 16). Production stays
# stopped. 1) test-moe-tiered on both GPUs, 2) perplexity tiered vs the UVA rung (bit-identical expected; ub 1024 for
# the MMQ paths, ub 8 for mmvq), 3) flashnext warm: UVA baseline, tiered, 4) pooled decode UVA vs tiered, 5) profile.
cd /mnt/gguf/r9v/bench
HOT=$(readlink -f ${1:-hot-r9v-uva16-uniform.txt})
TAG=$(basename $HOT .txt)
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
OT_UVA='per_layer_token_embd=CPU,blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)\.ffn_(gate|up|down)_exps\.weight=ROCm0_UVA'
OT_TIER='per_layer_token_embd=CPU,blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED'
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
# the tiered rung: the :XL args with the -ot replaced and the hot file added
python3 - "$OT_TIER" "$HOT" <<'PY'
import sys
a = open('fn-xl.args').read().split()
i = a.index('--override-tensor'); a[i + 1] = sys.argv[1]
a += ['--moe-hot-experts', sys.argv[2]]
open('fn-xl-tiered.args', 'w').writelines(x + chr(10) for x in a)
PY
sed 's/^262144$/65536/' fn-xl-tiered.args > fn-xl-tiered-65k.args

echo "### test-moe-tiered"
for d in ROCm0 ROCm1; do
  (cd /tmp && HIP_VISIBLE_DEVICES=0,1 timeout 900 $B/test-moe-tiered $d > /mnt/gguf/r9v/bench/tmt-$d.log 2>&1); echo "$d: $(tail -1 tmt-$d.log) (fails: $(grep -c FAIL tmt-$d.log))"
done
grep FAIL tmt-ROCm0.log | head -5

echo "### perplexity: UVA rung vs tiered ($TAG), expect bit-identical"
for ub in 1024 8; do
  for arm in uva tier; do
    if [ $arm = uva ]; then OT=$OT_UVA; X=; else OT=$OT_TIER; X="GGML_CUDA_MOE_HOT_FILE=$HOT"; fi
    c=4096; ch=4; [ $ub = 8 ] && { c=2048; ch=2; }
    r=$(env $X HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c $c -b $((ub > 8 ? c : 8)) -ub $ub --chunks $ch \
        -sm tensor -ngl 999 -ot "$OT" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-$arm-ub$ub.log |
        grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
    echo "ub $ub $arm: $r"
  done
done
grep -hE "tiered experts" ppl-tier-ub1024.log | head -4

echo "### flashnext (warm): UVA baseline, tiered $TAG"
FN_TAG=-uva bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
FN_ARGS=fn-xl-tiered.args FN_TAG=-$TAG bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
grep -hE "tiered experts" v19dev-$TAG.log | head -4
head -c 400 v19dev-$TAG.answer.txt; echo

echo "### pooled decode: UVA (A) vs tiered (B)"
MP_NO_RESTART=1 MP_ARGS_B=fn-xl-tiered-65k.args bash /opt/llamacpp/tmp-v19/scripts/fleet/multi-prompt.sh fn-xl-65k.args 0,1 v19dev v19dev 2>&1 | grep -E 'MEAN|p[0-9] '

echo "### decode profile, tiered"
DP_ARGS=fn-xl-tiered.args bash decode-profile.sh v19dev 2>&1 | grep -v Killed
echo STEP2_DONE

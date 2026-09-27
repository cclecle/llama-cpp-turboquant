#!/bin/bash
# step21.sh : v19, what a larger hot-expert set would buy (information for the user's VRAM / rung decision; no preset
# is changed). Today: 16,272 of 24,576 experts hot (339 per layer), 36.7 GiB of expert bytes in VRAM over both cards,
# peak 27.8 of 32.6 GB per card at 65k. The MoE is ~16 ms of a 46.7 ms step, bound by the cold experts' PCIe reads.
#   B  = today's set (hot-r9v-uva16-uniform.txt)
#   H1 = 44 GiB of expert bytes (+7.3 GiB, ~+3.65 GiB per card), the same number per layer (uniform)
#   H2 = 44 GiB, the most-routed experts per byte over all layers (greedy)
# Route counts: R9V's catalog (training counts), scored on its held-out counts. Placement changes no math: every text
# must be the reference (8a0ec2a9ccd2816c6250db15e16181eb). Production stays stopped. Throughput B H1 H2 H2 H1 B.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
CAT=/mnt/gguf/r9v/src/packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/wmma-prefill-r2/qualified-128k-r9/catalog.json
T=/opt/llamacpp/tmp-v19/scripts/fleet/moe_hotset.py

echo "### hot sets"
for pol in uniform greedy; do
  /root/venv/bin/python $T --model $M --counts $CAT --eval heldout:$CAT --budget-gib 44 --policy $pol -o hot-44g-$pol.txt 2>&1 | tail -3
  head -2 hot-44g-$pol.txt | tail -1
  sed "s#/mnt/gguf/r9v/bench/hot-r9v-uva16-uniform.txt#/mnt/gguf/r9v/bench/hot-44g-$pol.txt#" fn-xl-tiered-r9vlike.args > fn-xl-tiered-r9vlike-44g-$pol.args
done
/root/venv/bin/python $T --model $M --counts $CAT --eval heldout:$CAT --match-uva 16 --policy uniform -o /tmp/hot-check.txt 2>&1 | tail -2

echo "### throughput, warm, B H1 H2 H2 H1 B"
n=0
for arm in B H1 H2 H2 H1 B; do
  n=$((n + 1))
  case $arm in
    B)  A=fn-xl-tiered-r9vlike.args ;;
    H1) A=fn-xl-tiered-r9vlike-44g-uniform.args ;;
    H2) A=fn-xl-tiered-r9vlike-44g-greedy.args ;;
  esac
  FN_ENV="X=0" FN_ARGS=$A FN_TAG=-s21-$arm-$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:'
done
python3 step_ms.py v19dev-s21-B results.jsonl | grep -v '^arm'
python3 step_ms.py v19dev-s21-H results.jsonl
echo "### generated text per load (md5; reference 8a0ec2a9ccd2816c6250db15e16181eb)"
md5sum v19dev-s21-*.answer.txt
echo STEP21_DONE

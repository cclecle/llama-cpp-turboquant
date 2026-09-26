#!/bin/bash
# series 5: band retune under MTP (v17 vs attn, 6 prompts pooled, short and deep), plus the "abort" line of the
# -sm tensor perplexity runs. Production stopped here, restarted at exit.
cd /root/work-20260925
export MP_NO_RESTART=1
trap 'systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "SERIES5_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
B=/opt/llamacpp/llama-cpp-mine-attn/build3/bin
echo "##### abort line of the -sm tensor perplexity"
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B timeout 900 $B/llama-perplexity -f /root/work-20260926/corpus.txt -c 4096 -b 4096 -ub 1024 --chunks 1 -fa on -ngl 999 \
  -m /mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf -ctk f16 -ctv f16 -sm tensor > /root/work-20260926/ppl-tp.log 2>&1
grep -n -i 'abort' /root/work-20260926/ppl-tp.log | head -5
for chars in 8000 100000; do
  for a in a27 a27q atp afn; do
    D=0; case $a in atp|afn) D=0,1;; esac
    echo "##### $a chars=$chars v17 vs attn"
    MP_CHARS=$chars bash /opt/llamacpp/tmp-attn/scripts/fleet/multi-prompt.sh $a.args $D v17 attn 2>&1 | grep MEAN
    grep -iE "failed|error" /root/work-$(date +%Y%m%d)/mp-attn.log | head -3
  done
done

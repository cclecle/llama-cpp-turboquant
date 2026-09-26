#!/bin/bash
# head-512/576/320 WMMA A/B: v17 vs tmp-rdnab (df0c4ffcb); production stopped, restarted at exit
W=/root/work-20260926; cd $W
trap 'systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "HD512_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
NEW=/opt/llamacpp/tmp-rdnab/build3/bin; OLD=/opt/llamacpp/llama-cpp-mine-v17/build3/bin
(HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$NEW timeout 5400 $NEW/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > hd512-tbo.txt) &
TB=$!
bench() { # tag devices args...
  local tag=$1 dev=$2; shift 2
  for V in v17 new; do
    B=$OLD; [ $V = new ] && B=$NEW
    r=$(HIP_VISIBLE_DEVICES=$dev LD_LIBRARY_PATH=$B timeout 3600 $B/llama-bench "$@" -fa 1 -p 512 -n 0 -d 0,32768 -r 3 2>&1 | grep -E '\| +pp512' | awk -F'|' '{print $(NF-2), $(NF-1)}' | tr -s ' ' | tr '\n' ';')
    echo "$tag $V: $r"
  done
}
bench gemma-4-31B       1 -m /mnt/gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 -t 8
bench gemma-4-26B-A4B   1 -m /mnt/gguf/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 -t 8
bench GLM-4.7-Flash     1 -m /mnt/gguf/GLM-4.7-Flash-UD-Q6_K_XL.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 -t 8
wait $TB
echo "tbo: $(grep -c 'OK$' hd512-tbo.txt) ok, $(grep -c ': FAIL$' hd512-tbo.txt) fail, non-prec=def fail: $(grep ': FAIL$' hd512-tbo.txt | grep -vc 'prec=def')"
grep ': FAIL$' hd512-tbo.txt | grep -v 'prec=def' | head -5 | cut -c1-200
bench gemma-4-31B:HQ     0,1 -m /mnt/gguf/gemma-4-31B-it-Q8_0.gguf -ngl 999 -ctk f16 -ctv f16 -ub 1536 -sm tensor -t 12
bench Mistral-Small-4    0,1 -m /mnt/gguf/Mistral-Small-4-119B-2603-UD-Q6_K-00001-of-00004.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 1536 -sm tensor -ot 'exps=CPU' -t 12

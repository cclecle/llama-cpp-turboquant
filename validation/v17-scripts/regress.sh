#!/bin/bash
# fork-regress on v16 and rdnab3 with the most used Qwen model (dense 27B) and the MoE 35B with CPU experts
cd /root/work-20260925
S=/opt/llamacpp/tmp-rdnab/scripts/fork-regress.sh
for V in v16 rdnab3; do
  B=/opt/llamacpp/llama-cpp-mine-$V/build3
  echo "##### $V 27B"; HIP_VISIBLE_DEVICES=0,1 DEVICES=ROCm0,ROCm1 LD_LIBRARY_PATH=$B/bin timeout 3600 bash $S $B /mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf 2>&1 | grep -E 'PASS|FAIL|ppl|summary|passed|failed'
  echo "##### $V 35B"; HIP_VISIBLE_DEVICES=0,1 DEVICES=ROCm0,ROCm1 NCMOE=8 LD_LIBRARY_PATH=$B/bin timeout 3600 bash $S $B /mnt/gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf 2>&1 | grep -E 'PASS|FAIL|ppl|summary|passed|failed'
done
echo REGRESS_DONE

#!/bin/bash
# step1.sh : v19 step 1 on the dev tree (llama-cpp-mine-v19dev -> /opt/llamacpp/tmp-v19). Production stays stopped.
#   1.1 AllReduce hybrid: perplexity at ub 8 (every reduction is decode-sized) off vs on, flashnext off vs on,
#       pooled 6-prompt decode off vs on.  1.4: flashnext at ub 2048 / 4096.  Then the decode and host profiles.
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
M=/mnt/gguf/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
OT='per_layer_token_embd=CPU,blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)\.ffn_(gate|up|down)_exps\.weight=ROCm0_UVA'
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
sed 's/^262144$/65536/' fn-xl.args > fn-xl-65k.args

echo "### 1.1 correctness: perplexity at ub 8, AllReduce hybrid off vs on (expect bit-identical)"
for E in GGML_CUDA_AR_HYBRID=0 X=0; do
  r=$(env $E HIP_VISIBLE_DEVICES=0,1 timeout 3600 $B/llama-perplexity -m $M -f /root/ppl.txt -c 2048 -b 8 -ub 8 --chunks 2 \
      -sm tensor -ngl 999 -ot "$OT" --n-cpu-moe 0 -fa on -ctk q8_0 -ctv q8_0 -t 12 -fit off 2>&1 | tee ppl-$E.log |
      grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+|AllReduce: [^\n]*' | tr '\n' ' ')
  echo "$E: $r"
done

echo "### 1.1 speed: flashnext, hybrid off vs on"
FN_ENV=GGML_CUDA_AR_HYBRID=0 FN_TAG=-arnccl bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
FN_TAG=-arhyb bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'

echo "### 1.1 pooled decode, hybrid off (A) vs on (B)"
MP_NO_RESTART=1 MP_ENV_A=GGML_CUDA_AR_HYBRID=0 MP_ENV_B=X=0 bash /opt/llamacpp/tmp-v19/scripts/fleet/multi-prompt.sh fn-xl-65k.args 0,1 v19dev v19dev 2>&1 | grep -E 'MEAN|p[0-9] '

echo "### 1.4 prefill ubatch 2048 / 4096 (hybrid on)"
FN_UB=2048 FN_TAG=-ub2048 bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'
FN_UB=4096 FN_TAG=-ub4096 bash flashnext.sh v19dev 2>&1 | grep -v 'changed on disk'

echo "### decode profile (hybrid on)"
bash decode-profile.sh v19dev 2>&1 | grep -v Killed
echo "### host profile (hybrid on)"
bash hostprof.sh v19dev 40 2>&1 | tail -45
echo STEP1_DONE

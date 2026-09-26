#!/bin/bash
# r9 measurement: v17 vs r9 build (d6d8dbfc9, typed MMA K/V store + WMMA head>256 off by default), production stopped
cd /root/work-20260926
S=/opt/llamacpp/llama-cpp-mine-v17/scripts/fleet
trap 'systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "R9_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
ln -sfn /opt/llamacpp/tmp-rdnab /opt/llamacpp/llama-cpp-mine-r9
echo "### correctness"; bash $S/tbo-diff.sh v17 r9 FLASH_ATTN_EXT
Q="-fa 1 -p 4096 -n 0 -d 0,32768 -r 3 -t 8"
echo "### prefill, one GPU"
bash $S/benchab.sh Qwen3.8-27B-q8   0 v17 r9 -- -m /mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 $Q
bash $S/benchab.sh Qwen3.8-27B-f16  0 v17 r9 -- -m /mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf -ngl 999 -ctk f16 -ctv f16 -ub 512 $Q
bash $S/benchab.sh Qwen3.6-35B-A3B  0 v17 r9 -- -m /mnt/gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 -ot 'blk\.(32|34|36|37|38)\.ffn_.*_exps\.weight=CPU' $Q
bash $S/benchab.sh Qwen3.5-9B       0 v17 r9 -- -m $(ls /mnt/gguf/Qwen3.5-9B*.gguf | head -1) -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 $Q
bash $S/benchab.sh gemma-4-31B      0 v17 r9 -- -m /mnt/gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 $Q
bash $S/benchab.sh GLM-4.7-Flash    0 v17 r9 -- -m /mnt/gguf/GLM-4.7-Flash-UD-Q6_K_XL.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 $Q
bash $S/benchab.sh Devstral-24B     0 v17 r9 -- -m /mnt/gguf/Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 512 $Q
echo "### prefill, two GPUs, tensor split"
bash $S/benchab.sh TurboFable-HQ   0,1 v17 r9 -- -m /mnt/gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf -ngl 999 -ctk f16 -ctv f16 -ub 1536 -sm tensor -fa 1 -p 4096 -n 0 -d 0,32768 -r 3 -t 12
bash $S/benchab.sh Mistral-Small-4 0,1 v17 r9 -- -m /mnt/gguf/Mistral-Small-4-119B-2603-UD-Q6_K-00001-of-00004.gguf -ngl 999 -ctk q8_0 -ctv q8_0 -ub 1536 -sm tensor -ot 'exps=CPU' -fa 1 -p 512 -n 0 -d 0,32768 -r 3 -t 12
echo "### r5 f16 band settings (op level, head 256 GQA 6)"
B=/opt/llamacpp/llama-cpp-mine-r9/build3/bin
for E in 'X=0' 'GGML_HIP_FA_BAND_WMMA=2' 'GGML_HIP_FA_BAND_WMMA=2 GGML_HIP_FA_BAND_WMMA_SPLIT=48'; do
  echo "== $E"
  env $E HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B timeout 900 $B/test-backend-ops perf -o FLASH_ATTN_EXT -b ROCm0 -p 'hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=16384' 2>&1 | grep -oE 'nb=[0-9]+,.*type_K=(f16|q8_0)|[0-9.]+ us/run' | paste - - | sed -E 's/nb=([0-9]+),.*type_K=/nb=\1 /'
done

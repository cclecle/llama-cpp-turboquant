#!/bin/bash
# attention memory A/B on the scratch tree "attn" (derived kq mask, native q8_0 prefill, band retune), production stopped.
# D = LLAMA_KQ_MASK_DERIVED, N = GGML_CUDA_FA_KV_NATIVE_PREFILL
cd /root/work-20260926
S=/opt/llamacpp/tmp-attn/scripts/fleet
B=/opt/llamacpp/llama-cpp-mine-attn/build3/bin
M27=/mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf
MG=/mnt/gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
M35=/mnt/gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf
MD=/mnt/gguf/Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf
trap 'systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "ATTN_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
[ -s corpus.txt ] || cat /opt/llamacpp/tmp-attn/docs/*.md /opt/llamacpp/tmp-attn/docs/*/*.md > corpus.txt

# SKIP=tbo skips the kernel tests (already passed)
[ "${SKIP:-}" = tbo ] || [ "${SKIP:-}" = all ] || { echo "### FLASH_ATTN_EXT on ROCm0 (derived cases included), then with native q8_0 prefill"
for E in 'X=0' 'GGML_CUDA_FA_KV_NATIVE_PREFILL=1'; do
  env $E HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B timeout 3600 $B/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 > tbo-attn.log 2>&1
  echo "[$E] $(grep -cE '^ +FLASH_ATTN_EXT.*OK' tbo-attn.log) OK, $(grep -cE '^ +FLASH_ATTN_EXT.*FAIL' tbo-attn.log) FAIL, derived OK: $(grep -E 'derived=' tbo-attn.log | grep -c OK)"
  grep -E '^ +FLASH_ATTN_EXT.*FAIL' tbo-attn.log | head -10
done; }

# load a server, print the compute/KV buffers and VRAM, stop it
load() { # <tag> <env> <server args...>
  local tag=$1 e=$2; shift 2
  env $e HIP_VISIBLE_DEVICES=${DEV:-0} LD_LIBRARY_PATH=$B $B/llama-server "$@" --port 20095 -fit off -lv 4 > load.log 2>&1 &
  local pid=$! t=0
  while [ $t -lt 240 ] && kill -0 $pid 2>/dev/null && ! grep -q 'listening on http' load.log; do sleep 2; t=$((t+2)); done
  local vr=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used' | grep -oE '[0-9]+$' | awk '{printf "%.2f ", $1/1e9}')
  echo "$tag [$e] compute: $(grep -oE 'ROCm[0-9] compute buffer size = +[0-9.]+ MiB' load.log | awk '{print $1, $6}' | sort -u | tr '\n' ' ') host: $(grep -oE '(ROCm_Host|CPU) compute buffer size = +[0-9.]+ MiB' load.log | awk '{print $6}' | sort -u | tr '\n' ' ') vram GB: $vr $(grep -m1 -o 'kq_mask_derived *= *[a-z]*' load.log)"
  grep -iE 'error|abort|failed' load.log | head -3
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}
if [ "${SKIP:-}" != all ]; then
echo "### compute buffers"
A27="-m $M27 -ngl 999 -c 131072 -np 1 -ub 1024 -b 4096 -ctk q8_0 -ctv q8_0 -fa on"
for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1' 'LLAMA_KQ_MASK_DERIVED=1 GGML_CUDA_FA_KV_NATIVE_PREFILL=1'; do load 27B-c131k-ub1024 "$E" $A27; done
AG="-m $MG -ngl 999 -c 131072 -np 1 -ub 1024 -b 4096 -ctk q8_0 -ctv q8_0 -fa on"
for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1'; do load gemma31B-c131k "$E" $AG; done
DEV=0,1 load 27B-TP-c262k 'LLAMA_KQ_MASK_DERIVED=0' -m $M27 -ngl 999 -c 262144 -np 1 -ub 1536 -b 4096 -ctk f16 -ctv f16 -fa on -sm tensor
DEV=0,1 load 27B-TP-c262k 'LLAMA_KQ_MASK_DERIVED=1' -m $M27 -ngl 999 -c 262144 -np 1 -ub 1536 -b 4096 -ctk f16 -ctv f16 -fa on -sm tensor

echo "### perplexity, one sequence per batch (derived must not move it)"
ppl() { # <tag> <env> <args...>
  local tag=$1 e=$2; shift 2
  r=$(env $e HIP_VISIBLE_DEVICES=${DEV:-0} LD_LIBRARY_PATH=$B timeout 1800 $B/llama-perplexity -f corpus.txt -c 4096 -b 4096 -ub 1024 --chunks 4 -fa on -ngl 999 "$@" 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+|abort.*|error.*' | head -2 | tr '\n' ' ')
  echo "$tag [$e]: $r"
}
for m in "27B $M27 -ctk q8_0 -ctv q8_0" "27B-f16 $M27 -ctk f16 -ctv f16" "gemma31B $MG -ctk q8_0 -ctv q8_0" "35B-A3B $M35 -ctk q8_0 -ctv q8_0" "Devstral $MD -ctk q8_0 -ctv q8_0"; do
  set -- $m; t=$1; shift
  for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1'; do ppl $t "$E" -m "$@"; done
done
ppl 27B 'LLAMA_KQ_MASK_DERIVED=1 GGML_CUDA_FA_KV_NATIVE_PREFILL=1' -m $M27 -ctk q8_0 -ctv q8_0
for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1'; do ppl 27B-nckvc2048 "$E" -m $M27 -ctk q8_0 -ctv q8_0 -nckvc 2048; done
for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1'; do DEV=0,1 ppl 27B-TP "$E" -m $M27 -ctk f16 -ctv f16 -sm tensor; done
for E in 'LLAMA_KQ_MASK_DERIVED=0' 'LLAMA_KQ_MASK_DERIVED=1'; do DEV=0,1 ppl gemma31B-TP "$E" -m $MG -ctk q8_0 -ctv q8_0 -sm tensor; done

fi # SKIP=all: prefill A/B only
echo "### prefill, derived and native prefill (attn tree both arms)"
Q="-fa 1 -p 4096 -n 0 -d 0,32768 -r 3 -t 8 -ub 1024"
ENV_A='LLAMA_KQ_MASK_DERIVED=0' ENV_B='LLAMA_KQ_MASK_DERIVED=1' bash $S/benchab.sh 27B-q8 0 attn attn -- -m $M27 -ngl 999 -ctk q8_0 -ctv q8_0 $Q
ENV_A='LLAMA_KQ_MASK_DERIVED=1' ENV_B='LLAMA_KQ_MASK_DERIVED=1 GGML_CUDA_FA_KV_NATIVE_PREFILL=1' bash $S/benchab.sh 27B-q8 0 attn attn -- -m $M27 -ngl 999 -ctk q8_0 -ctv q8_0 $Q
ENV_A='LLAMA_KQ_MASK_DERIVED=0' ENV_B='LLAMA_KQ_MASK_DERIVED=1' bash $S/benchab.sh 27B-f16 0 attn attn -- -m $M27 -ngl 999 -ctk f16 -ctv f16 $Q
ENV_A='LLAMA_KQ_MASK_DERIVED=0' ENV_B='LLAMA_KQ_MASK_DERIVED=1' bash $S/benchab.sh gemma31B 0 attn attn -- -m $MG -ngl 999 -ctk q8_0 -ctv q8_0 $Q
ENV_A='LLAMA_KQ_MASK_DERIVED=0' ENV_B='LLAMA_KQ_MASK_DERIVED=1' bash $S/benchab.sh 35B-A3B 0 attn attn -- -m $M35 -ngl 999 -ctk q8_0 -ctv q8_0 -ot 'blk\.(32|34|36|37|38)\.ffn_.*_exps\.weight=CPU' $Q
ENV_A='LLAMA_KQ_MASK_DERIVED=0' ENV_B='LLAMA_KQ_MASK_DERIVED=1' bash $S/benchab.sh 27B-TP 0,1 attn attn -- -m $M27 -ngl 999 -ctk f16 -ctv f16 -sm tensor -fa 1 -p 4096 -n 0 -d 0,32768 -r 3 -t 12 -ub 1536

[ "${SKIP:-}" = all ] && exit 0
echo "### decode, band retune (v17 vs attn)"
T="-fa 1 -p 0 -n 64 -d 0,16384,65536 -r 3 -t 8"
bash $S/benchab.sh 27B-q8-tg 0 v17 attn -- -m $M27 -ngl 999 -ctk q8_0 -ctv q8_0 $T
bash $S/benchab.sh 27B-f16-tg 0 v17 attn -- -m $M27 -ngl 999 -ctk f16 -ctv f16 $T
bash $S/benchab.sh 35B-A3B-tg 0 v17 attn -- -m $M35 -ngl 999 -ctk q8_0 -ctv q8_0 -ot 'blk\.(32|34|36|37|38)\.ffn_.*_exps\.weight=CPU' $T

#!/bin/bash
# block 13 fused MoE gate+up+GLU MMQ (tests, perplexity, prefill A/B) and the Qwen3.8-27B prefill kernel profile
# (does the gated delta net justify porting block 02?). Scratch tree "attn", production stopped.
cd /root/work-20260926
S=/opt/llamacpp/tmp-attn/scripts/fleet
B=/opt/llamacpp/llama-cpp-mine-attn/build3/bin
M27=/mnt/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf
M35=/mnt/gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf
MAW=/mnt/gguf/Qwen-AgentWorld-35B-A3B-UD-Q5_K_S.gguf
MGLM=/mnt/gguf/GLM-4.7-Flash-UD-Q6_K_XL.gguf
OT35='blk\.(32|34|36|37|38)\.ffn_.*_exps\.weight=CPU'
OFF='GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1'
trap 'systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "MOE_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5

echo "### MUL_MAT_VEC_FUSION / MUL_MAT_ID on ROCm0"
for o in MUL_MAT_VEC_FUSION MUL_MAT_ID; do
  HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B timeout 3600 $B/test-backend-ops -o $o -b ROCm0 > tbo-moe.log 2>&1
  echo "$o: $(grep -cE ' OK$|: OK' tbo-moe.log) OK, $(grep -cE 'FAIL' tbo-moe.log) FAIL, prefill m=512 id cases OK: $(grep -E 'm=512,.*use_id=1' tbo-moe.log | grep -c OK)"
  grep -E 'FAIL' tbo-moe.log | head -5
done

echo "### perplexity, fused off vs on"
ppl() { # <tag> <env> <args...>
  local tag=$1 e=$2; shift 2
  r=$(env $e HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B timeout 1800 $B/llama-perplexity -f corpus.txt -c 4096 -b 4096 -ub 1024 --chunks 4 -fa on -ngl 999 -fit off "$@" 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tr '\n' ' ')
  echo "$tag [$e]: $r"
}
for E in "$OFF" 'X=0'; do ppl 35B-A3B-Q6_K "$E" -m $M35 -ctk q8_0 -ctv q8_0 -ot "$OT35"; done
for E in "$OFF" 'X=0'; do ppl AgentWorld-Q5_K_S "$E" -m $MAW -ctk q8_0 -ctv q8_0; done
for E in "$OFF" 'X=0'; do ppl GLM-4.7-Flash "$E" -m $MGLM -ctk q8_0 -ctv q8_0; done

echo "### MoE prefill, fused off (A) vs on (B)"
Q="-fa 1 -p 512,2048 -n 0 -r 3 -t 8 -ub 1024 -b 4096"
ENV_A="$OFF" ENV_B='X=0' bash $S/benchab.sh 35B-A3B-Q6_K 0 attn attn -- -m $M35 -ngl 999 -ctk q8_0 -ctv q8_0 -ot "$OT35" $Q
ENV_A="$OFF" ENV_B='X=0' bash $S/benchab.sh AgentWorld-Q5_K_S 0 attn attn -- -m $MAW -ngl 999 -ctk q8_0 -ctv q8_0 $Q
ENV_A="$OFF" ENV_B='X=0' bash $S/benchab.sh GLM-4.7-Flash 0 attn attn -- -m $MGLM -ngl 999 -ctk q8_0 -ctv q8_0 $Q
ENV_A="$OFF" ENV_B='X=0' bash $S/benchab.sh 35B-A3B-Q6_K-tg 0 attn attn -- -m $M35 -ngl 999 -ctk q8_0 -ctv q8_0 -ot "$OT35" -fa 1 -p 0 -n 64 -r 3 -t 8

echo "### Qwen3.8-27B prefill kernel profile (rocprofv3), pp4096 at d0 and with a 32k fill"
for d in 0 32768; do
  rm -rf prof27-$d
  HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B timeout 1800 rocprofv3 --kernel-trace --stats -f csv -d prof27-$d -- \
    $B/llama-bench -m $M27 -ngl 999 -ctk q8_0 -ctv q8_0 -fa 1 -p 4096 -n 0 -d $d -r 1 -t 8 -ub 1024 > prof27-$d.log 2>&1
  f=$(find prof27-$d -name '*kernel_stats.csv' | head -1)
  echo "== d=$d ($f)"
  [ -n "$f" ] && python3 - "$f" <<'PY'
import csv, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r['TotalDurationNs']) for r in rows)
agg = {}
for r in rows:
    n = r['Name']
    k = ('gated_delta_net' if 'gated_delta' in n else 'mul_mat_q' if 'mul_mat_q' in n else 'flash_attn' if 'flash_attn' in n
         else 'mul_mat_vec' if 'mul_mat_vec' in n else 'quantize' if 'quantize' in n else 'ssm_conv' if 'conv' in n else re.sub(r'[<(].*', '', n)[:40])
    agg[k] = agg.get(k, 0) + float(r['TotalDurationNs'])
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:12]:
    print('  %-42s %9.1f ms  %5.1f%%' % (k, v / 1e6, 100 * v / tot))
print('  total %.1f ms' % (tot / 1e6))
PY
done

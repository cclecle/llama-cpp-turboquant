#!/bin/bash
# q35-profile.sh <build> [tokens 256] : where the time of ONE Qwen3.5-122B request goes (prefill and decode apart).
# q35.args (the :XS production rung args, one slot, 65k, no host KV cells: the v19 default build segfaults on the
# latter) under rocprofv3 --kernel-trace, a ~32k-token prompt (prompt-32k.txt), greedy, cache_prompt false.
# Prefill window = [end - predicted_ms - prompt_ms, end - predicted_ms] from the server timings, end = last kernel.
# Prints, per phase: wall, per-GPU busy (union of kernel intervals), top kernels. Production must be stopped.
# QP_ENV: env assignments for the server; QP_TAG: output directory suffix; QP_ARGS: other args file.
cd /mnt/gguf/r9v/bench
b=${1:?build}; n=${2:-256}
out=prof-q35-$b${QP_TAG:-}
rm -rf $out; mkdir -p $out
mapfile -t A < ${QP_ARGS:-q35.args}
env ${QP_ENV:-X=0} HIP_VISIBLE_DEVICES=0,1 setsid rocprofv3 --kernel-trace --output-format csv -d $out -o trace -- \
  /opt/llamacpp/llama-cpp-mine-$b/build3/bin/llama-server "${A[@]}" --host 127.0.0.1 --port 8090 > $out/server.log 2>&1 &
SRV=$!
t0=$(date +%s)
until curl -sf -o /dev/null http://127.0.0.1:8090/health; do
  sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $out/server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo timeout; kill -TERM -- -$SRV; exit 1; }
done
echo "loaded in $(( $(date +%s) - t0 )) s; VRAM used per card: $(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 && NR<4 {printf "%.2f GB ", $3/1e9}')"
grep -aE 'tiered experts on|moe cache|failed|error' $out/server.log | head -6
python3 - $n > $out/body.json <<'PY'
import json, sys
text = open('prompt-32k.txt', errors='ignore').read()
print(json.dumps({'messages': [{'role': 'user', 'content': text}], 'max_tokens': int(sys.argv[1]), 'temperature': 0, 'cache_prompt': False}))
PY
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @$out/body.json > $out/answer.json
python3 -c "import json; d=json.load(open('$out/answer.json')); t=d['timings']; print('timings', {k: t.get(k) for k in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second','draft_n','draft_n_accepted')})"
kill -TERM -- -$SRV; sleep 15; kill -KILL -- -$SRV 2>/dev/null; sleep 2
f=$(find $out -name '*kernel_trace.csv' | head -1)
echo "trace: $f"
python3 /mnt/gguf/r9v/bench/q35_phases.py "$f" $out/answer.json

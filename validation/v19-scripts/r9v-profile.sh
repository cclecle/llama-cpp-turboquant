#!/bin/bash
# r9v-profile.sh [prompt file] [tokens, default 512] : rocprofv3 kernel trace of R9V (r9v-run.sh 65536, all its worker
# processes) serving one greedy request, for a per-step comparison with our decode-profile.sh traces
# (step_anatomy.py on both). Production must be stopped.
cd /mnt/gguf/r9v/bench
out=prof-decode-r9v
rm -rf $out; mkdir -p $out
setsid rocprofv3 --kernel-trace --output-format csv -d $out -o trace -- bash ./r9v-run.sh 65536 > $out/server.log 2>&1 &
SRV=$!
t0=$(date +%s)
until curl -sf -o /dev/null http://127.0.0.1:8004/health; do
  sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $out/server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo timeout; exit 1; }
done
echo "r9v up after $(( $(date +%s) - t0 )) s under the profiler"
python3 - "${1:-}" "${2:-512}" > $out/body.json <<'PY'
import json, sys
text = open(sys.argv[1], errors='ignore').read() if sys.argv[1] else 'Write a long, detailed essay about the history of the printing press.'
print(json.dumps({'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': text}], 'max_tokens': int(sys.argv[2]), 'temperature': 0}))
PY
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}' > /dev/null
m0=$(curl -s http://127.0.0.1:8004/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total' | awk '{s[$1]+=$2} END {for (k in s) print k, s[k]}')
t1=$(date +%s.%N)
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' -d @$out/body.json > $out/answer.json
t2=$(date +%s.%N)
m1=$(curl -s http://127.0.0.1:8004/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total' | awk '{s[$1]+=$2} END {for (k in s) print k, s[k]}')
python3 -c "import json; d=json.load(open('$out/answer.json')); print('usage', d.get('usage'))"
echo "request wall $(python3 -c "print(round($t2 - $t1, 2))") s"
echo "spec before: $m0" | tr '\n' ' '; echo; echo "spec after: $m1" | tr '\n' ' '; echo
kill -TERM -- -$SRV; sleep 20; kill -KILL -- -$SRV 2>/dev/null; sleep 5
find $out -name '*kernel_trace.csv' | xargs ls -la

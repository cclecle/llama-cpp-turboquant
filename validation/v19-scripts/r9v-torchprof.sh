#!/bin/bash
# r9v-torchprof.sh [prompt file] [steps, default 24] : R9V's decode steps under vLLM's torch profiler (the tracer R9V
# supports; rocprofv3 loses the worker traces at shutdown). One greedy request on the prompt (1024 tokens); once the
# prefill is done, POST /start_profile records 1 warm-up + N decode steps per worker and writes Chrome traces into
# prof-torch-r9v/. Analyse with torchtrace_anatomy.py. Production must be stopped.
cd /mnt/gguf/r9v/bench
out=$(pwd)/prof-torch-r9v
rm -rf $out; mkdir -p $out
R9V_TORCH_PROFILE_DIR=$out R9V_TORCH_PROFILE_STEPS=${2:-24} setsid bash ./r9v-run.sh 65536 > $out/server.log 2>&1 &
SRV=$!
t0=$(date +%s)
until curl -sf -o /dev/null http://127.0.0.1:8004/health; do
  sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $out/server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo timeout; exit 1; }
done
echo "r9v up after $(( $(date +%s) - t0 )) s"
python3 - "${1:-}" > $out/body.json <<'PY'
import json, sys
text = open(sys.argv[1], errors='ignore').read() if sys.argv[1] else 'Write a long, detailed essay about the history of the printing press.'
print(json.dumps({'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': text}], 'max_tokens': 1024, 'temperature': 0}))
PY
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}' > /dev/null
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' -d @$out/body.json > $out/answer.json &
REQ=$!
sleep 30   # the 32k prefill takes ~19 s warm; profile decode steps only
curl -s -X POST http://127.0.0.1:8004/start_profile; echo " start_profile $(date +%T)"
wait $REQ
curl -s -X POST http://127.0.0.1:8004/stop_profile; echo " stop_profile $(date +%T)"
sleep 30
ls -la $out
api=$(pgrep -f 'vllm.entrypoints.cli.main serve' | head -1)
[ -n "$api" ] && kill -INT $api
for i in $(seq 1 120); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -TERM -- -$SRV 2>/dev/null; sleep 20; kill -KILL -- -$SRV 2>/dev/null
echo R9V_TORCHPROF_DONE

#!/bin/bash
# rungab.sh <args file> <HIP devices> <prompt chars> <release>... : one rung's exact child args (see rungargs.py) on
# several builds, same prompt (/root/ppl.txt slice), greedy, 128 tokens; prints prompt_n and the answer head.
# Production must be stopped by the caller if the GPUs are needed. Server logs: ab-<release>.log in the cwd.
set -u
ARGS=${1:?args file}; DEV=${2:?HIP_VISIBLE_DEVICES}; CH=${3:?prompt chars}; shift 3
PORT=20096
mapfile -t A < "$ARGS"
for V in "$@"; do
  B=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
  HIP_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=$B $B/llama-server "${A[@]}" --host 127.0.0.1 --port $PORT > ab-$V.log 2>&1 &
  P=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  python3 - "$V" "$CH" "$PORT" <<'PY'
import json, re, sys, urllib.request
V, CH, PORT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = open('/root/ppl.txt', errors='ignore').read()[10000:10000 + CH]
b = {'messages': [{'role': 'user', 'content': c + '\n\nSummarise the code above in five bullet points, then write a four-line poem about it.'}],
     'max_tokens': 128, 'temperature': 0}
try:
    j = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions' % PORT,
        data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'}), timeout=3600))
    m = j['choices'][0]['message']; t = (m.get('content') or '') + ' ' + (m.get('reasoning_content') or '')
    tm = j['timings']
    print('%s prompt_n %d pp %.0f t/s tg %.1f t/s | %s' % (V, tm['prompt_n'], tm['prompt_per_second'], tm['predicted_per_second'], re.sub(r'\s+', ' ', t)[:160]))
except Exception as e:
    print(V, 'FAIL', e)
PY
  kill -9 $P 2>/dev/null; sleep 5
done

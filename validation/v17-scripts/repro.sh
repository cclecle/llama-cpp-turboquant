#!/bin/bash
# repro: Mistral-Small-4-119B-A6B:S child args, ~35k-token prompt; BIN dir as $1
cd /root/work-20260925
B=${1:-/opt/llamacpp/tmp-rdnab/build3/bin}
awk '/spawning server instance with name=Mistral-Small-4-119B-A6B:S /{f=1} f&&/with args:/{g=1;next} g{ if ($0 !~ /load: {3,}/) exit; sub(/.*load: +/,""); print }' sweep-v17-dual.router.log | tail -n +2 | sed 's/^54869$/20097/' > ms4s.args
mapfile -t A < ms4s.args
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B $B/llama-server "${A[@]}" > repro.log 2>&1 &
P=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:20097/health >/dev/null 2>&1 && break; sleep 1; done
python3 - <<'PY'
import json,urllib.request
c=open('/root/ppl.txt',errors='ignore').read()[10000:10000+112000]
b={'messages':[{'role':'user','content':c+'\n\nSummarise the code above in five bullet points.'}],'max_tokens':64,'temperature':0}
try:
    j=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:20097/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'}),timeout=3600))
    print('OK prompt_n', j['timings']['prompt_n'], j['choices'][0]['message'].get('content','')[:100])
except Exception as e: print('FAIL', e)
PY
kill -9 $P 2>/dev/null; sleep 3
grep -A8 'no common split state' repro.log | head -12

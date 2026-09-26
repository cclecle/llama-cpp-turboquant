#!/bin/bash
# rungab.sh <rung name> <router log> <chars> <dev> <release>... : run one rung's exact child args on several builds, same prompt, greedy
cd /root/work-20260926
NAME=$1; RLOG=$2; CH=$3; DEV=$4; shift 4
awk -v n="$NAME" 'index($0,"spawning server instance with name=" n " ")>0{f=1} f&&/with args:/{g=1;next} g{ if ($0 !~ /load: {3,}/) exit; sub(/.*load: +/,""); print }' $RLOG | tail -n +2 > ab.args
sed -i '/^--port$/{n;s/.*/20096/}' ab.args
mapfile -t A < ab.args
for V in "$@"; do
  B=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
  HIP_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=$B $B/llama-server "${A[@]}" > ab-$V.log 2>&1 &
  P=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:20096/health >/dev/null 2>&1 && break; sleep 1; done
  python3 - $V $CH <<'PY'
import json,urllib.request,sys,re
V,CH=sys.argv[1],int(sys.argv[2])
c=open('/root/ppl.txt',errors='ignore').read()[10000:10000+CH]
b={'messages':[{'role':'user','content':c+'\n\nSummarise the code above in five bullet points, then write a four-line poem about it.'}],'max_tokens':128,'temperature':0}
try:
    j=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:20096/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'}),timeout=1800))
    m=j['choices'][0]['message']; t=(m.get('content') or '')+' '+(m.get('reasoning_content') or '')
    print(V,'prompt_n',j['timings']['prompt_n'],'|',re.sub(r'\s+',' ',t)[:160])
except Exception as e: print(V,'FAIL',e)
PY
  kill -9 $P 2>/dev/null; sleep 5
done

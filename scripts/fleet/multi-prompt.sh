#!/bin/bash
# multi-prompt.sh <child-args-file> <HIP devices> <release-a> <release-b> : same rung, 6 prompts, two builds.
# A speculative-decoding tg from one greedy prompt is a trajectory sample; pool 6 before reading a delta.
# The args file is the child command line the router prints after "spawning server instance with args:".
set -u; PORT=20098; ARGSFILE=${1:?args file}; DEV=${2:?HIP_VISIBLE_DEVICES}; RA=${3:?release a, e.g. v15}; RB=${4:?release b}
mkdir -p /root/work-$(date +%Y%m%d)
# MP_NO_RESTART=1: the caller already stopped production and restarts it itself (a series of runs)
# MP_ENV_A / MP_ENV_B: extra env assignments for release a / b (e.g. "GGML_HIP_FA_BAND_WMMA=0")
# MP_CHARS: prompt size in corpus characters (default 8000, about 2.5k tokens); raise it to test deep-context decode
# MP_ARGS_B: a second args file for release b (A/B of two configurations on one build)
finish() { [ -n "${MP_NO_RESTART:-}" ] || systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo MP_ALL_DONE; }; trap finish EXIT
[ -n "${MP_NO_RESTART:-}" ] || { systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3; }
mapfile -t ARGS < "$ARGSFILE"
for V in $RA $RB; do
  BIN=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
  if [ "$V" = "$RA" ]; then XENV=${MP_ENV_A:-}; mapfile -t ARGS < "$ARGSFILE"; else XENV=${MP_ENV_B:-}; mapfile -t ARGS < "${MP_ARGS_B:-$ARGSFILE}"; fi
  env $XENV HIP_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=$BIN $BIN/llama-server "${ARGS[@]}" --host 127.0.0.1 --port $PORT > /root/work-$(date +%Y%m%d)/mp-$V.log 2>&1 &
  P=$!
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  python3 - "$V" <<PY
import json,urllib.request,sys
V=sys.argv[1]; corpus=open('/root/ppl.txt',errors='ignore').read(); C=int('${MP_CHARS:-8000}')
tasks=['Summarise the code above in five bullet points, then write a four-line poem about it.',
       'Explain what this code does to a junior developer, step by step.',
       'List every function defined above with a one-line description each.',
       'Rewrite the most complex function above in Python.',
       'What are three possible bugs or risks in this code? Be specific.',
       'Write unit test ideas for the code above.']
tot_t=tot_n=0; tot_acc=tot_dr=0
for k,task in enumerate(tasks):
    off=20000+k*(C+1000)
    body={'messages':[{'role':'user','content':corpus[off:off+C]+'\n\n'+task}],'max_tokens':400,'temperature':0}
    r=urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    j=json.load(urllib.request.urlopen(r,timeout=600)); tm=j['timings']; n=tm['predicted_n']; t=tm['predicted_ms']/1000
    tot_t+=t; tot_n+=n; tot_acc+=tm.get('draft_n_accepted',0) or 0; tot_dr+=tm.get('draft_n',0) or 0
    print('%s p%d  n=%3d  tg %5.1f t/s  acc %3s/%3s' % (V,k,n,tm['predicted_per_second'],tm.get('draft_n_accepted'),tm.get('draft_n')))
print('%s MEAN  tokens=%d  tg %5.1f t/s (pooled)  acceptance %.0f%%' % (V,tot_n,tot_n/tot_t,100*tot_acc/max(1,tot_dr)))
PY
  kill -TERM $P; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; sleep 2
done

#!/bin/bash
# s3-crash-bt.sh [build, default v19dev] : the Qwen3.5-122B :XS segfault (v19 defaults; not with the campaign switches
# off) under gdb. The exact :XS production args (Qwen3.5_XS.args: 4 slots, host KV cells, MTP + ngram drafts), two
# ~32k-token requests from different /root/ppl.txt slices (the crash came at the end of the second prompt), and the
# backtrace of every thread at the fault. S3_ENV: env for the server; S3_TOKENS: tokens per answer (2048, as when it
# crashed: with 512 it did not). Production stays stopped.
cd /mnt/gguf/r9v/bench
b=${1:-v19dev}
B=/opt/llamacpp/llama-cpp-mine-$b/build3/bin
out=crash-$b${S3_TAG:-}; rm -rf $out; mkdir -p $out
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
mapfile -t A < /root/work-20261009-regress/Qwen3.5_XS.args
env ${S3_ENV:-X=0} HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B setsid gdb -batch -nx -ex 'set pagination off' \
  -ex 'handle SIGPIPE nostop noprint' -ex 'handle SIG32 nostop noprint' -ex 'handle SIG33 nostop noprint' -ex run \
  -ex 'echo \n=== FAULT ===\n' -ex bt -ex 'info registers rip' -ex 'thread apply all bt 25' \
  --args $B/llama-server "${A[@]}" --host 127.0.0.1 --port 8091 > $out/gdb.log 2>&1 &
G=$!
until curl -sf -o /dev/null http://127.0.0.1:8091/health; do sleep 3; kill -0 $G 2>/dev/null || { echo "died at load"; tail $out/gdb.log; exit 1; }; done
for R in 1 2; do
  python3 - $R ${S3_TOKENS:-2048} > $out/r$R.json <<'PY'
import json, sys, urllib.request
R = int(sys.argv[1])
c = open('/root/ppl.txt', errors='ignore').read()
off = 20000 + (R - 1)*150000
b = {'messages': [{'role': 'user', 'content': c[off:off + 105000] + '\n\nExplain the code above in great detail, file by file and function by function. Be exhaustive and long.'}],
     'max_tokens': int(sys.argv[2]), 'temperature': 0, 'cache_prompt': True}
try:
    j = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8091/v1/chat/completions',
        data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'}), timeout=7200))
    t = j['timings']; print(json.dumps({k: t.get(k) for k in ('prompt_n', 'prompt_per_second', 'predicted_n', 'predicted_per_second')}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
PY
  echo "request $R: $(cat $out/r$R.json)"
done
sleep 20
if kill -0 $G 2>/dev/null; then
  echo "no crash"; kill -TERM -- -$G 2>/dev/null; sleep 15; kill -KILL -- -$G 2>/dev/null
fi
grep -n -A40 '=== FAULT ===' $out/gdb.log | head -60
echo S3_DONE

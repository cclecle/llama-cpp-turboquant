#!/bin/bash
# hostprof.sh <build> [samples 40] [env assignments...] : where the host thread of a Flash-Next decode step spends
# its time (the GPUs idle ~1/3 of every step). The production :XL rung args (65k ctx), one long greedy decode, and
# gdb stack samples of the server during it (poor man's profiler: the rig has no perf). Prints, over the threads
# that run llama/ggml/common code, the innermost frame of our code per sample. Production must be stopped already.
cd /mnt/gguf/r9v/bench
b=$1; n=${2:-40}; shift 2 2>/dev/null
out=hostprof-$b; rm -rf $out; mkdir -p $out
mapfile -t A < ${DP_ARGS:-fn-xl.args}   # DP_ARGS: another args file (same layout)
for i in "${!A[@]}"; do [ "${A[$i]}" = --ctx-size ] && A[$((i + 1))]=65536; done
env "$@" HIP_VISIBLE_DEVICES=0,1 setsid /opt/llamacpp/llama-cpp-mine-$b/build3/bin/llama-server "${A[@]}" \
  --host 127.0.0.1 --port 8090 > $out/server.log 2>&1 &
SRV=$!
until curl -sf -o /dev/null http://127.0.0.1:8090/health; do sleep 2; kill -0 $SRV 2>/dev/null || { tail $out/server.log; exit 1; }; done
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a very long, detailed essay about the history of the printing press, at least 3000 words."}],"max_tokens":1500,"temperature":0}' > $out/answer.json &
REQ=$!
sleep 12   # past the prompt, into steady decode
for i in $(seq 1 $n); do
  kill -0 $REQ 2>/dev/null || break
  timeout 20 gdb -p $SRV -batch -nx -ex 'set pagination off' -ex 'thread apply all bt 30' > $out/s$i.txt 2>/dev/null
done
wait $REQ
python3 -c "import json; d=json.load(open('$out/answer.json')); print('timings', {k: d['timings'].get(k) for k in ('predicted_n','predicted_per_second','draft_n','draft_n_accepted')})"
kill -TERM -- -$SRV; sleep 12; kill -KILL -- -$SRV 2>/dev/null
python3 - $out <<'PY'
import collections, glob, re, sys
ours = re.compile(r'(llama|ggml|common_|server|speculative|sampl|gguf)', re.I)
top = collections.Counter(); stacks = collections.Counter(); nsamp = 0
for f in sorted(glob.glob(sys.argv[1] + '/s*.txt')):
    nsamp += 1
    for th in open(f, errors='ignore').read().split('\nThread ')[1:]:
        frames = re.findall(r'^#\d+\s+(?:0x[0-9a-f]+ in )?([^\s(]+)', th, re.M)
        if not any(ours.search(fr) for fr in frames):
            continue
        # innermost frame + the innermost frame of our own code
        mine = next((fr for fr in frames if ours.search(fr)), '?')
        key = '%s  <-  %s' % (frames[0][:50], mine[:70])
        top[key] += 1
        stacks[' < '.join(fr[:40] for fr in frames[:6])] += 1
print('%d samples' % nsamp)
for k, v in top.most_common(25):
    print('%5d  %s' % (v, k))
print('--- top stacks')
for k, v in stacks.most_common(12):
    print('%5d  %s' % (v, k))
PY

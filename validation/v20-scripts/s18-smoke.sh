#!/bin/bash
# s18-smoke.sh [build, default v23] [args file, default fn-xl-tiered-r9vlike.args] : quick smoke test of a release next
# to an IDLE production (nothing loaded: check the VRAM first), on port 8093, production untouched. Load time, which
# libamdhip64 / rocblas the process mapped (ROCm 10.1 build: must be /opt/rocm-10.1), then two temperature-0 requests
# (a short question, ~1.5k-token prompt) with the server's timings, the text head and its md5.
cd /mnt/gguf/r9v/bench
b=${1:-v23}; A_FILE=${2:-fn-xl-tiered-r9vlike.args}
B=/opt/llamacpp/llama-cpp-mine-$b/build3/bin
mapfile -t A < $A_FILE
HIP_VISIBLE_DEVICES=0,1 setsid $B/llama-server "${A[@]}" --ctx-size 65536 --host 127.0.0.1 --port 8093 > smoke-$b.log 2>&1 &
SRV=$!
t0=$(date +%s)
until curl -sf -o /dev/null http://127.0.0.1:8093/health; do
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -30 smoke-$b.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "no /health after 600 s"; break; }
  sleep 1
done
echo "loaded in $(( $(date +%s) - t0 )) s"
P=$(pgrep -n -P $SRV llama-server || echo $SRV)
grep -oE '/[^ ]*(libamdhip64|librocblas|librccl|libhsa-runtime64)[^ ]*' /proc/$P/maps /proc/$SRV/maps 2>/dev/null | cut -d: -f2 | sort -u
for c in 0000:03:00.0 0000:07:00.0; do echo "vram $c: $(( $(cat /sys/bus/pci/devices/$c/mem_info_vram_used) >> 20 )) MiB"; done
python3 - <<'PY'
import hashlib, json, urllib.request
long = ' '.join('Section %d: the river carries silt from the mountains to the delta, where farmers plant rice.' % i for i in range(80))
for name, prompt, n in (('short', 'Explain in a few sentences why the sky is blue.', 256),
                        ('1.5k', long + '\n\nSummarise the text above in three sentences.', 256)):
    req = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': n, 'temperature': 0, 'cache_prompt': False}
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8093/v1/chat/completions',
        json.dumps(req).encode(), {'Content-Type': 'application/json'}), timeout=600))
    m = r['choices'][0]['message']
    text = (m.get('reasoning_content') or '') + '|' + (m.get('content') or '')
    t = r.get('timings', {})
    print('%-5s prompt %5d tok %7.1f t/s | gen %4d tok %6.1f t/s | md5 %s' % (name, t.get('prompt_n', 0),
          t.get('prompt_per_second', 0), t.get('predicted_n', 0), t.get('predicted_per_second', 0),
          hashlib.md5(text.encode()).hexdigest()[:8]))
    print('      ' + text.replace('\n', ' ')[:300])
PY
grep -ciE 'error|abort|assert' smoke-$b.log | sed 's/^/error-ish log lines: /'
kill -TERM -- -$SRV; sleep 10; kill -KILL -- -$SRV 2>/dev/null
echo S18_DONE

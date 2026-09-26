#!/bin/bash
# decode-profile.sh [build, default v18] : where one Flash-Next decode step goes. The production :XL rung args (65k ctx)
# under rocprofv3 --kernel-trace, one short prompt, 512 generated tokens (MTP on, as in production), then per-kernel
# totals and, for the MoE mat-vec kernels, the per-dispatch duration split (host-resident UVA experts read over PCIe
# are an order of magnitude slower than VRAM ones). Production must already be stopped; it is left stopped.
cd /mnt/gguf/r9v/bench
b=${1:-v18}
out=prof-decode-$b
rm -rf $out; mkdir -p $out
mapfile -t A < ${DP_ARGS:-fn-xl.args}   # DP_ARGS: another args file (same layout)
for i in "${!A[@]}"; do [ "${A[$i]}" = --ctx-size ] && A[$((i + 1))]=65536; done
HIP_VISIBLE_DEVICES=0,1 setsid rocprofv3 --kernel-trace --output-format csv -d $out -o trace -- \
  /opt/llamacpp/llama-cpp-mine-$b/build3/bin/llama-server "${A[@]}" --host 127.0.0.1 --port 8090 > $out/server.log 2>&1 &
SRV=$!
t0=$(date +%s)
until curl -sf -o /dev/null http://127.0.0.1:8090/health; do
  sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $out/server.log; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo timeout; exit 1; }
done
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a long, detailed essay about the history of the printing press."}],"max_tokens":512,"temperature":0}' > $out/answer.json
python3 -c "import json; d=json.load(open('$out/answer.json')); print('timings', {k: d['timings'].get(k) for k in ('prompt_n','predicted_n','predicted_per_second','draft_n','draft_n_accepted')})"
kill -TERM -- -$SRV; sleep 15; kill -KILL -- -$SRV 2>/dev/null; sleep 2
f=$(find $out -name '*kernel_trace.csv' | head -1)
echo "trace: $f"
python3 - "$f" <<'PY'
import csv, re, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
# decode window: from the first MoE mat-vec dispatch to the end (the short prompt's prefill is negligible)
rows.sort(key=lambda r: int(r['Start_Timestamp']))
first = next(i for i, r in enumerate(rows) if 'mul_mat_vec_q' in r['Kernel_Name'])
rows = rows[first:]
agg = collections.defaultdict(float); cnt = collections.Counter()
moe = []
for r in rows:
    n = r['Kernel_Name']; d = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e6
    k = re.sub(r'[<(].*', '', n).replace('void ', '')[:48]
    if 'mul_mat_vec_q_moe' in n: k = 'mul_mat_vec_q_moe'
    agg[k] += d; cnt[k] += 1
    if 'mul_mat_vec_q_moe' in n or ('mul_mat_vec_q' in n and 'moe' not in n): moe.append((k, d, r.get('Agent_Id', '')))
tot = sum(agg.values())
print('decode-window GPU kernel time %.1f ms over %d dispatches (both GPUs summed)' % (tot, len(rows)))
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:18]:
    print('  %-48s %9.1f ms %5.1f%%  n=%d  avg %.1f us' % (k, v, 100 * v / tot, cnt[k], 1000 * v / cnt[k]))
d = sorted(x[1] for x in moe if x[0] == 'mul_mat_vec_q_moe')
if d:
    import statistics
    print('mul_mat_vec_q_moe dispatch us: p10 %.0f p50 %.0f p90 %.0f p99 %.0f' % tuple(1000 * d[int(len(d) * q)] for q in (0.1, 0.5, 0.9, 0.99)))
    for thr in (100, 200, 400):
        slow = [x for x in d if 1000 * x > thr]
        print('  > %d us: %d dispatches, %.1f ms (%.1f%% of all kernel time)' % (thr, len(slow), sum(slow), 100 * sum(slow) / tot))
PY

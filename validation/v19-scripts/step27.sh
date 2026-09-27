#!/bin/bash
# step27.sh : v19, why the prefill staging of step 26 lost (1,030 vs 1,152 t/s): a kernel + memory-copy trace of the
# 32k prefill with the staging on and off. Per GPU: the expert matmul time, the DMA time and rate of the staging
# copies, and the time the matmuls wait. Production stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
for arm in on off; do
  E="X=0"; [ $arm = off ] && E="GGML_CUDA_MOE_STAGE=0"
  DP_ENV="$E" DP_EXTRA=--memory-copy-trace DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=16 DP_TAG=-s27-$arm \
    bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
  d=prof-decode-v19dev-s27-$arm
  python3 - $d <<'PY'
import csv, glob, sys, collections
d = sys.argv[1]
k = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
m = glob.glob(d + '/**/*memory_copy_trace.csv', recursive=True)
rows = [r for r in csv.DictReader(open(k))]
last = max(int(r['End_Timestamp']) for r in rows if 'mul_mat_q<' in r['Kernel_Name'])
first = min(int(r['Start_Timestamp']) for r in rows if 'mul_mat_q<' in r['Kernel_Name'])
for a in sorted({r['Agent_Id'] for r in rows}):
    rs = [r for r in rows if r['Agent_Id'] == a and first <= int(r['Start_Timestamp']) <= last]
    moe = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in rs if 'mul_mat_q<' in r['Kernel_Name'] and 'Grid_Size_Z' in r and int(r['Grid_Size_Z']) > 1)
    busy = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in rs)
    print(f'{a}: prefill window {(last-first)/1e9:.2f} s, busy {busy/1e9:.2f} s, expert mmq {moe/1e9:.2f} s')
if m:
    cp = [r for r in csv.DictReader(open(m[0]))]
    print('copy trace columns:', list(cp[0].keys())[:12] if cp else 'none')
    by = collections.defaultdict(lambda: [0, 0, 0])
    for r in cp:
        s0, e0 = int(r['Start_Timestamp']), int(r['End_Timestamp'])
        if not (first <= s0 <= last): continue
        key = (r.get('Direction', r.get('Kind', '?')), r.get('Dst_Agent_Id', r.get('Agent_Id', '?')))
        by[key][0] += 1; by[key][1] += e0 - s0; by[key][2] += int(r.get('Size', r.get('Bytes', 0)) or 0)
    for key, (n, t, b) in sorted(by.items(), key=lambda kv: -kv[1][1])[:6]:
        print(f'  copies {key}: {n} x, {t/1e9:.2f} s, {b/1e9:.1f} GB, {b/max(t,1):.1f} GB/s')
PY
done
echo STEP27_DONE

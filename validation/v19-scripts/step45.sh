#!/bin/bash
# step45.sh : v19 prefill, where a 32k prefill at ubatch 4096 goes now (after steps 29-36): a kernel trace of the
# prompt, per GPU the busy/idle split and the kernels with the most time (stage_trace.py --kernels). Production
# stays stopped.
cd /mnt/gguf/r9v/bench
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
# the same-config args at ubatch 4096 (batch raised to it)
python3 - <<'PY'
a = open('fn-xl-tiered-r9vlike.args').read().split('\n')
for i, x in enumerate(a[:-1]):
    if x in ('--ubatch-size', '--batch-size') and int(a[i + 1]) < 4096:
        a[i + 1] = '4096'
open('fn-xl-tiered-r9vlike-ub4096.args', 'w').write('\n'.join(a))
PY
DP_ARGS=fn-xl-tiered-r9vlike-ub4096.args DP_PROMPT=prompt-32k.txt DP_TOKENS=16 DP_TAG=-s45 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s45
f=$(find $d -name '*kernel_trace.csv' | head -1)
python3 - $f <<'PY'
import csv, sys, collections, re
rows = list(csv.DictReader(open(sys.argv[1])))
mm = [r for r in rows if 'mul_mat_q<' in r['Kernel_Name']]
first = min(int(r['Start_Timestamp']) for r in mm); last = max(int(r['End_Timestamp']) for r in mm)
for a in sorted({r['Agent_Id'] for r in rows}):
    by = collections.Counter(); n = collections.Counter(); busy = 0
    for r in rows:
        s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
        if r['Agent_Id'] != a or not (first <= s <= last): continue
        k = re.sub(r'<.*', '', re.sub(r'\(.*', '', r['Kernel_Name']))[:48]
        if 'mul_mat_q' in k and int(r.get('Grid_Size_Z', 1)) > 1: k = 'mul_mat_q (MoE experts)'
        by[k] += e - s; n[k] += 1; busy += e - s
    print(f'{a}: prefill window {(last-first)/1e9:.2f} s, busy {busy/1e9:.2f} s')
    for k, t in by.most_common(22):
        print(f'  {t/1e9:7.3f} s {n[k]:7d} x  {k}')
PY
echo STEP45_DONE

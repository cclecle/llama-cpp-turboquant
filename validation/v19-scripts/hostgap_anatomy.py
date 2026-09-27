#!/usr/bin/env python3
# hostgap_anatomy.py <rocprofv3 output dir> <steps> [min_gap_us 20] : what the host does while a GPU idles in decode.
# Needs a trace with --kernel-trace and --hip-runtime-trace (decode-profile.sh DP_HIP=1). The decode window starts
# after the last prefill mat-mul (mul_mat_q; verify batches of <= 8 tokens use the mat-vec kernels). For every idle
# interval of the first GPU longer than min_gap_us, the host HIP calls that overlap it are charged (by overlap);
# idle time that no HIP call covers is host code between calls (llama/ggml/server). Per speculative step:
#   - GPU idle, split by the HIP function running on the host during it, or "(host code)"
#   - the host's time in each HIP function over the whole window (sync waits show how long the host waited)
import collections
import csv
import glob
import sys

d, steps = sys.argv[1], int(sys.argv[2])
min_gap = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 20e3

kf = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
hf = glob.glob(d + '/**/*hip_api_trace.csv', recursive=True)[0]

kern = collections.defaultdict(list)
last_mmq = 0
for r in csv.DictReader(open(kf)):
    s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
    kern[r['Agent_Id']].append((s, e))
    if 'mul_mat_q<' in r['Kernel_Name'] or 'mul_mat_q_' in r['Kernel_Name']:
        last_mmq = max(last_mmq, e)
agent = sorted(kern)[0]
ks = sorted(k for k in kern[agent] if k[0] > last_mmq)
t0, t1 = ks[0][0], ks[-1][1]

api = []
for r in csv.DictReader(open(hf)):
    s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
    if e < t0 or s > t1:
        continue
    api.append((s, e, r['Function'], r['Thread_Id']))
api.sort()
# the thread issuing most launches is the decode thread
launches = collections.Counter(a[3] for a in api if 'Launch' in a[2] or 'Graph' in a[2])
main = launches.most_common(1)[0][0] if launches else None
mapi = [a for a in api if a[3] == main]

# GPU idle intervals of the first agent
gaps = []
end = ks[0][1]
for s, e in ks[1:]:
    if s - end > min_gap:
        gaps.append((end, s))
    end = max(end, e)

charged = collections.Counter()
j = 0
for gs, ge in gaps:
    covered = 0
    while j < len(mapi) and mapi[j][1] < gs:
        j += 1
    k = j
    while k < len(mapi) and mapi[k][0] < ge:
        s, e, fn, _ = mapi[k]
        ov = min(e, ge) - max(s, gs)
        if ov > 0:
            charged[fn] += ov
            covered += ov
        k += 1
    charged['(host code)'] += max(0, (ge - gs) - covered)

tot = collections.Counter()
cnt = collections.Counter()
for s, e, fn, _ in mapi:
    tot[fn] += e - s
    cnt[fn] += 1

wall = (t1 - t0) / 1e6 / steps
idle = sum(ge - gs for gs, ge in gaps) / 1e6 / steps
print(f'{kf.split("/")[-1]}: decode window {steps} steps, {wall:.2f} ms/step, GPU {agent} idle in gaps > {min_gap/1e3:.0f} us:'
      f' {idle:.2f} ms/step in {len(gaps)/steps:.1f} gaps; decode thread {main}')
print('--- GPU idle, by what the decode thread was doing (ms/step)')
for fn, v in charged.most_common(14):
    print(f'{v / 1e6 / steps:8.3f}  {fn}')
print('--- decode thread time in HIP calls over the window (ms/step, calls/step)')
for fn, v in tot.most_common(14):
    print(f'{v / 1e6 / steps:8.3f} {cnt[fn] / steps:8.1f}  {fn}')

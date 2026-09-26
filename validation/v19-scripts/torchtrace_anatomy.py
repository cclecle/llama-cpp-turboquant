#!/usr/bin/env python3
# torchtrace_anatomy.py : per decode step, the same numbers for both engines.
#   torchtrace_anatomy.py torch <trace.json[.gz]> [steps]      vLLM torch-profiler trace of one worker (one GPU)
#   torchtrace_anatomy.py rocprof <kernel_trace.csv> <steps>    our rocprofv3 trace; the decode window is everything
#                                                                after the last long prefill evaluation (first GPU)
# Prints: kernels per step, GPU busy per step (union of kernel intervals), wall per step, idle per step, and the top
# kernels by time per step with counts.
import collections, csv, gzip, json, re, sys


def short(name):
    n = re.sub(r'\(.*', '', name)
    n = re.sub(r'<.*', '', n).replace('void ', '').strip()
    return n[-48:]


def report(kernels, steps, label):
    kernels.sort()
    t0, t1 = kernels[0][0], max(e for _, e, _ in kernels)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e, _ in kernels:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = t1 - t0
    print('%s: %d steps, per step: %.0f kernels, wall %.2f ms, GPU busy %.2f ms, idle %.2f ms' % (
        label, steps, len(kernels) / steps, wall / steps / 1e3, busy / steps / 1e3, (wall - busy) / steps / 1e3))
    cnt, tim = collections.Counter(), collections.Counter()
    for s, e, n in kernels:
        cnt[short(n)] += 1
        tim[short(n)] += e - s
    for k, v in tim.most_common(28):
        print('   %-48s %7.1f x  %7.2f ms/step  avg %6.1f us' % (k, cnt[k] / steps, v / steps / 1e3, v / cnt[k]))


mode, path = sys.argv[1], sys.argv[2]
if mode == 'torch':
    op = gzip.open if path.endswith('.gz') else open
    d = json.load(op(path, 'rt'))
    ev = d['traceEvents'] if isinstance(d, dict) else d
    k = [(float(e['ts']), float(e['ts']) + float(e.get('dur', 0)), e.get('name', '?'))
         for e in ev if e.get('ph') == 'X' and e.get('cat', '').lower() in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, sum(1 for e in ev if str(e.get('name', '')).startswith('ProfilerStep')))
    report(k, steps, 'torch trace %s' % path.split('/')[-1])
else:
    steps = int(sys.argv[3])
    by = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r.get('Agent_Id', '')].append((int(r['Start_Timestamp']) / 1e3, int(r['End_Timestamp']) / 1e3, r['Kernel_Name']))
    agent, ev = sorted(by.items())[0]
    ev.sort()
    # evaluations separated by > 150 us idle; the decode window starts after the last one above 4500 dispatches
    segs, cur, last = [], [], None
    for s, e, n in ev:
        if last is not None and s - last > 150:
            segs.append(cur); cur = []
        cur.append((s, e, n)); last = e if last is None else max(last, e)
    segs.append(cur)
    idx = max(i for i, sg in enumerate(segs) if len(sg) > 4500)
    dec = [x for sg in segs[idx + 1:] for x in sg]
    report(dec, steps, 'rocprof %s %s decode window' % (path.split('/')[-1], agent))

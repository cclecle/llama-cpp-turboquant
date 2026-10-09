#!/usr/bin/env python3
# q35_phases.py <kernel_trace.csv> <answer.json> : split a one-request rocprofv3 kernel trace into prefill and decode
# (from the server timings), print per phase the wall time, per-GPU busy (union of kernel intervals) and idle share,
# launches, and the top kernels by summed time.
import collections, csv, json, re, sys

rows = list(csv.DictReader(open(sys.argv[1])))
t = json.load(open(sys.argv[2]))['timings']
ev = [(int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Agent_Id'], r['Kernel_Name']) for r in rows]
ev.sort()
end = max(e[1] for e in ev)
dec_ms, pre_ms = t['predicted_ms'], t['prompt_ms']
cut = end - int(dec_ms * 1e6)
start = cut - int(pre_ms * 1e6)
phases = {'prefill': (start, cut), 'decode': (cut, end)}


def union(iv):
    iv.sort()
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def short(n):
    n = n.replace('void ', '')
    return re.sub(r'[<(].*', '', n)[:52] + ('<' + re.search(r'<([^>]*)', n).group(1)[:30] + '>' if re.search(r'<([^>]*)', n) else '')


for name, (a, b) in phases.items():
    sel = [e for e in ev if e[0] >= a and e[1] <= b]
    wall = (b - a) / 1e6
    print('\n== %s: wall %.0f ms, %d kernels' % (name, wall, len(sel)))
    agents = sorted({e[2] for e in sel})
    for ag in agents:
        iv = [(e[0], e[1]) for e in sel if e[2] == ag]
        busy = union(iv) / 1e6
        print('   agent %s: busy %.0f ms = %.1f%% of wall, %d launches' % (ag, busy, 100 * busy / wall, len(iv)))
    agg, cnt = collections.defaultdict(float), collections.Counter()
    for s, e, ag, n in sel:
        k = short(n)
        agg[k] += (e - s) / 1e6
        cnt[k] += 1
    tot = sum(agg.values())
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:16]:
        print('   %-70s %9.1f ms %5.1f%% n=%d avg %.1f us' % (k, v, 100 * v / tot, cnt[k], 1000 * v / cnt[k]))
    steps = t['predicted_n'] if name == 'decode' else None
    if steps:
        print('   per generated token: %.1f ms wall (%.1f t/s), %.0f kernels/token (both GPUs)' % (wall / steps, 1000 * steps / wall, len(sel) / steps))

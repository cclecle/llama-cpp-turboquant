#!/usr/bin/env python3
# step_anatomy.py <rocprofv3 kernel_trace.csv> [gap us, default 150] : split a trace into graph evaluations per GPU
# (dispatch runs separated by an idle gap longer than the threshold) and classify them by size. Works on llama.cpp
# and on vLLM (R9V) traces. The "verify" evaluations are the most frequent size above 300 dispatches (the target
# decode pass). For those: wall, busy, a histogram of the idle gaps between consecutive dispatches, and the kernels
# that make up one evaluation. Everything is per GPU (agent).
import collections, csv, statistics, sys

GAP_US = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0

by_agent = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        by_agent[r.get('Agent_Id', '')].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))


def short(name):
    return name.split('(')[0].split('<')[0].replace('void ', '').strip()[-44:]


for agent, ev in sorted(by_agent.items()):
    ev.sort()
    segs, cur, last_end = [], [], None
    for s, e, n in ev:
        if last_end is not None and (s - last_end) / 1e3 > GAP_US:
            segs.append(cur)
            cur = []
        cur.append((s, e, n))
        last_end = e if last_end is None else max(last_end, e)
    segs.append(cur)
    info = [(len(sg), (max(x[1] for x in sg) - sg[0][0]) / 1e6, sum(x[1] - x[0] for x in sg) / 1e6, sg) for sg in segs]
    cls = collections.defaultdict(list)
    for n, w, b, sg in info:
        cls[n // 100 * 100].append((n, w, b))
    print('%s: %d evaluations, %d dispatches' % (agent, len(info), len(ev)))
    for k in sorted(cls):
        v = cls[k]
        if len(v) < 3:
            continue
        print('  %5d-%5d dispatches: %4d evals, median %5d disp, median wall %8.3f ms, median busy %8.3f ms' % (
            k, k + 99, len(v), statistics.median(x[0] for x in v), statistics.median(x[1] for x in v),
            statistics.median(x[2] for x in v)))
    big_sizes = collections.Counter(n // 100 * 100 for n, _, _, _ in info if n > 300)
    if not big_sizes:
        continue
    mode = big_sizes.most_common(1)[0][0]
    big = [x for x in info if mode <= x[0] < mode + 100]
    nb = len(big)
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e12)]
    hist = [[0, 0.0] for _ in bins]
    cnt, tim = collections.Counter(), collections.Counter()
    for _, _, _, sg in big:
        end = sg[0][1]
        for i, (s0, e0, nm) in enumerate(sg):
            cnt[short(nm)] += 1
            tim[short(nm)] += (e0 - s0) / 1e3
            if i == 0:
                continue
            g = max(0, s0 - end) / 1e3
            for j, (lo, hi) in enumerate(bins):
                if lo <= g < hi:
                    hist[j][0] += 1
                    hist[j][1] += g
                    break
            end = max(end, e0)
    print('  verify evaluations (%d-%d dispatches, %d of them): median wall %.3f ms, busy %.3f ms; gaps per evaluation:' % (
        mode, mode + 99, nb, statistics.median(x[1] for x in big), statistics.median(x[2] for x in big)))
    for (lo, hi), (c, t) in zip(bins, hist):
        if c:
            print('    gap %4d-%-6s us: %6.0f gaps, %6.2f ms' % (lo, str(hi) if hi < 1e12 else 'inf', c / nb, t / 1e3 / nb))
    print('  kernels per verify evaluation (top 26):')
    for k, v in cnt.most_common(26):
        print('    %-44s %6.1f x  avg %7.1f us  = %6.2f ms' % (k, v / nb, tim[k] / v, tim[k] / 1e3 / nb))

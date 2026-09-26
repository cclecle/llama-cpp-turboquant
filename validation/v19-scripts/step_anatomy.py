#!/usr/bin/env python3
# step_anatomy.py <rocprofv3 kernel_trace.csv> [gap us, default 150] : split a decode trace into graph evaluations
# per GPU (dispatch runs separated by an idle gap) and classify them by size: the target verify pass (all layers,
# thousands of dispatches) vs the MTP draft passes (one layer, a few dozen). Per class: count, dispatches, wall
# and busy time; then, inside the target passes, a histogram of the idle gaps between consecutive dispatches and
# the kernels that start after a long gap (to tell subgraph boundaries from per-kernel dispatch overhead).
import collections, csv, statistics, sys

GAP_US = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0

by_agent = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        by_agent[r.get('Agent_Id', '')].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))


def short(name):
    return name.split('(')[0].split('<')[0].replace('void ', '')[-38:]


for agent, ev in sorted(by_agent.items()):
    ev.sort()
    ev = ev[next(i for i, x in enumerate(ev) if 'mul_mat_vec' in x[2]):]
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
        cls[n // 200 * 200].append((n, w, b))
    print('%s: %d evaluations' % (agent, len(info)))
    for k in sorted(cls):
        v = cls[k]
        print('  %5d-%5d dispatches: %4d evals, median %5d disp, median wall %7.3f ms, median busy %7.3f ms' % (
            k, k + 199, len(v), statistics.median(x[0] for x in v), statistics.median(x[1] for x in v),
            statistics.median(x[2] for x in v)))
    big = [x for x in info if x[0] >= 2000]
    if not big:
        continue
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e12)]
    hist = [[0, 0.0] for _ in bins]
    after_long = collections.Counter()
    for _, _, _, sg in big:
        end = sg[0][1]
        for s0, e0, nm in sg[1:]:
            g = max(0, s0 - end) / 1e3   # us
            for j, (lo, hi) in enumerate(bins):
                if lo <= g < hi:
                    hist[j][0] += 1
                    hist[j][1] += g
                    if lo >= 20:
                        after_long[short(nm)] += 1
                    break
            end = max(end, e0)
    nb = len(big)
    print('  inside the %d target evaluations, idle between dispatches, per evaluation:' % nb)
    for (lo, hi), (c, t) in zip(bins, hist):
        print('    gap %4d-%-6s us: %6.0f gaps, %6.2f ms' % (lo, str(hi) if hi < 1e12 else 'inf', c / nb, t / 1e3 / nb))
    print('    kernels starting after a gap >= 20 us (per evaluation): ' +
          ', '.join('%s x%.0f' % (k, v / nb) for k, v in after_long.most_common(8)))

# which kernels make up one target evaluation (per GPU, first agent only)
for agent, ev in sorted(by_agent.items())[:1]:
    ev.sort()
    ev = ev[next(i for i, x in enumerate(ev) if 'mul_mat_vec' in x[2]):]
    segs, cur, last_end = [], [], None
    for s, e, n in ev:
        if last_end is not None and (s - last_end) / 1e3 > GAP_US:
            segs.append(cur)
            cur = []
        cur.append((s, e, n))
        last_end = e if last_end is None else max(last_end, e)
    segs.append(cur)
    big = [sg for sg in segs if len(sg) >= 2000]
    cnt = collections.Counter()
    tim = collections.Counter()
    for sg in big:
        for s0, e0, nm in sg:
            cnt[short(nm)] += 1
            tim[short(nm)] += (e0 - s0) / 1e3
    nb = len(big)
    print('%s: dispatches per target evaluation (%d evaluations), top 24' % (agent, nb))
    for k, v in cnt.most_common(24):
        print('  %-38s %6.0f   avg %6.1f us' % (k, v / nb, tim[k] / v))

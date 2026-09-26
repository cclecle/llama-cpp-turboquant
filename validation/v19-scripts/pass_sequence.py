#!/usr/bin/env python3
# pass_sequence.py <kernel_trace.csv> [n] : the kernel sequence of one steady-state target verify pass (first GPU),
# run-length collapsed, first n entries (default 260): the per-layer op pattern, to pick fusion targets.
import collections, csv, sys
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 260
ev = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        ev[r.get('Agent_Id', '')].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))
a, x = sorted(ev.items())[0]
x.sort()
segs, cur, last = [], [], None
for s, e, nm in x:
    if last is not None and (s - last) > 150e3:
        segs.append(cur); cur = []
    cur.append((s, e, nm)); last = e if last is None else max(last, e)
segs.append(cur)
big = [sg for sg in segs if 4000 <= len(sg) <= 4200]
sg = big[len(big) // 2]
short = lambda nm: nm.split('(')[0].split('<')[0].replace('void ', '')[-34:]
out, prev, rep = [], None, 0
for s, e, nm in sg:
    k = short(nm)
    if k == prev:
        rep += 1
    else:
        if prev is not None:
            out.append('%s%s' % (prev, ' x%d' % rep if rep > 1 else ''))
        prev, rep = k, 1
out.append('%s%s' % (prev, ' x%d' % rep if rep > 1 else ''))
print('%d dispatches, %d runs; first %d:' % (len(sg), len(out), n_show))
print('\n'.join(out[:n_show]))

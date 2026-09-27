#!/usr/bin/env python3
# stage_trace.py : compare two rocprofv3 prefill traces (kernel + memory-copy), e.g. the prefill staging on and off.
#   python3 stage_trace.py <prof dir A> <prof dir B> [top]
# Per GPU, inside the prefill window (first to last mul_mat_q): busy time, idle time, the kernels whose summed time
# changed most between A and B, and the host-to-device copies landing on that GPU (count, busy time as the union of
# the copy intervals, the part of it overlapped by kernels of that GPU).
import csv, glob, sys, collections, re

def load(d):
    k = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
    m = glob.glob(d + '/**/*memory_copy_trace.csv', recursive=True)
    rows = list(csv.DictReader(open(k)))
    cps = list(csv.DictReader(open(m[0]))) if m else []
    return rows, cps

def short(name):
    name = re.sub(r'\(.*', '', name)
    name = re.sub(r'<.*', '', name)
    return name[:60]

def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def overlap(a, b):
    # a, b: sorted disjoint interval lists
    i = j = 0; t = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: t += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return t

def analyse(d):
    rows, cps = load(d)
    mm = [r for r in rows if 'mul_mat_q<' in r['Kernel_Name']]
    first = min(int(r['Start_Timestamp']) for r in mm)
    last = max(int(r['End_Timestamp']) for r in mm)
    res = {}
    for a in sorted({r['Agent_Id'] for r in rows}):
        rs = [r for r in rows if r['Agent_Id'] == a and first <= int(r['Start_Timestamp']) <= last]
        by = collections.Counter()
        iv = []
        for r in rs:
            s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            by[short(r['Kernel_Name'])] += e - s
            iv.append((s, e))
        ku = union(iv)
        cv = [(int(c['Start_Timestamp']), int(c['End_Timestamp'])) for c in cps
              if c.get('Destination_Agent_Id') == a and first <= int(c['Start_Timestamp']) <= last]
        cu = union(cv)
        res[a] = dict(window=last - first, busy=sum(by.values()), kunion=sum(e - s for s, e in ku), by=by,
                      ncopy=len(cv), copy=sum(e - s for s, e in cu), copy_ov=overlap(ku, cu),
                      copy_sum=sum(e - s for s, e in cv))
    return res

if sys.argv[1] == '--timeline':
    # stage_trace.py --timeline <prof dir> <agent> <start fraction> <ms>: the expert mul_mat_q, the copies onto that
    # GPU and the idle gaps over a slice of the prefill window
    rows, cps = load(sys.argv[2])
    ag = sys.argv[3]
    mm = [r for r in rows if 'mul_mat_q<' in r['Kernel_Name']]
    first = min(int(r['Start_Timestamp']) for r in mm)
    last = max(int(r['End_Timestamp']) for r in mm)
    t0 = first + int(float(sys.argv[4]) * (last - first)); t1 = t0 + int(float(sys.argv[5]) * 1e6)
    ev = []
    ks = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), r) for r in rows if r['Agent_Id'] == ag)
    prev = None
    for s, e, r in ks:
        if prev is not None and s - prev > 300000 and t0 <= s <= t1:
            ev.append((prev, 'gap', f'{(s - prev)/1e6:.2f} ms idle, next {short(r["Kernel_Name"])}'))
        prev = max(prev or 0, e)
        if t0 <= s <= t1 and 'mul_mat_q<' in r['Kernel_Name'] and int(r.get('Grid_Size_Z', 1)) > 1:
            ev.append((s, 'mmq', f'{(e - s)/1e6:.2f} ms expert mmq'))
    for c in cps:
        s, e = int(c['Start_Timestamp']), int(c['End_Timestamp'])
        if c.get('Destination_Agent_Id') == ag and t0 <= s <= t1:
            ev.append((s, 'copy', f'{(e - s)/1e6:.2f} ms copy, ends +{(e - t0)/1e6:.2f}'))
    for t, kind, txt in sorted(ev):
        print(f'{(t - t0)/1e6:9.2f} ms  {kind:5s} {txt}')
    sys.exit(0)

if sys.argv[1] == '--ubatch':
    # stage_trace.py --ubatch <prof dir> <agent> <expert mmq per ubatch>: per ubatch (every n-th expert mul_mat_q) the
    # wall time and the idle time, and where in the ubatch (expert mmq index) the idle gaps sit
    rows, cps = load(sys.argv[2])
    ag = sys.argv[3]; per = int(sys.argv[4])
    ks = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), r) for r in rows if r['Agent_Id'] == ag)
    ex = [i for i, (s, e, r) in enumerate(ks) if 'mul_mat_q<' in r['Kernel_Name'] and int(r.get('Grid_Size_Z', 1)) > 1]
    print(f'{len(ex)} expert mmq, {len(ex) / per:.2f} ubatches')
    pos_idle = collections.Counter()
    for u in range(len(ex) // per):
        a, b = ex[u * per], ex[(u + 1) * per - 1]
        idle = 0; big = []
        j = 0
        for i in range(a + 1, b + 1):
            g = ks[i][0] - max(e for s, e, r in ks[a:i][-8:])
            while j < per and ex[u * per + j] < i: j += 1
            if g > 0:
                idle += g; pos_idle[j] += g
                if g > 1e6: big.append(f'{g/1e6:.1f}@{j}')
        print(f'ubatch {u:2d}: {(ks[b][1] - ks[a][0])/1e6:7.1f} ms, idle {idle/1e6:6.1f} ms  {" ".join(big[:8])}')
    tot = sum(pos_idle.values())
    print('idle by position in the ubatch (expert mmq index, share of idle):')
    print('  ' + ' '.join(f'{p}:{v/tot*100:.0f}%' for p, v in sorted(pos_idle.items(), key=lambda kv: -kv[1])[:16]))
    sys.exit(0)

A, B = analyse(sys.argv[1]), analyse(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
for a in A:
    if a not in B: continue
    x, y = A[a], B[a]
    print(f'agent {a}:')
    for tag, v in (('A', x), ('B', y)):
        print(f'  {tag}: window {v["window"]/1e9:.2f} s, kernel sum {v["busy"]/1e9:.2f} s, kernel union {v["kunion"]/1e9:.2f} s,'
              f' idle {(v["window"]-v["kunion"])/1e9:.2f} s; H2D copies {v["ncopy"]} x, union {v["copy"]/1e9:.2f} s'
              f' (sum {v["copy_sum"]/1e9:.2f} s), overlapped by kernels {v["copy_ov"]/1e9:.2f} s')
    keys = set(x['by']) | set(y['by'])
    diff = sorted(keys, key=lambda k: -abs(y['by'][k] - x['by'][k]))[:top]
    print(f'  kernels, B - A (s):')
    for k in diff:
        print(f'    {(y["by"][k]-x["by"][k])/1e9:+7.3f}   A {x["by"][k]/1e9:7.3f}  B {y["by"][k]/1e9:7.3f}  {k}')

#!/usr/bin/env python3
# kernel_ngrams.py <kernel_trace.csv> [n, default 3] [top, default 40] : fusion targets. Takes the steady-state target
# verify passes of the first GPU (evaluations separated by > 150 us of idle, the biggest ones), counts the kernel
# sequences of length 1..n per pass, and prints the most frequent ones with their GPU time per pass.
import collections, csv, re, sys

path = sys.argv[1]
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 3
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40


def short(name):
    n = name.replace('(anonymous namespace)::', '')
    n = re.sub(r'\(.*', '', n)
    n = re.sub(r'<.*', '', n).replace('void ', '').strip()
    return n[-40:]


ev = collections.defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        ev[r.get('Agent_Id', '')].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))
agent, x = sorted(ev.items())[0]
x.sort()

segs, cur, last = [], [], None
for s, e, nm in x:
    if last is not None and s - last > 150e3:
        segs.append(cur); cur = []
    cur.append((s, e, nm)); last = e if last is None else max(last, e)
segs.append(cur)

# verify passes: the modal size among the big evaluations (prefill ubatches are bigger, drafts much smaller)
sizes = collections.Counter(len(sg) // 50 for sg in segs if len(sg) > 1000)
mode = sizes.most_common(1)[0][0]
passes = [sg for sg in segs if len(sg) // 50 == mode]
print('%s: %d verify passes of ~%d kernels' % (agent, len(passes), mode*50))

for n in range(1, n_max + 1):
    cnt, tim = collections.Counter(), collections.Counter()
    for sg in passes:
        names = [short(nm) for _, _, nm in sg]
        durs = [(e - s)/1e3 for s, e, _ in sg]
        for i in range(len(names) - n + 1):
            k = ' > '.join(names[i:i + n])
            cnt[k] += 1
            tim[k] += sum(durs[i:i + n])
    print('\n== sequences of %d (per pass: count, GPU us)' % n)
    for k, v in cnt.most_common(top):
        print('%7.1f x %8.1f us  %s' % (v/len(passes), tim[k]/len(passes), k))

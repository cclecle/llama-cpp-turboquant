#!/usr/bin/env python3
# mmv_breakdown.py <rocprof kernel_trace.csv> <steps> [agent] [filter] : the mat-vec kernels of a decode trace grouped
# by template arguments and launch shape, per speculative step, on one GPU (default: the first agent seen). A group
# is one weight shape at one batch width, so this maps the dense mat-vec bill to matrices: blocks = grid / workgroup.
# filter (default 'mul_mat_vec') is a substring of the kernel name.
import collections
import csv
import sys

path, steps = sys.argv[1], int(sys.argv[2])
agent = (sys.argv[3] if len(sys.argv) > 3 else '') or None
flt = sys.argv[4] if len(sys.argv) > 4 else 'mul_mat_vec'

n = collections.Counter()
t = collections.Counter()
for r in csv.DictReader(open(path)):
    if agent is None:
        agent = r['Agent_Id']
    if r['Agent_Id'] != agent or flt not in r['Kernel_Name']:
        continue
    name = r['Kernel_Name']
    name = name[5:] if name.startswith('void ') else name
    name = name.split('>(')[0] + '>' if '>(' in name else name.split('(')[0]
    wx, wy = int(r['Workgroup_Size_X']), int(r['Workgroup_Size_Y'])
    bx, by, bz = int(r['Grid_Size_X']) // wx, int(r['Grid_Size_Y']) // wy, int(r['Grid_Size_Z']) // max(1, int(r['Workgroup_Size_Z']))
    key = (name, f'{bx}x{by}x{bz}', f'{wx}x{wy}')
    n[key] += 1
    t[key] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])

total = sum(t.values())
print(f'agent {agent}: {sum(n.values()) / steps:.0f} kernels, {total / 1e6 / steps:.2f} ms per step ({flt})')
print(f'{"per step":>8} {"us/step":>8} {"us each":>8}  blocks     wg     kernel')
for key, v in sorted(t.items(), key=lambda kv: -kv[1])[:30]:
    print(f'{n[key] / steps:8.1f} {v / 1e3 / steps:8.1f} {v / 1e3 / n[key]:8.1f}  {key[1]:<10} {key[2]:<6} {key[0]}')

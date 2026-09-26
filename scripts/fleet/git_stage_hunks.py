#!/usr/bin/env python3
# git_stage_hunks.py list <file> | stage <file> <idx,idx,...> : stage chosen hunks of a file's working-tree diff, to split
# one file's changes over several commits without an interactive git. Patches go to git apply as bytes, so Windows text
# mode cannot turn them into CRLF (git apply then rejects them). Run from the repository root.
import subprocess, sys
mode, path = sys.argv[1], sys.argv[2]
d = subprocess.run(['git', 'diff', '-U3', '--', path], capture_output=True).stdout
lines = d.splitlines(keepends=True)
head, hunks, cur = [], [], None
for l in lines:
    if l.startswith(b'@@'):
        cur = [l]; hunks.append(cur)
    elif cur is None:
        head.append(l)
    else:
        cur.append(l)
if mode == 'list':
    for i, h in enumerate(hunks):
        adds = [x[1:].strip().decode('utf8', 'replace') for x in h[1:] if x.startswith(b'+')][:3]
        print(i, h[0].strip()[:40].decode(), '|', ' / '.join(a[:70] for a in adds))
else:
    sel = [int(x) for x in sys.argv[3].split(',')]
    patch = b''.join(head) + b''.join(b''.join(hunks[i]) for i in sel)
    r = subprocess.run(['git', 'apply', '--cached', '--recount', '-'], input=patch, capture_output=True)
    print(r.returncode, r.stderr.decode())

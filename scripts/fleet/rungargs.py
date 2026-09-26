#!/usr/bin/env python3
# rungargs.py <unit> <rung id> <out file> : write the exact child args the router last used for a rung
# (from the unit's journal, "spawning server instance with args:"), one per line, minus --host/--port/--alias,
# ready for `mapfile -t A < out; llama-server "${A[@]}" --port N`.
import re, subprocess, sys
unit, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
j = subprocess.run(['journalctl', '-u', unit, '--since', '-30 days', '--no-pager', '-o', 'cat'],
                   capture_output=True, text=True).stdout.splitlines()
idx = [i for i, l in enumerate(j) if 'spawning server instance with name=' + name + ' ' in l]
if not idx:
    sys.exit('no spawn of %s in %s' % (name, unit))
k = [x for x in range(idx[-1], idx[-1] + 5) if 'with args:' in j[x]][0]
args = []
for l in j[k + 1:k + 400]:
    m = re.match(r'^.*load:\s{3,}(\S.*)$', l)
    if not m:
        break
    args.append(m.group(1).strip())
clean, skip = [], False
for x in args[1:]:
    if skip:
        skip = False
        continue
    if x in ('--host', '--port', '--alias'):
        skip = True
        continue
    clean.append(x)
open(out, 'w').write('\n'.join(clean) + '\n')
print(name, len(clean), 'args ->', out)

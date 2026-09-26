#!/usr/bin/env python3
# ctxcheck.py [release] : flag every rung whose per-slot context exceeds what the model can use: the pre-YaRN
# original context when the rung runs with rope-scaling none on a YaRN model, else the trained context_length.
# Starts two GPU-less scratch routers (ports 20090/20091) over both preset stores; safe while production runs.
import collections, json, os, subprocess, sys, time, urllib.request
REL = sys.argv[1] if len(sys.argv) > 1 else 'v17'
B = '/opt/llamacpp/llama-cpp-mine-%s/build3/bin' % REL
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, HIP_VISIBLE_DEVICES='-1', LD_LIBRARY_PATH=B)
procs = []
for port, store in ((20090, 'SINGLEGPU'), (20091, 'DUALGPU')):
    procs.append(subprocess.Popen([B + '/llama-server', '--host', '127.0.0.1', '--port', str(port), '--models-preset',
        '/opt/llamacpp-config/%s/main.ini' % store, '--no-models-autoload'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
def models(port):
    for _ in range(60):
        try:
            return json.load(urllib.request.urlopen('http://127.0.0.1:%d/models' % port))['data']
        except Exception:
            time.sleep(1)
    sys.exit('router on %d did not start' % port)
def kv(path):
    out = subprocess.run([sys.executable, os.path.join(HERE, 'ggufkv.py'), path, 'context_length', 'rope.scaling'],
                         capture_output=True, text=True).stdout
    return {l.split(' = ')[0].split('.', 1)[1]: l.split(' = ', 1)[1] for l in out.splitlines() if ' = ' in l}
try:
    cache, rows = {}, []
    for port, store in ((20090, 'SINGLE'), (20091, 'DUAL')):
        for m in models(port):
            a = m['status']['args']; g = lambda k: a[a.index(k) + 1] if k in a else None
            path = g('--model')
            if not path or '--embedding' in a or '--reranking' in a:
                continue
            d = cache.setdefault(path, kv(path))
            train = int(d.get('context_length', 0) or 0); orig = d.get('rope.scaling.original_context_length')
            yarn_off = g('--rope-scaling') == 'none' and d.get('rope.scaling.type') == 'yarn' and orig
            limit = int(orig) if yarn_off else train
            per = int(g('--ctx-size') or 0) // max(1, int(g('--parallel') or 1))
            rows.append((store, m['id'], per, limit, g('--rope-scaling'), d.get('rope.scaling.type'), orig, train, g('--override-kv') or ''))
    bad = [r for r in rows if r[3] and r[2] > r[3]]
    print('rungs checked: %d, over the usable context: %d' % (len(rows), len(bad)))
    by = collections.defaultdict(list)
    for r in bad:
        by[(r[0], r[1].split(':')[0])].append(r)
    for (st, fam), rs in sorted(by.items()):
        r = rs[0]
        print('%-6s %-34s rungs=%2d per-slot %d..%d usable %d (ctx_train %s, rope-scaling %s, gguf %s orig %s) %s' % (
            st, fam, len(rs), min(x[2] for x in rs), max(x[2] for x in rs), r[3], r[7], r[4], r[5], r[6], r[8][:60]))
finally:
    for p in procs:
        p.kill()

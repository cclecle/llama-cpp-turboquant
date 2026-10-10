#!/bin/bash
# s9-router-args.sh <new preset file> <rung id>=<measured args file>... : a preset change checked through the router
# before it goes into the config store. A copy of /opt/llamacpp-config with the new file in DUALGPU/, a scratch router
# (port 20099, llamacpp-both's flags, the v20dev binary); per rung: /models/load, the child args from the router log
# ("spawning server instance with args"), then unload; diff against the args the rung was measured with (host/port/
# alias dropped, order ignored). Production stays stopped and is not restarted.
cd /mnt/gguf/r9v/bench
NEW=${1:?new preset file}; shift
W=/root/work-$(date +%Y%m%d)/routercheck; rm -rf $W; mkdir -p $W
cp -a /opt/llamacpp-config $W/store
cp "$NEW" $W/store/DUALGPU/$(basename "$NEW")
B=/opt/llamacpp/llama-cpp-mine-v20dev/build3/bin
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B setsid $B/llama-server --host 127.0.0.1 --port 20099 --models-preset $W/store/DUALGPU/main.ini \
  --models-max 1 --no-models-autoload --slots --cpu-mask 3f3f --cpu-strict 1 --cpu-mask-batch 3f3f --cpu-strict-batch 1 > $W/router.log 2>&1 &
R=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:20099/health >/dev/null 2>&1 && break; sleep 1; done
for pair in "$@"; do
  id=${pair%%=*}; measured=${pair#*=}
  curl -s -X POST http://127.0.0.1:20099/models/load -H 'Content-Type: application/json' -d "{\"model\": \"$id\"}" > /dev/null
  for i in $(seq 1 60); do grep -aq "spawning server instance with name=$id " $W/router.log && break; sleep 1; done
  sleep 3
  python3 - "$W/router.log" "$id" "$measured" <<'PY'
import re, sys
log, name, measured = sys.argv[1:4]
j = open(log, errors='ignore').read().splitlines()
idx = [i for i, l in enumerate(j) if 'spawning server instance with name=' + name + ' ' in l]
if not idx:
    sys.exit('%s: no spawn line (rejected by the router?)' % name)
k = [x for x in range(idx[-1], idx[-1] + 5) if 'with args:' in j[x]][0]
args = []
for l in j[k + 1:k + 400]:
    m = re.match(r'^.*load:\s{3,}(\S.*)$', l)
    if not m:
        break
    args.append(m.group(1).strip())
def pairs(a):
    out, i = {}, 0
    while i < len(a):
        x = a[i]
        if x.startswith('--') and i + 1 < len(a) and not a[i + 1].startswith('--'):
            out.setdefault(x, []).append(a[i + 1]); i += 2
        else:
            out.setdefault(x, []).append(''); i += 1
    return out
drop = {'--host', '--port', '--alias'}
got = {k: v for k, v in pairs(args[1:]).items() if k not in drop}
want = {k: v for k, v in pairs([x for x in open(measured).read().split('\n') if x]).items() if k not in drop}
diff = [(k, want.get(k), got.get(k)) for k in sorted(set(got) | set(want)) if sorted(got.get(k, [])) != sorted(want.get(k, []))]
print('%s: %d args from the router, %s' % (name, len(args) - 1, 'identical to the measured ones' if not diff else '%d differences:' % len(diff)))
for k, w, g in diff:
    print('   %-32s measured %s   router %s' % (k, w, g))
PY
  curl -s -X POST http://127.0.0.1:20099/models/unload -H 'Content-Type: application/json' -d "{\"model\": \"$id\"}" > /dev/null
  sleep 15
done
kill -TERM -- -$R 2>/dev/null; sleep 10; kill -KILL -- -$R 2>/dev/null
echo "production (left stopped): $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr '\n' ' ')"
echo S9_DONE

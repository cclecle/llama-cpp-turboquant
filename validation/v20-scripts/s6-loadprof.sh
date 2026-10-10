#!/bin/bash
# s6-loadprof.sh [args file, default q35-tiered-44g.args] [build, default v20dev] : where a tiered Qwen3.5-122B load
# spends its 200-555 s (v19 :XS maps the experts and loads in 42 s). Starts llama-server, takes a gdb stack sample of
# every thread every LP_EVERY (5) s until /health answers, then counts per sample the innermost frame of our code on the
# busiest threads, and prints load time, page-cache and RAM state before/after. Production stays stopped.
cd /mnt/gguf/r9v/bench
A_FILE=${1:-q35-tiered-44g.args}; b=${2:-v20dev}
B=/opt/llamacpp/llama-cpp-mine-$b/build3/bin
out=loadprof-$b; rm -rf $out; mkdir -p $out
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "before: loadavg $(cut -d' ' -f1-3 /proc/loadavg); $(grep -E 'MemAvailable|^Cached' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
mapfile -t A < $A_FILE
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B setsid $B/llama-server "${A[@]}" --host 127.0.0.1 --port 8092 > $out/server.log 2>&1 &
SRV=$!
t0=$(date +%s); i=0
until curl -sf -o /dev/null http://127.0.0.1:8092/health; do
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail $out/server.log; exit 1; }
  i=$((i + 1))
  timeout 20 gdb -p $SRV -batch -nx -ex 'set pagination off' -ex 'thread apply all bt 25' > $out/s$i.txt 2>/dev/null
  echo "$(( $(date +%s) - t0 )) $(cut -d' ' -f1 /proc/loadavg) $(awk '/MemAvailable/ {print $2}' /proc/meminfo) $(top -b -n1 -p $SRV | awk 'NR==8 {print $9}')" >> $out/timeline.txt
  sleep ${LP_EVERY:-5}
done
echo "loaded in $(( $(date +%s) - t0 )) s, $i samples"
echo "after: loadavg $(cut -d' ' -f1-3 /proc/loadavg); $(grep -E 'MemAvailable|^Cached' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
kill -TERM -- -$SRV; sleep 15; kill -KILL -- -$SRV 2>/dev/null
echo "timeline (s, loadavg, MemAvailable kB, server %CPU):"; awk 'NR%6==1' $out/timeline.txt
python3 - $out <<'PY'
import collections, glob, re, sys
ours = re.compile(r'(llama|ggml|gguf|tiered|meta|load|memcpy|madvise|pread|read|mmap|hip|hsa|amd)', re.I)
top = collections.Counter(); n = 0
for f in sorted(glob.glob(sys.argv[1] + '/s*.txt')):
    n += 1
    for th in open(f, errors='ignore').read().split('\nThread ')[1:]:
        frames = re.findall(r'^#\d+\s+(?:0x[0-9a-f]+ in )?([^\s(]+)', th, re.M)
        if not frames or frames[0] in ('__futex_abstimed_wait_common', 'futex_wait', '__GI___poll', 'epoll_wait', '__GI_epoll_wait',
                                        'pthread_cond_wait@@GLIBC_2.3.2', '__pthread_cond_wait', 'syscall', '__libc_recv', 'nanosleep'):
            continue
        inner = next((fr for fr in frames if ours.search(fr)), frames[0])
        top['%s <- %s' % (frames[0], inner) if inner != frames[0] else frames[0]] += 1
print('%d samples; busy innermost frames (frame <- first frame of our code):' % n)
for k, v in top.most_common(18):
    print('  %4d  %s' % (v, k[:150]))
PY
echo S6_DONE

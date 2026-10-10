#!/bin/bash
# s17-release.sh [build, default v22] : load time with each copied tensor's mmap pages released during the load
# (2582fa44c). Production's Qwen3.5-122B :XS (prod-xs.args, rebuilt from the router's spawn log) took 6 min 50 s on
# v21 with RAM full. Loads it once, then Flash-Next :XL tiered twice (the second is the cached reload, 12 s on v21).
# Per load: seconds to /health (flashnext.sh FN_LOAD_ONLY=1), the lowest MemAvailable during the load, page faults.
# Stops production; restarts it at the end.
cd /mnt/gguf/r9v/bench
b=${1:-v22}
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5
run() { # <tag> <args file> <ctx>
  echo "### $1 loadavg $(cut -d' ' -f1-3 /proc/loadavg) $(grep -E '^Cached|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  ( m=999999999; while :; do a=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); [ $a -lt $m ] && m=$a && echo $m > /tmp/s17-min; sleep 1; done ) &
  W=$!
  FN_LOAD_ONLY=1 FN_CTX=$3 FN_ARGS=$2 FN_TAG=-s17-$1 bash flashnext.sh $b 2>&1 | grep -E 'load:|died|memory:'
  kill $W; echo "    lowest MemAvailable during the load: $(( $(cat /tmp/s17-min) / 1048576 )) GiB"; rm -f /tmp/s17-min
  grep -E 'pswpin|pgmigrate_success|compact_stall' /proc/vmstat | tr '\n' ' '; echo
}
run q35xs prod-xs.args 262144
run fn fn-xl-tiered-r9vlike.args 65536
run fn fn-xl-tiered-r9vlike.args 65536
systemctl start llamacpp-0 llamacpp-1 llamacpp-both; sleep 20
for port in 20000 20001 20002; do echo "production $port: $(curl -s -m 5 http://127.0.0.1:$port/health)"; done
echo S17_DONE

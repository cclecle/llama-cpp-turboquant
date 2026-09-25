#!/bin/bash
# rungsweep.sh <release> [frac] : every rung of BOTH preset stores once, through scratch routers whose args mirror
# llamacpp-0 (SINGLEGPU, GPU 0) and llamacpp-both (DUALGPU, GPU 0+1). Prompt = frac (default 0.5) of the per-slot
# context. Stops the 3 production units and restarts them at the end whatever happens. Re-running resumes.
# Output: /root/work-<date>/sweep-<release>-{single,dual}.log (see rungsweep.py for the line format)
set -u
V=${1:?release, e.g. v17}; FRAC=${2:-0.5}; PORT=20099
BIN=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
W=/root/work-$(date +%Y%m%d); mkdir -p $W
PID=
finish() { [ -n "$PID" ] && { kill -TERM $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; }; systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "SWEEP_ALL_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr '\n' ' ')"; }
trap finish EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 4

run_store() {
    local name=$1 dev=$2; shift 2
    HIP_VISIBLE_DEVICES=$dev LD_LIBRARY_PATH=$BIN nohup $BIN/llama-server --host 127.0.0.1 --port $PORT --models-max 1 --no-models-autoload --slots "$@" > $W/sweep-$V-$name.router.log 2>&1 &
    PID=$!
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
    python3 "$(dirname "$0")/rungsweep.py" $PORT $W/sweep-$V-$name.log $FRAC
    kill -TERM $PID 2>/dev/null; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -9 $PID 2>/dev/null; PID=; sleep 5
}

run_store dual 0,1 --models-preset /opt/llamacpp-config/DUALGPU/main.ini --cpu-mask 3f3f --cpu-strict 1 --cpu-mask-batch 3f3f --cpu-strict-batch 1
run_store single 0 --models-preset /opt/llamacpp-config/SINGLEGPU/main.ini -t 8 -tb 8 --cpu-range 0-7 --cpu-strict 1 --cpu-range-batch 0-7 --cpu-strict-batch 1

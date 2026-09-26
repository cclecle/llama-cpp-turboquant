#!/bin/bash
# rungsweep.sh <release> [frac] : every rung of BOTH preset stores once, through scratch routers whose args mirror
# llamacpp-0 (SINGLEGPU, GPU 0) and llamacpp-both (DUALGPU, GPU 0+1). Prompt = frac (default 0.5) of the per-slot
# context. Stops the 3 production units and restarts them at the end whatever happens. Re-running resumes.
# Output: /root/work-<date>/sweep-<release>-{dual,single,single-g1}.log (see rungsweep.py for the line format).
# SINGLEGPU runs on both cards at once, one router per card, each claiming the next free rung.
set -u
V=${1:?release, e.g. v17}; FRAC=${2:-0.5}; PORT=20099
BIN=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
W=${SWEEP_DIR:-/root/work-$(date +%Y%m%d)}; mkdir -p $W   # set SWEEP_DIR to resume a sweep started on another day
PID=
finish() { [ -n "$PID" ] && { kill -TERM $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; }; pkill -P $$ -f rungsweep.py 2>/dev/null; systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "SWEEP_ALL_DONE production: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr '\n' ' ')"; }
trap finish EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 4

# start_router <tag> <port> <HIP devices> <router args...> ; sets RPID
start_router() {
    local tag=$1 port=$2 dev=$3; shift 3
    HIP_VISIBLE_DEVICES=$dev LD_LIBRARY_PATH=$BIN nohup $BIN/llama-server --host 127.0.0.1 --port $port --models-max 1 --no-models-autoload --slots "$@" > $W/sweep-$V-$tag.router.log 2>&1 &
    RPID=$!
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && break; sleep 1; done
}
stop_router() { kill -TERM $1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -9 $1 2>/dev/null; sleep 5; }

SINGLE="--models-preset /opt/llamacpp-config/SINGLEGPU/main.ini -t 8 -tb 8 --cpu-strict 1 --cpu-strict-batch 1"

# DUALGPU: one router over both cards
start_router dual $PORT 0,1 --models-preset /opt/llamacpp-config/DUALGPU/main.ini --cpu-mask 3f3f --cpu-strict 1 --cpu-mask-batch 3f3f --cpu-strict-batch 1
PID=$RPID
python3 "$(dirname "$0")/rungsweep.py" $PORT $W/sweep-$V-dual.log $FRAC
stop_router $PID; PID=

# SINGLEGPU: one router per card (mirroring llamacpp-0 / llamacpp-1), each claims the next free rung
rm -rf $W/claims
start_router single $PORT 0 $SINGLE --cpu-range 0-7 --cpu-range-batch 0-7
P0=$RPID
start_router single-g1 $((PORT-1)) 1 $SINGLE --cpu-range 8-15 --cpu-range-batch 8-15
P1=$RPID
PID="$P0 $P1"
python3 "$(dirname "$0")/rungsweep.py" $PORT $W/sweep-$V-single.log $FRAC 0 0/2 &
S0=$!
python3 "$(dirname "$0")/rungsweep.py" $((PORT-1)) $W/sweep-$V-single-g1.log $FRAC 1 1/2 &
S1=$!
wait $S0 $S1
stop_router $P0; stop_router $P1; PID=

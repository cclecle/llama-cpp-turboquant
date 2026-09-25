#!/bin/bash
# rungpass.sh <main.ini> <tag> <model-id>... : measure rungs through a scratch router whose args mirror llamacpp-0 (GPU 0).
# Stops the 3 production units for the duration and restarts them at the end. Output in /root/work-<date>/.
set -u; PORT=20099; PRESET=${1:?usage: rungpass.sh <main.ini> <tag> [ids...]}; TAG=${2:-pass}; LOG=/root/work-$(date +%Y%m%d)/rungpass-$TAG.log; mkdir -p "$(dirname "$LOG")"; shift 2 || true
BIN=${BIN:-/opt/llamacpp/llama-cpp-mine-v16/build3/bin/llama-server}
finish() { kill -TERM $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo RUNGPASS_DONE >> $LOG; }
trap finish EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 4
HIP_VISIBLE_DEVICES=0 nohup $BIN --host 127.0.0.1 --port $PORT -t 8 -tb 8 --models-preset "$PRESET" --models-max 1 --no-models-autoload --slots --cpu-range 0-7 --cpu-strict 1 --cpu-range-batch 0-7 --cpu-strict-batch 1 > "$(dirname "$LOG")/router-$TAG.log" 2>&1 &
PID=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
python3 "$(dirname "$0")/rungmeasure.py" $PORT "$@" > $LOG 2>&1

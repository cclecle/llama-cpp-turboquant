#!/bin/bash
# fleetcheck.sh: fleet smoke check before a cutover (27 mid rungs, both stores). one medium profile per family, both preset stores, args mirror the units
# (llamacpp-both for DUALGPU, llamacpp-0 for SINGLEGPU) minus --no-models-autoload and the slot cache.
# Restarts production at the end whatever happens. Set BIN to the release under test.
set -u
BIN=${BIN:-/opt/llamacpp/llama-cpp-mine-v16/build3/bin/llama-server}
PORT=20099
LOG=/root/fleetcheck
mkdir -p $LOG
DUAL="Qwen3.5-122B-A10B:M Mistral-Small-4-119B-A6B:M Mistral-Medium-3.5-128B:M Qwen3.6-40B:CODE:M Qwen3.6-27B:HQ:M gemma-4-31B:HQ:M Qwen-AgentWorld-35B-A3B:HQ:M Qwen3.6-35B-A3B:HQ:M Qwen3.6-27B-Fable:HQ:M Qwen3.8-27B:HQ:M Qwen3.8-Flash-Next:M"
SINGLE="Qwen3.6-27B:M Qwen3.5-9B:M Qwen3-Embedding-8B jina-reranker-v2-base-multilingual GLM-4.7-Flash-30B-A3B:M Qwen-AgentWorld-35B-A3B:M Qwen3.6-35B-A3B:M gemma-4-31B:M gemma-4-26B-A4B:M Qwen3-Coder-Next Devstral-Small-2-24B-Insctruct:M Magistral-Small:M Qwen3.6-27B-Fable:M Muse-Glimmer-30B:M Qwen3.8-27B:M gemma-4-12B:M"
finish() { systemctl start llamacpp-0 llamacpp-1 llamacpp-both; echo "production restarted: $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr '\n' ' ')"; echo FLEET_ALL_DONE; }
trap finish EXIT
stopsrv() { kill -TERM $1 2>/dev/null; for i in $(seq 1 120); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -9 $1 2>/dev/null; sleep 5; }

echo "### DUALGPU ($(date +%H:%M))"
HIP_VISIBLE_DEVICES=0,1 nohup $BIN --host 127.0.0.1 --port $PORT --models-preset /opt/llamacpp-config/DUALGPU/main.ini --models-max 1 --slots \
    --cpu-mask 3f3f --cpu-strict 1 --cpu-mask-batch 3f3f --cpu-strict-batch 1 > $LOG/dual.server.log 2>&1 &
PID=$!
python3 "$(dirname "$0")/fleetcheck.py" $PORT $DUAL 2>&1 | tee $LOG/dual.txt
stopsrv $PID

echo "### SINGLEGPU on GPU0 ($(date +%H:%M))"
HIP_VISIBLE_DEVICES=0 nohup $BIN --host 127.0.0.1 --port $PORT -t 8 -tb 8 --models-preset /opt/llamacpp-config/SINGLEGPU/main.ini --models-max 1 --slots \
    --cpu-range 0-7 --cpu-strict 1 --cpu-range-batch 0-7 --cpu-strict-batch 1 > $LOG/single.server.log 2>&1 &
PID=$!
python3 "$(dirname "$0")/fleetcheck.py" $PORT $SINGLE 2>&1 | tee $LOG/single.txt
stopsrv $PID

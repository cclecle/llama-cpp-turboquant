#!/bin/bash
# envab.sh <build> <args file> <env setting>... : decode speed of one server config under several environment
# settings (e.g. HIP runtime knobs). Each setting: load, one warm-up, then one greedy decode of 1024 tokens of the
# same fixed prompt. Same prompt + greedy = same token trajectory, so acceptance is identical and only speed
# differs. The settings run in order, then in reverse (A B C C B A) to cancel drift. Production must be stopped.
# Use X=0 for "no extra env". ENVAB_REPEAT=n: n measured requests per load (within-load vs across-load spread);
# ENVAB_NOREV=1: run the settings once, in the given order.
cd /mnt/gguf/r9v/bench
b=$1; args=$2; shift 2
settings=("$@")
order=()
for ((i = 0; i < ${#settings[@]}; i++)); do order+=("${settings[$i]}"); done
[ -n "${ENVAB_NOREV:-}" ] || for ((i = ${#settings[@]} - 1; i >= 0; i--)); do order+=("${settings[$i]}"); done
mapfile -t A < $args
for i in "${!A[@]}"; do [ "${A[$i]}" = --ctx-size ] && A[$((i + 1))]=65536; done
BODY='{"messages":[{"role":"user","content":"Write a very long, detailed essay about the history of the printing press, at least 3000 words."}],"max_tokens":1024,"temperature":0,"cache_prompt":false}'
for e in "${order[@]}"; do
  env $e HIP_VISIBLE_DEVICES=0,1 setsid /opt/llamacpp/llama-cpp-mine-$b/build3/bin/llama-server "${A[@]}" \
    --host 127.0.0.1 --port 8090 > envab.log 2>&1 &
  SRV=$!
  until curl -sf -o /dev/null http://127.0.0.1:8090/health; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "$e: server died"; tail -5 envab.log; break; }; done
  curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}' > /dev/null
  for r in $(seq 1 ${ENVAB_REPEAT:-1}); do
    curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" > envab.json
    python3 -c "import json,sys; t=json.load(open('envab.json'))['timings']; print('%-44s run %s  tg %6.2f t/s  n %d  acc %s/%s' % (sys.argv[1], sys.argv[2], t['predicted_per_second'], t['predicted_n'], t.get('draft_n_accepted'), t.get('draft_n')))" "$e" "$r"
  done
  kill -TERM -- -$SRV; sleep 10; kill -KILL -- -$SRV 2>/dev/null; sleep 2
done

#!/bin/bash
# s10-census.sh [args file, default prof-xl.args] : decode route census of Qwen3.5-122B (--moe-census) on two prompt
# sets that do not include the benchmark prompt (prompt-32k.txt): set A to build hot sets from, set B to score them on.
# One server per set (the census is written when the model is freed), cache 8 slots, the preset's sampling, ~4k-token
# prompts, up to 1,536 tokens each. Then moe_hotset.py: hot sets of 44 GiB from A (uniform, greedy) and from the
# llama-imatrix counts, each scored on B. Production stays stopped.
cd /mnt/gguf/r9v/bench
A_FILE=${1:-prof-xl.args}
B=/opt/llamacpp/llama-cpp-mine-v20dev/build3/bin
T=/opt/llamacpp/tmp-v20
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
for set in A B; do
  rm -f census-$set.txt
  mapfile -t A < $A_FILE
  GGML_CUDA_MOE_CACHE_SLOTS=8 GGML_CUDA_MOE_CENSUS=$PWD/census-$set.txt HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B setsid \
    $B/llama-server "${A[@]}" --host 127.0.0.1 --port 8093 > census-$set.log 2>&1 &
  SRV=$!
  until curl -sf -o /dev/null http://127.0.0.1:8093/health; do sleep 3; kill -0 $SRV 2>/dev/null || { echo "server died"; tail census-$set.log; exit 1; }; done
  python3 - $set <<'PY'
import json, sys, time, urllib.request
s = sys.argv[1]
cpp = open('/root/ppl.txt', errors='ignore').read()
py = open('q35-corpus-heldout.txt', errors='ignore').read()
asks = {
 'A': [('code', cpp[300000:314000], 'Explain what this C++ code does, then propose three concrete improvements with code.'),
       ('code', cpp[600000:614000], 'Find the bugs or risky constructs in this code and explain each one.'),
       ('q', '', 'Explain the causes and consequences of the fall of the Western Roman Empire, in depth.'),
       ('q', '', 'Solve step by step: a tank fills in 6 h with pipe A and empties in 9 h with pipe B; with both open and a leak that empties it in 18 h, how long to fill it? Then generalise.'),
       ('q', '', 'Write a 1500-word short story about a lighthouse keeper who receives letters from the future.'),
       ('code', cpp[900000:914000], 'Rewrite the core of this code in modern, idiomatic C++20, explaining each change.'),
       ('q', '', 'Design a REST API for a library management system: resources, endpoints, error handling, and an example session.'),
       ('q', '', 'Compare TCP and QUIC in detail: handshake, congestion control, head-of-line blocking, and deployment.')],
 'B': [('code', py[100000:114000], 'Explain what this Python code does, module by module.'),
       ('code', py[500000:514000], 'Write unit tests (pytest) for the main functions of this code and explain the coverage.'),
       ('q', '', 'Explain how photosynthesis works, from photons to sugars, for an undergraduate.'),
       ('q', '', 'Prove that there are infinitely many primes, then explain two other proofs and compare them.'),
       ('q', '', 'Write a detailed project plan to migrate a monolith to microservices, with risks and milestones.'),
       ('code', py[900000:914000], 'Refactor this code for readability and performance; show the new code and justify it.'),
       ('q', '', 'Discuss the economic effects of the printing press in early modern Europe.'),
       ('q', '', 'Explain the transformer architecture and how mixture-of-experts layers change it.')],
}[s]
for kind, ctx, q in asks:
    content = (ctx + '\n\n' + q) if ctx else q
    b = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': 1536, 'cache_prompt': True}
    t0 = time.time()
    try:
        j = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8093/v1/chat/completions',
            data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'}), timeout=3600))
        t = j['timings']
        print('%s %-4s prompt %5d gen %4d  %.1f t/s  (%.0f s)' % (s, kind, t['prompt_n'], t['predicted_n'], t['predicted_per_second'], time.time() - t0), flush=True)
    except Exception as e:
        print(s, kind, 'FAIL', e, flush=True)
PY
  kill -TERM -- -$SRV; for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -KILL -- -$SRV 2>/dev/null
  echo "census-$set.txt: $(wc -l < census-$set.txt 2>/dev/null) lines"
done
M=/mnt/gguf/Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf
for src in census-A.txt q35-imatrix.gguf; do
  for pol in uniform greedy; do
    /root/venv/bin/python3 $T/scripts/fleet/moe_hotset.py --model $M --counts $src --eval census-B.txt --budget-gib 44 --policy $pol \
      --missing nearest -o hot-q35-44g-$pol-$(basename $src .txt | sed 's/.gguf//').txt 2>/dev/null | sed 's#^.*: hot#hot#; s#^#'"$src $pol: "'#'
  done
done
echo S10_DONE

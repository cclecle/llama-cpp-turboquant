#!/bin/bash
# flashnext.sh [backends, default "v16 v18 r9v"] : Qwen3.8-Flash-Next UD-IQ4_XS, one benchmark per backend: 65,536 ctx,
# the 32k-token prompt (mkprompt.sh), up to 2,048 generated tokens, temperature 0, shared client openai_bench.py.
#   v16 / v18: the production :XL rung args (rungargs.py, llamacpp-both journal) with --ctx-size 65536
#   r9v:       R9V profile qwen38-mtp4 via r9v-run.sh 65536 (vLLM fork, host process from the unpacked image)
# Production is stopped and left stopped; VRAM per card + host RAM sampled every second. Everything lands in
# /mnt/gguf/r9v/bench. The rig is a Proxmox container: the page cache cannot be dropped between backends, and
# its locked memory is capped at 8 MiB for ssh sessions and units alike. Run it detached, as a transient service:
#   systemd-run --unit=flashnext-bench --collect -p WorkingDirectory=/mnt/gguf/r9v/bench \
#     /bin/bash -c 'bash flashnext.sh > run.log 2>&1'
# FN_ENV: env assignments for the llama servers of this run; FN_TAG: suffix for the result label;
# FN_UB: override the rung's ubatch (and raise the batch to at least it); FN_ARGS: another args file;
# FN_LD: LD_LIBRARY_PATH for a copied build (llama-cpp-mine-<name>/build3/bin copied out of a tree).
# FN_LOAD_ONLY=1: load the server and stop it again (no benchmark), e.g. to read its load log.
cd /mnt/gguf/r9v/bench
BACKENDS=${1:-v16 v18 r9v}
FN_ENV=${FN_ENV:-X=0}; FN_TAG=${FN_TAG:-}
CARDS="0000:03:00.0 0000:07:00.0"
SRV=
stop_srv() { [ -n "$SRV" ] && kill -TERM -- -"$SRV" 2>/dev/null; sleep 10; [ -n "$SRV" ] && kill -KILL -- -"$SRV" 2>/dev/null; SRV=; sleep 3; }
# production is left stopped for the whole campaign (the user restarts it); the trap only cleans up the server
trap 'stop_srv; kill $SAMPLER 2>/dev/null; echo "FLASHNEXT_DONE production (left stopped): $(systemctl is-active llamacpp-0 llamacpp-1 llamacpp-both | tr "\n" " ")"' EXIT
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 5

sample() { # <file>: t, vram used MiB per card, host MemAvailable MiB
  while true; do
    l="$(date +%s)"
    for c in $CARDS; do l="$l $(( $(cat /sys/bus/pci/devices/$c/mem_info_vram_used) >> 20 ))"; done
    echo "$l $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) >> 10 ))" >> "$1"
    sleep 1
  done
}
wait_up() { # <url> <timeout s>: seconds until /health answers 200
  local t0=$(date +%s)
  while ! curl -sf -o /dev/null "$1/health"; do
    sleep 2
    kill -0 "$SRV" 2>/dev/null || { echo "server died"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt "$2" ] && { echo "timeout"; return 1; }
  done
  echo $(( $(date +%s) - t0 ))
}
summ() { # <tag>: idle, peak VRAM per card and host RAM used, from the sampler file
  python3 - "$1.mem" "$1" <<'PY'
import sys
rows = [list(map(int, l.split())) for l in open(sys.argv[1]) if len(l.split()) == 4]
base = rows[0]
print('%s memory: idle VRAM %d/%d MiB, peak VRAM %d/%d MiB, host RAM used peak %d MiB' % (
    sys.argv[2], base[1], base[2], max(r[1] for r in rows), max(r[2] for r in rows), base[3] - min(r[3] for r in rows)))
PY
}

for b in $BACKENDS; do
  echo "=== $b $(date +%T)"
  rm -f $b$FN_TAG.mem; sample $b$FN_TAG.mem & SAMPLER=$!
  sleep 3
  if [ $b = r9v ]; then
    URL=http://127.0.0.1:8004; MODEL=qwen3.8-flash-next
    setsid bash ./r9v-run.sh 65536 > $b$FN_TAG.log 2>&1 & SRV=$!
    up=$(wait_up $URL 3600)
  else
    URL=http://127.0.0.1:8090; MODEL=Qwen3.8-Flash-Next
    mapfile -t A < ${FN_ARGS:-fn-xl.args}
    for i in "${!A[@]}"; do
      [ "${A[$i]}" = --ctx-size ] && A[$((i + 1))]=65536
      [ -n "${FN_UB:-}" ] && [ "${A[$i]}" = --ubatch-size ] && A[$((i + 1))]=$FN_UB
      [ -n "${FN_UB:-}" ] && [ "${A[$i]}" = --batch-size ] && [ "${A[$((i + 1))]}" -lt "$FN_UB" ] && A[$((i + 1))]=$FN_UB
    done
    # a *-rocm714 tree was built on /opt/rocm-7.14 and must run on it (the system linker path is 7.2.4)
    # FN_LD: a library path for a copied build (its RUNPATH still names the tree it was built in)
    LD=${FN_LD:-}; [[ $b == *rocm714 ]] && LD=/opt/rocm-7.14/lib
    env $FN_ENV LD_LIBRARY_PATH=$LD HIP_VISIBLE_DEVICES=0,1 setsid /opt/llamacpp/llama-cpp-mine-$b/build3/bin/llama-server "${A[@]}" \
      --host 127.0.0.1 --port 8090 > $b$FN_TAG.log 2>&1 & SRV=$!
    up=$(wait_up $URL 1200)
  fi
  echo "$b load: $up s"
  if [[ $up =~ ^[0-9]+$ ]] && [ "${FN_LOAD_ONLY:-0}" = 1 ]; then
    :
  elif [[ $up =~ ^[0-9]+$ ]]; then
    BENCH_SAVE_TEXT=$b$FN_TAG.answer.txt python3 openai_bench.py $URL $MODEL prompt-32k.txt $b$FN_TAG results.jsonl 2048 ${FN_WARM:-2}
  else
    tail -30 $b$FN_TAG.log
  fi
  stop_srv
  kill $SAMPLER 2>/dev/null
  summ $b$FN_TAG
done

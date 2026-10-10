#!/bin/bash
# s8-rung.sh <tag> <ctx> <np> <host kv cells per slot> <ub> <hot GiB> <cache slots> [more ...] : Qwen3.5-122B rung
# candidates on tiered experts (the ladder rule, user 2026-10-10: :XL 1x262k, :M 2x131k, :S 4x65k, :XS 8x32k, 1/4 of
# the slot's context as host KV). Each group of 7 arguments is one candidate: its args are the production :XS rung's
# (Qwen3.5_XS.args: the router's spawn, the [*] block included) with --cpu-moe / moe-hybrid replaced by ROCm0_TIERED +
# the hot set of that budget (route counts q35-imatrix.gguf), then flashnext.sh like a client: prompt caching on, the
# preset's sampling, a random first word per request (BENCH_NONCE: no request reuses another's cache), the 32k prompt, 2,048 tokens. Prints the peak VRAM per card, prefill and decode, loadavg.
# Needs the v20 build (host KV under -sm tensor crashed on v19; the cache). Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v20
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
while [ $# -ge 7 ]; do
  tag=$1 ctx=$2 np=$3 kv=$4 ub=$5 G=$6 slots=$7; shift 7
  hot=/opt/llamacpp-config/DUALGPU/hot/Qwen3.5-122B-A10B-${G}g-uniform.txt
  [ -f $hot ] || /root/venv/bin/python3 $T/scripts/fleet/moe_hotset.py --model /mnt/gguf/Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf \
    --counts q35-imatrix.gguf --budget-gib $G --policy uniform --missing nearest -o $hot | tail -1
  python3 - $np $kv $hot > rung-$tag.args <<'PY'
import sys
np_, kv, hot = sys.argv[1:4]
a = open('/root/work-20261009-regress/Qwen3.5_XS.args').read().split('\n')
out, i = [], 0
while i < len(a):
    x = a[i]
    if x == '--cpu-moe':
        i += 1; continue
    if x.startswith('--moe-hybrid'):
        i += 2; continue
    if x == '--parallel':
        out += [x, np_]; i += 2; continue
    if x == '--n-cpu-kv-cells':
        if kv != '0':
            out += [x, kv]
        i += 2; continue
    out.append(x); i += 1
out += ['--override-tensor', r'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED', '--moe-hot-experts', hot]
print('\n'.join(x for x in out if x != ''))
PY
  E="X=0"; [ "$slots" != 0 ] && E="GGML_CUDA_MOE_CACHE_SLOTS=$slots"
  echo "### $tag: ${np}x$((ctx / np)) (c $ctx), host kv $kv/slot, ub $ub, hot ${G} GiB, cache $slots; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  BENCH_NONCE=1 BENCH_CACHE_PROMPT=1 BENCH_SERVER_SAMPLING=1 FN_CTX=$ctx FN_UB=$ub FN_ENV="$E" FN_ARGS=rung-$tag.args FN_TAG=-s8-$tag \
    bash flashnext.sh v20dev 2>&1 | grep -E 'load:|died|memory:'
  python3 step_ms.py v20dev-s8-$tag results.jsonl | head -1
done
echo S8_DONE

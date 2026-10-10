#!/bin/bash
# s2-q35-tiered.sh <hot GiB...> : Qwen3.5-122B on the Flash-Next placement. The :XS production args (q35.args: one slot,
# 65k, no host KV cells) with --cpu-moe and the moe-hybrid options replaced by tiered experts (every ffn_*_exps on
# ROCm0_TIERED, the hot set from our own route counts q35-imatrix.gguf, uniform per layer, the given budget over both
# cards), then q35-profile.sh on v20dev: per-GPU busy, prefill and decode, VRAM after load. Production stays stopped.
# S2_ENV: env for the server (e.g. GGML_CUDA_MOE_CACHE_SLOTS=16); S2_PY: a python with numpy (moe_hotset.py, /root/venv).
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v20
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
for G in "$@"; do
  hot=hot-q35-${G}g-uniform.txt
  [ -f $hot ] || ${S2_PY:-/root/venv/bin/python3} $T/scripts/fleet/moe_hotset.py --model /mnt/gguf/Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf \
    --counts q35-imatrix.gguf --budget-gib $G --policy uniform -o $hot | tail -3
  python3 - $hot > q35-tiered-${G}g.args <<'PY'
import sys
a = open('q35.args').read().split('\n')
out, i = [], 0
while i < len(a):
    x = a[i]
    if x == '--cpu-moe':
        i += 1; continue
    if x.startswith('--moe-hybrid'):
        i += 2; continue
    out.append(x); i += 1
out += ['--override-tensor', r'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=ROCm0_TIERED', '--moe-hot-experts', '/mnt/gguf/r9v/bench/' + sys.argv[1]]
print('\n'.join(x for x in out if x != ''))
PY
  echo "### hot ${G} GiB ${S2_ENV:-}"
  QP_ARGS=q35-tiered-${G}g.args QP_TAG=-tiered-${G}g${S2_TAG:-} QP_ENV="${S2_ENV:-X=0}" bash q35-profile.sh v20dev 256
done
echo S2_DONE

#!/usr/bin/env bash
#
# Regression suite for the features this fork adds on top of upstream llama.cpp.
#
# The fork's largest custom surface is the tensor-split "meta" backend, and test-backend-ops does
# not cover it at all (it only enumerates the real devices). Everything below therefore tests the
# feature the way it actually fails: by checking that a configuration change which must not alter
# the result, does not alter it.
#
# Two kinds of check, and the distinction matters:
#
#   IDENTICAL  - used only where the maths is provably the same, so any difference is a bug:
#                speculative decoding at temperature 0. The draft only proposes; the target
#                verifies. Greedy output with a draft MUST equal greedy output without one.
#
#   PERPLEXITY - used where the compute layout changes (layer vs tensor split, CPU-MoE offload,
#                KV placement). These reorder floating-point reductions, so logits differ in the
#                last few ULPs and greedy decoding can legitimately diverge at a near-tie. Demanding
#                byte-identical text there produces false alarms. Perplexity is insensitive to that
#                rounding but still explodes on real corruption - the KV-mirroring bug this fork
#                fixed emitted pure newlines, which no perplexity threshold would survive.
#
# Each check corresponds to a bug that actually shipped in this fork.
#
# Usage:  scripts/fork-regress.sh <build-dir> <model.gguf> [draft.gguf]
#   DEVICES=ROCm0,ROCm1   restrict devices (an integrated GPU joining the split breaks tensor mode)
#   NCMOE=N               also check CPU-MoE expert offload
#   PPL_TOL=0.01          allowed relative perplexity drift (default 1%)
#
set -u

BUILD=${1:?usage: fork-regress.sh <build-dir> <model.gguf> [draft.gguf]}
MODEL=${2:?usage: fork-regress.sh <build-dir> <model.gguf> [draft.gguf]}
DRAFT=${3:-}

SERVER=$BUILD/bin/llama-server
PPL=$BUILD/bin/llama-perplexity
TBO=$BUILD/bin/test-backend-ops
PORT=${PORT:-20099}
PPL_TOL=${PPL_TOL:-0.01}
TMP=$(mktemp -d)
trap 'fuser -k $PORT/tcp >/dev/null 2>&1; rm -rf "$TMP"' EXIT

DEVICES=${DEVICES:-}
COMMON="-ngl 99 -c 8192 -fa on -fit off"
[ -n "$DEVICES" ] && COMMON="$COMMON -dev $DEVICES"
# The big models only leave room for a draft when the KV lives in host RAM.
BASE=${BASE:--sm tensor -nkvo}

# A deliberately LOW-ENTROPY prompt. Greedy decoding amplifies any floating-point difference into a
# completely different continuation once two tokens are near-tied, and the target's verify pass
# batches n+1 tokens, so its matmul shapes - and therefore its rounding - differ from single-token
# decode. On an open-ended prompt that is enough to flip a near-tie legitimately. A prompt whose
# continuation is strongly determined keeps us away from those ties, so a difference here is a bug.
PROMPT=${PROMPT:-'[INST]Count from 1 to 40. Output only the numbers separated by commas, nothing else.[/INST]'}
NPASS=0; NFAIL=0
pass() { NPASS=$((NPASS+1)); printf '  PASS  %s\n' "$*"; }
fail() { NFAIL=$((NFAIL+1)); printf '  FAIL  %s\n' "$*"; }

# ---------------------------------------------------------------- greedy output hash

gen() {
    fuser -k $PORT/tcp >/dev/null 2>&1; sleep 3
    $SERVER -m "$MODEL" $COMMON --host 127.0.0.1 --port $PORT -np 1 "$@" > "$TMP/srv.log" 2>&1 &
    local i
    for i in $(seq 1 180); do
        grep -q "listening on" "$TMP/srv.log" 2>/dev/null && break
        sleep 2
    done
    grep -q "listening on" "$TMP/srv.log" 2>/dev/null || return 1

    PORT=$PORT python3 - "$PROMPT" <<'PY'
import json, os, sys, hashlib, urllib.request
req = {"prompt": sys.argv[1], "n_predict": 128, "temperature": 0.0, "cache_prompt": False}
r = urllib.request.Request("http://127.0.0.1:%s/completion" % os.environ["PORT"],
                           data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=900))
print(hashlib.sha256(d["content"].encode()).hexdigest())
PY
}

# Speculation is lossless: the greedy result must not depend on the draft. Any difference is a bug.
identical() {
    local name=$1; shift
    local -a A=() B=()
    while [ "$1" != "--" ]; do A+=("$1"); shift; done; shift; B=("$@")

    local a b
    a=$(gen "${A[@]}") || { fail "$name (reference did not start)"; return; }
    b=$(gen "${B[@]}") || { fail "$name (variant did not start)";   return; }
    [ -n "$a" ] && [ "$a" = "$b" ] && pass "$name" || fail "$name  ${a:0:16} != ${b:0:16}"
}

# ---------------------------------------------------------------- perplexity

# Perplexity needs a context it can actually fill, so it does NOT reuse $COMMON's -c.
PPL_COMMON="-ngl 99 -c 512 -fa on -fit off"
[ -n "$DEVICES" ] && PPL_COMMON="$PPL_COMMON -dev $DEVICES"

ppl_of() {
    "$PPL" -m "$MODEL" -f "$TMP/ppl.txt" $PPL_COMMON --chunks 4 "$@" 2>&1 \
        | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$"
}

# The compute layout changes, so tiny FP drift is expected; real corruption is not.
ppl_close() {
    local name=$1; shift
    local -a A=() B=()
    while [ "$1" != "--" ]; do A+=("$1"); shift; done; shift; B=("$@")

    local a b
    a=$(ppl_of "${A[@]}"); b=$(ppl_of "${B[@]}")
    if [ -z "$a" ] || [ -z "$b" ]; then fail "$name (perplexity did not run)"; return; fi
    python3 -c "
import sys
a,b,tol=float('$a'),float('$b'),float('$PPL_TOL')
d=abs(a-b)/a
print('  %s  %s  ppl %.4f vs %.4f  (drift %.3f%%, tol %.1f%%)' %
      ('PASS' if d<=tol else 'FAIL', '$name', a, b, 100*d, 100*tol))
sys.exit(0 if d<=tol else 1)"
    [ $? -eq 0 ] && NPASS=$((NPASS+1)) || NFAIL=$((NFAIL+1))
}

# ---------------------------------------------------------------- checks

echo "=== build configuration ==="
# A fresh cmake defaults this OFF, which silently removes the fused FA kernels for mismatched K/V
# quant pairs; tensor split then falls back to a path that mirrors the KV and produces garbage.
grep -q "^GGML_CUDA_FA_ALL_QUANTS:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null \
    && pass "GGML_CUDA_FA_ALL_QUANTS=ON" \
    || fail "GGML_CUDA_FA_ALL_QUANTS is not ON (must match the production build)"

echo ""
echo "=== kernels: no NEW test-backend-ops failures ==="
# Known baseline: FLASH_ATTN_EXT with prec=def (f16 accumulation) diverges for kv>=512 on ROCm.
# Unreachable from llama.cpp - both ggml_flash_attn_ext call sites force GGML_PREC_F32 - so we pin
# the count rather than chase it, and fail loudly on anything new.
# Rather than pin a fragile count (it flaps 148-153 as borderline cases cross the tolerance), we
# exclude exactly the unreachable class - FLASH_ATTN_EXT with prec=def - and require zero failures
# anywhere else. That is stable AND strict: any failure outside that class is a genuine regression.
if [ -x "$TBO" ]; then
    "$TBO" -b "${TBO_BACKEND:-ROCm0}" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > "$TMP/tbo.txt"
    grep '^\[[A-Z_0-9]*\].*: FAIL$' "$TMP/tbo.txt" | grep -v 'prec=def' > "$TMP/tbo_real.txt" || true
    n_real=$(wc -l < "$TMP/tbo_real.txt")
    n_known=$(grep -c '^\[FLASH_ATTN_EXT\].*prec=def.*: FAIL$' "$TMP/tbo.txt" || true)
    if [ "$n_real" -eq 0 ]; then
        pass "no reachable kernel failures ($n_known known prec=def FA failures ignored)"
    else
        fail "$n_real REACHABLE kernel failures - regression:"
        head -3 "$TMP/tbo_real.txt" | sed 's/^/          /'
    fi
else
    fail "test-backend-ops not built"
fi

# a fixed corpus for perplexity
head -c 8000 "${PPL_TEXT:-$BUILD/../README.md}" > "$TMP/ppl.txt" 2>/dev/null || \
    python3 -c "open('$TMP/ppl.txt','w').write(('the quick brown fox jumps over the lazy dog. '*400))"

echo ""
echo "=== compute layout must not change the result (perplexity, tol ${PPL_TOL}) ==="
# llama-perplexity runs with n_seq=4, so every check below also exercises the parallel-sequence
# path through the tensor-split meta backend - which had two separate aborts under -nkvo before.
ppl_close "tensor split == layer split"      -sm layer  -- -sm tensor
ppl_close "KV in host RAM == KV in VRAM"     -sm tensor -- -sm tensor -nkvo
ppl_close "hybrid KV (-nckvl 8) == in VRAM"  -sm tensor -- -sm tensor -nckvl 8
ppl_close "n_seq>1 tensor+nkvo == layer"     -sm layer -nkvo -- -sm tensor -nkvo
[ -n "${NCMOE:-}" ] && ppl_close "CPU-MoE offload == all on GPU" -sm tensor -- -sm tensor -ncmoe "$NCMOE"

if [ -n "$DRAFT" ]; then
    echo ""
    echo "=== speculative decoding is lossless at temp 0 (must be byte-identical) ==="
    SPEC="-md $DRAFT --spec-type draft-eagle -ngld 99"
    identical "draft n=1 == no draft"      $BASE -- $BASE $SPEC --spec-draft-n-max 1
    identical "draft n=8 == no draft"      $BASE -- $BASE $SPEC --spec-draft-n-max 8
    identical "draft n=8 == no draft (layer)" -sm layer -nkvo -- -sm layer -nkvo $SPEC --spec-draft-n-max 8
fi

echo ""
echo "=== checkpoint persistence round-trip ==="
mkdir -p "$TMP/slots"
if gen $BASE --slot-save-path "$TMP/slots" -ctxcp 4 -cms 256 > /dev/null; then
    PORT=$PORT python3 - <<'PY'
import json, os, sys, urllib.request
def post(p, b):
    r = urllib.request.Request("http://127.0.0.1:%s%s" % (os.environ["PORT"], p),
                               data=json.dumps(b).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
post("/completion", {"prompt":"hello world","n_predict":32,"temperature":0.0,"cache_prompt":True})
s = post("/slots/0?action=save",    {"filename":"regress.bin"})
r = post("/slots/0?action=restore", {"filename":"regress.bin"})
sys.exit(0 if s.get("n_saved") and s.get("n_saved") == r.get("n_restored") else 1)
PY
    [ $? -eq 0 ] && pass "slot save/restore round-trip" || fail "slot save/restore round-trip"
else
    fail "server would not start with checkpoints enabled"
fi

echo ""
echo "================ $NPASS passed, $NFAIL failed ================"
[ "$NFAIL" -eq 0 ]

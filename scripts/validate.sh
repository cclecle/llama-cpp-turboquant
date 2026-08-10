#!/usr/bin/env bash
#
# Tiered validation for this fork, on the 2x R9700 (gfx1201) box.
#
# Why this exists: our patches live in shared code paths, so "the model it was written for still
# works" is not evidence. Each tier below is chosen to exercise patches through models that were
# never the reason for the patch.
#
#   Tier 0  SMOKE  ~25 min   run after every rebuild
#   Tier 1  GATE   ~50 min   must be green before Tier 2
#   Tier 2  DEEP   ~2-3 h    before deployment only
#
# Usage:
#   scripts/validate.sh --tier 0|1|2 --build <dir> [--baseline] [--models a,b,c]
#
#   --build <dir>   build directory to test (e.g. /tmp/llama-sync/build)
#   --baseline      record results as the reference instead of comparing against it.
#                   Point --build at the PRODUCTION build to capture the reference.
#   --presets <f>   preset ini to use (default: the stripped copies under validation/presets)
#
# NEVER run against /opt directly for anything that writes. Production is the systemd units
# llamacpp-ha / llamacpp-both / llamacpp-unified; they hold the GPUs. See the VRAM guard below.
#
set -uo pipefail

TIER=0
BUILD=""
BASELINE=0
MODELS=""
PRESET_DIR="$(cd "$(dirname "$0")/.." && pwd)/validation/presets"
REF="$(cd "$(dirname "$0")/.." && pwd)/validation/reference.json"
OUT="${OUT:-/tmp/validation-$(date +%Y%m%d-%H%M%S)}"

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)     TIER=$2; shift 2 ;;
        --build)    BUILD=$2; shift 2 ;;
        --baseline) BASELINE=1; shift ;;
        --models)   MODELS=$2; shift 2 ;;
        --presets)  PRESET_DIR=$2; shift 2 ;;
        --out)      OUT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$BUILD" ] || { echo "--build is required" >&2; exit 2; }
mkdir -p "$OUT"

SERVER=$BUILD/bin/llama-server
BENCH=$BUILD/bin/llama-bench
PPL=$BUILD/bin/llama-perplexity
TBO=$BUILD/bin/test-backend-ops
PORT=${PORT:-20099}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}   # hides the iGPU; it corrupts tensor split

NPASS=0; NFAIL=0; NSKIP=0
pass() { NPASS=$((NPASS+1)); printf '  PASS  %s\n' "$*"; }
fail() { NFAIL=$((NFAIL+1)); printf '  FAIL  %s\n' "$*"; }
skip() { NSKIP=$((NSKIP+1)); printf '  SKIP  %s\n' "$*"; }
head1() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- guards

# Production holds ~32 GB on each card. Starting a model on top of that OOMs and, worse, an OOM
# here looks identical to a real regression. Refuse rather than produce a misleading failure.
require_free_vram() {
    # Poll rather than sample once: after a server is stopped the driver takes a little while to
    # release, and a single early sample makes the NEXT model look like it does not fit - which
    # reads as a VRAM regression when it is only a race.
    local need_gb=$1 wait_s=${2:-180} i free_gb ok t0
    t0=$(date +%s)
    while :; do
        ok=1
        for i in 0 1; do
            free_gb=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
                | awk -F, -v d="card$i" '$1 ~ d {print int(($2-$3)/1073741824)}' | head -1)
            [ -z "$free_gb" ] && free_gb=0
            [ "$free_gb" -ge "$need_gb" ] || ok=0
        done
        [ "$ok" -eq 1 ] && { printf '        VRAM ok (>= %s GiB free on both)\n' "$need_gb"; return 0; }
        [ $(( $(date +%s) - t0 )) -ge "$wait_s" ] && break
        sleep 10
    done
    for i in 0 1; do
        free_gb=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
            | awk -F, -v d="card$i" '$1 ~ d {print int(($2-$3)/1073741824)}' | head -1)
        printf '        GPU%d free: %s GiB (need %s) after %ss\n' "$i" "${free_gb:-0}" "$need_gb" "$wait_s"
    done
    return 1
}

# Kill only what we started, by PID, never pkill by name - production must survive.
OUR_PIDS=()
cleanup() {
    local p
    for p in "${OUR_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    sleep 2
    for p in "${OUR_PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT

# ---------------------------------------------------------------- tier 0: smoke

tier0() {
    head1 "T0.1 build configuration"
    if grep -q "^GGML_CUDA_FA_ALL_QUANTS:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null; then
        pass "GGML_CUDA_FA_ALL_QUANTS=ON"
    else
        fail "GGML_CUDA_FA_ALL_QUANTS is not ON (must match production)"
    fi
    # upstream removed the option; if it is still being passed, the build script is stale
    if grep -q "^GGML_HIP_ROCWMMA_FATTN" "$BUILD/CMakeCache.txt" 2>/dev/null; then
        fail "GGML_HIP_ROCWMMA_FATTN present - upstream removed it, drop it from the build script"
    else
        pass "GGML_HIP_ROCWMMA_FATTN absent (correct after the rebase)"
    fi

    head1 "T0.2 every preset section parses"
    # The cheap catch for the whole 'a flag was removed upstream' class of breakage - exactly how
    # --rocwmma-fattn would have taken the router down on first start.
    #
    # There is no --list-models flag. The parse happens in server_models::load_models(), called
    # from the constructor at startup (tools/server/server-models.cpp:426), and an unknown key
    # throws out of it. So simply reaching /health with --no-models-autoload proves every section
    # parsed - and loads no weights, so it costs no VRAM and a couple of seconds.
    local ini base lg pid i ready
    for ini in "$PRESET_DIR"/*.ini; do
        [ -e "$ini" ] || continue
        base=$(basename "$ini"); lg="$OUT/parse-$base.log"
        "$SERVER" --host 127.0.0.1 --port "$PORT" --models-preset "$ini" \
                  --no-models-autoload --models-max 1 >"$lg" 2>&1 &
        pid=$!; OUR_PIDS+=("$pid")
        ready=0
        # no curl on this box; python3 stdlib only
        for i in $(seq 1 60); do
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
            if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=3).read()" 2>/dev/null; then ready=1; break; fi
            sleep 1
        done
        if [ "$ready" = 1 ]; then
            python3 -c "import urllib.request; open('$OUT/models-$base.json','wb').write(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=10).read())" 2>/dev/null
            pass "parsed $base ($(grep -o '\"id\"' "$OUT/models-$base.json" 2>/dev/null | wc -l) sections)"
        else
            fail "parse failed: $base - $(grep -oE "option '[^']+' not recognized in preset '[^']+'" "$lg" | head -1)"
        fi
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    done

    head1 "T0.3 kernels (test-backend-ops)"
    # SKIP_TBO=1 during script shake-out only; it is ~20-40 min and must run for a real gate.
    if [ "${SKIP_TBO:-0}" = 1 ]; then
        skip "test-backend-ops (SKIP_TBO=1)"
    else
    # Known-unreachable baseline: FLASH_ATTN_EXT with prec=def (f16 accumulation). Both call sites
    # in llama.cpp force GGML_PREC_F32, so those can never fire in practice. Everything else must
    # be clean - excluding the class rather than pinning a flaky count keeps this strict.
    if [ -x "$TBO" ]; then
        timeout 3600 "$TBO" -b "${TBO_BACKEND:-ROCm0}" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > "$OUT/tbo.txt"
        grep '^\[[A-Z_0-9]*\].*: FAIL$' "$OUT/tbo.txt" | grep -v 'prec=def' > "$OUT/tbo_real.txt" || true
        local n_real n_known
        n_real=$(wc -l < "$OUT/tbo_real.txt")
        n_known=$(grep -c 'prec=def.*: FAIL$' "$OUT/tbo.txt" || true)
        if [ "$n_real" -eq 0 ]; then
            pass "no reachable kernel failures ($n_known known prec=def ignored)"
        else
            fail "$n_real reachable kernel failures:"
            head -5 "$OUT/tbo_real.txt" | sed 's/^/          /'
        fi
    else
        fail "test-backend-ops not built (needs -DLLAMA_BUILD_TESTS=ON)"
    fi
    fi

    head1 "T0.4 the two heaviest presets load and generate"
    # Mistral-Medium and Qwen3.5-122B between them cover most of the wide-blast-radius patches:
    # pinned host KV (-nkvo), CPU-resident experts under tensor split, EAGLE and MTP draft
    # checkpointing, P2P AllReduce, and the scatter path.
    #
    # Loaded at PRODUCTION context deliberately. Mistral-Medium is a very tight VRAM fit and the
    # draft-KV-in-full patch makes checkpoints larger, so a VRAM regression from it surfaces here
    # as an OOM and nowhere else. Shrinking the context to save time would hide the failure this
    # test exists to catch.
    smoke_preset "Mistral-Medium-3.5-128B"
    smoke_preset "Qwen3.5-122B-A10B"
}

# Load one preset section faithfully, ask for a few tokens, unload. Sequential: they do not fit
# together, and neither fits beside production.
smoke_preset() {
    local name=$1
    if ! require_free_vram 30; then
        skip "$name (insufficient free VRAM - production is loaded; stop it or wait for idle unload)"
        return
    fi
    python3 "$(dirname "$0")/validate-perf.py" smoke \
        --build "$BUILD" --presets "$PRESET_DIR" --model "$name" --out "$OUT" \
        && pass "$name loaded and generated" \
        || fail "$name failed to load or generate (see $OUT/$name.log)"
}

# ---------------------------------------------------------------- tier 1: gate

tier1() {
    tier0
    head1 "T1 equivalence and perf"
    echo "  (delegated to fork-regress.sh for equivalence, validate-perf.py for numbers)"
    bash "$(dirname "$0")/fork-regress.sh" "$BUILD" "${T1_MODEL:?set T1_MODEL}" "${T1_DRAFT:-}" \
        | tee "$OUT/fork-regress.log"
    [ ${PIPESTATUS[0]} -eq 0 ] && pass "fork-regress equivalence suite" || fail "fork-regress equivalence suite"

    python3 "$(dirname "$0")/validate-perf.py" gate \
        --build "$BUILD" --presets "$PRESET_DIR" --out "$OUT" --ref "$REF" \
        ${MODELS:+--models "$MODELS"} $([ "$BASELINE" = 1 ] && echo --baseline) \
        && pass "perf within tolerance" || fail "perf regression (see $OUT/perf-report.txt)"
}

# ---------------------------------------------------------------- tier 2: deep

tier2() {
    tier1
    head1 "T2 full fleet sweep + soak"
    python3 "$(dirname "$0")/validate-perf.py" deep \
        --build "$BUILD" --presets "$PRESET_DIR" --out "$OUT" --ref "$REF" \
        ${MODELS:+--models "$MODELS"} $([ "$BASELINE" = 1 ] && echo --baseline) \
        && pass "deep sweep within tolerance" || fail "deep sweep regression"
}

# ---------------------------------------------------------------- main

echo "build:   $BUILD"
echo "tier:    $TIER"
echo "presets: $PRESET_DIR"
echo "output:  $OUT"
echo "mode:    $([ "$BASELINE" = 1 ] && echo 'BASELINE (recording reference)' || echo 'COMPARE against reference')"

case "$TIER" in
    0) tier0 ;;
    1) tier1 ;;
    2) tier2 ;;
    *) echo "bad tier: $TIER" >&2; exit 2 ;;
esac

echo ""
echo "================ $NPASS passed, $NFAIL failed, $NSKIP skipped ================"
echo "artifacts: $OUT"
[ "$NFAIL" -eq 0 ]

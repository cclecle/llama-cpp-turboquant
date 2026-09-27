#!/bin/bash
# rung-m128k.sh <tag> [env...] : one load of the Flash-Next :M candidate (131072 ctx, 1 slot) on the v19 build, the
# 32k-token prompt, peak VRAM per card. RM_ARGS: args file (default fn-m128k.args), RM_UB: ubatch, RM_LOAD_ONLY=1:
# load only (read the peak), RM_PROMPT: prompt file. Production stays stopped.
cd /mnt/gguf/r9v/bench
tag=$1; shift
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
FN_CTX=131072 FN_UB=${RM_UB:-} FN_LOAD_ONLY=${RM_LOAD_ONLY:-0} FN_ENV="${*:-X=0}" FN_ARGS=${RM_ARGS:-fn-m128k.args} FN_TAG=-m128k-$tag \
  bash flashnext.sh v19dev 2>&1 | grep -E 'load:|memory:|died'
[ "${RM_LOAD_ONLY:-0}" = 1 ] || python3 step_ms.py v19dev-m128k-$tag results.jsonl

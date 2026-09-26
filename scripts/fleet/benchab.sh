#!/bin/bash
# benchab.sh <tag> <HIP devices> <release A> <release B> -- <llama-bench args> : same llama-bench run on two builds,
# prints the pp/tg columns per depth on one line per build. Env for one arm: ENV_A / ENV_B (e.g. ENV_B="X=1 Y=2").
# Production must be stopped by the caller.
set -u
TAG=$1; DEV=$2; RA=$3; RB=$4; shift 4; [ "${1:-}" = "--" ] && shift
for ARM in A B; do
  # pick by arm, not by name: A and B may be the same build with different env
  if [ $ARM = A ]; then V=$RA; E=${ENV_A:-}; else V=$RB; E=${ENV_B:-}; fi
  B=/opt/llamacpp/llama-cpp-mine-$V/build3/bin
  r=$(env $E HIP_VISIBLE_DEVICES=$DEV LD_LIBRARY_PATH=$B timeout 3600 $B/llama-bench "$@" 2>&1 | grep -E '\| +(pp|tg)[0-9]' | awk -F'|' '{print $(NF-2), $(NF-1)}' | tr -s ' ' | tr '\n' ';')
  echo "$TAG $V${E:+ [$E]}: ${r:-FAILED}"
done

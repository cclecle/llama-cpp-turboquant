#!/bin/bash
# build a scratch tree with the production options (copied from scripts/fleet/build-release.sh)
# usage: build-scratch.sh [tree dir, default /opt/llamacpp/tmp-rdnab]
set -eu
cd "${1:-/opt/llamacpp/tmp-rdnab}"
cmake -B build3 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_CUDA_NCCL=ON -DGGML_SCHED_MAX_COPIES=4 -DGGML_CCACHE=ON -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=ON > configure.log 2>&1
nice -n 10 cmake --build build3 --config Release -j 20 > build.log 2>&1 || true
echo "errors: $(grep -cE 'error:|\*\*\* \[' build.log)"
grep -E 'error:' build.log | head -30
ls -la build3/bin/llama-server 2>&1

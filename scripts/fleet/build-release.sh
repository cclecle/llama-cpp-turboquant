#!/bin/bash
# build-release.sh <N> : configure and build /opt/llamacpp/llama-cpp-mine-vN/build3 on the box with the production
# options. Unpack the release tarball there first (git archive of the branch). Then compare CMakeCache.txt against the
# running release with a FULL diff - a filtered grep proves nothing.
set -eu
V=${1:?release number}; cd /opt/llamacpp/llama-cpp-mine-v$V
cmake -B build3 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_CUDA_NCCL=ON -DGGML_SCHED_MAX_COPIES=4 -DGGML_CCACHE=ON -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=ON
nice -n 5 cmake --build build3 --config Release -j 24 2>&1 | tee build.log | grep -E 'error:|\*\*\* \[|Built target llama-server' || true
echo "errors: $(grep -cE 'error:|\*\*\* \[' build.log)"

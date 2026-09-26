#!/bin/bash
# build-v19-rocm714.sh : the v19 dev source (/opt/llamacpp/tmp-v19) built against ROCm 7.14, the version R9V runs, with
# the options of validation/v17-scripts/build-scratch.sh (the dev tree's, only the ROCm path differs).
# Tree /opt/llamacpp/llama-cpp-mine-v19-rocm714 (temporary, removed after the campaign); flashnext.sh runs it with
# LD_LIBRARY_PATH=/opt/rocm-7.14/lib (any tree named *rocm714). /opt/rocm-7.14 is a symlink to the core-7.14 tree of
# R9V's unpacked image: copying its 19 GB would evict the page cache (the mmap'd PLE), which is what made the first
# 7.14 measurement cold.
set -eu
C=/opt/rocm-7.14
R=/mnt/gguf/r9v/rootfs/opt/rocm/core-7.14
S=/opt/llamacpp/tmp-v19
T=/opt/llamacpp/llama-cpp-mine-v19-rocm714
[ -e $C ] || ln -s $R $C
mkdir -p $T
# the source only: the dev tree's build dir and logs stay behind
(cd $S && tar cf - --exclude=./build3 --exclude='./*.log' .) | (cd $T && tar xf - --no-same-owner)
echo "$(cd $S && git rev-parse --short HEAD 2>/dev/null || echo tmp-v19) + tmp-v19 files, rocm 7.14" > $T/RELEASE_COMMIT
cd $T
export ROCM_PATH=$C HIP_PATH=$C HIP_PLATFORM=amd PATH=$C/bin:$C/lib/llvm/bin:$PATH
cmake -B build3 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$C/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$C -DROCM_PATH=$C \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_CUDA_NCCL=ON -DGGML_SCHED_MAX_COPIES=4 -DGGML_CCACHE=ON -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=ON > configure.log 2>&1
nice -n 10 cmake --build build3 --config Release -j 20 > build.log 2>&1 || true
echo "errors: $(grep -cE 'error:|\*\*\* \[' build.log)"
grep -E 'error:' build.log | head -20
ls -la build3/bin/llama-server 2>&1
ldd build3/bin/libggml-hip.so | grep -E 'amdhip|hipblas|rocblas|rccl'

#!/bin/bash
# build-rocm714.sh : the v18 source built against ROCm 7.14 (/opt/rocm-7.14, a copy of the core-7.14 tree of R9V's image)
# instead of the host's 7.2.4, same cmake options as production. Tree /opt/llamacpp/llama-cpp-mine-v18-rocm714
# (temporary, removed after the campaign). Tells a ROCm-version effect apart from R9V's own kernels.
set -eu
C=/opt/rocm-7.14
T=/opt/llamacpp/llama-cpp-mine-v18-rocm714
if [ ! -d $T ]; then
  mkdir -p $T.x && tar xzf /opt/llamacpp/llama-cpp-mine-v18.tar.gz -C $T.x && mv $T.x/llama-cpp-mine-v18 $T && rmdir $T.x
  echo "$(cat /opt/llamacpp/llama-cpp-mine-v18/RELEASE_COMMIT) rocm 7.14" > $T/RELEASE_COMMIT
fi
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
readelf -d build3/bin/libggml-hip.so | grep -E 'RUNPATH' || true
ldd build3/bin/libggml-hip.so | grep -E 'amdhip|hipblas|rocblas|rccl'

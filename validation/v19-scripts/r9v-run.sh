#!/bin/bash
# r9v-run.sh [max model len, default 65536] : start R9V (vLLM fork) for Qwen3.8-Flash-Next, profile qwen38-mtp4
# (UD-IQ4_XS, runtime mtp4-v2, WMMA grouped MoE prefill image), as a plain host process: no docker, no chroot.
# The runtime is the image's /opt/r9v venv + its ROCm 7.14, unpacked by scripts/fleet/oci_unpack.py into
# $B/rootfs. The env and the `vllm serve` arguments are R9V scripts/launch.sh (at src aeee44f) transcribed, with
# the container paths replaced by the host ones: /models -> $B/package, /ple/... -> $B/ple/..., /cache -> $B/cache,
# /placement/experts.json -> the profile's manifest. Not done here, as launch.sh does: the preflight doctor
# (it drives docker). CED stays off (runtime has no overlays; CED is approximate prefill). Runs in the foreground.
set -euo pipefail
B=/mnt/gguf/r9v
R=$B/rootfs
SRC=$B/src
RV=$R/opt/r9v
ROCM=$R/opt/rocm/core-7.14
IMAGE=sha256:2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03

export R9V_MAX_MODEL_LEN=${1:-65536}
set -a
# shellcheck disable=SC1091
source "$SRC/profiles/qwen38-flash-next/dual-r9700-mtp4/profile.env"
set +a

model_dir=$B/package
ple_path=$B/ple/per_layer_token_embd.iq4_nl.bin
cache_dir=$B/cache
manifest_path=$R9V_EXPERT_MANIFEST_PATH
mtp=$model_dir/mtp
metadata=$model_dir/metadata
for f in "$model_dir"/target/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf "$metadata/config.json" \
         "$mtp/config.json" "$mtp/model.safetensors" "$model_dir/vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf" \
         "$manifest_path" "$ple_path"; do
  [[ -f $f ]] || { echo "missing: $f" >&2; exit 1; }
done

ns=$(cd "$SRC" && python3 tools/runtime_cache_key.py "$IMAGE" "$manifest_path")
root=$cache_dir/vllm/$ns
mkdir -p "$root"
echo "runtime compile cache: $root"

local_argmax=false; [[ $R9V_MTP_LOCAL_ARGMAX == 0 ]] || local_argmax=true
auto_tool=(); [[ $R9V_ENABLE_AUTO_TOOL_CHOICE == 0 ]] || auto_tool=(--enable-auto-tool-choice)
prefix=(--enable-prefix-caching); [[ ${R9V_ENABLE_PREFIX_CACHING:-1} == 1 ]] || prefix=(--no-enable-prefix-caching)
sizes=$(python3 -c "import json,sys; d=int(sys.argv[1]); s=sorted({1,d+1}); print(json.dumps({'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':s,'max_cudagraph_capture_size':max(s)}))" "$R9V_MTP_SPEC_TOKENS")
case $R9V_PLE_RESIDENCY_MODE in pinned) ple_reg=1 ;; *) ple_reg=0 ;; esac

# image config Env, /opt/r9v and /opt/rocm moved to the unpacked copies
export PATH=$RV/bin:$ROCM/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=$ROCM/lib
export ROCM_PATH=$ROCM HIP_PATH=$ROCM VIRTUAL_ENV=$RV HIP_PLATFORM=amd VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1201 HIP_ARCHITECTURES=gfx1201 AMDGPU_TARGETS=gfx1201 GPU_ARCHS=gfx1201
export HSA_NO_SCRATCH_RECLAIM=1 SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=4
K=$RV/kernels/retained-mtp4
export QWEN38_DENSE_MMVQ_HIP_SO=$K/qwen38_dense_mmvq_hip.so QWEN38_TIERED_IQ_MOE_HIP_SO=$K/qwen38_tiered_iq_moe_hip.so
export QWEN38_FUSED_GDN_MTP_HIP_SO=$RV/kernels/qwen38_fused_gdn_mtp_hip.so
export R9V_HC_MMQ4_SO=$K/r9v_hc_mmq4.so R9V_HC_MIX_MMQ4_SO=$K/r9v_hc_mixmmq4.so R9V_HC_MIX_MMQ45_SO=$K/r9v_hc_mixmmq45.so
export R9V_CACHE80_SO=$K/r9v_cache80.so R9V_CACHE192_SO=$K/r9v_cache192.so R9V_Q8_MMVQ5_SO=$K/r9v_q8_mmvq5.so
export R9V_DRAFT_W2_RANK_SO=$K/r9v_draft_w2_rank.so R9V_DRAFT_INDEXED_SO=$K/r9v_draft_indexed_q6.so R9V_MOE_WMMA_SO=$K/r9v_moe_wmma.so
export PYTHONPATH=$RV/retained-mtp4/python

# launch.sh --env set
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export QWEN38_EXPERT_TP_SPLIT=$R9V_EXPERT_TP_SPLIT
export R9V_RETAINED_MTP4 R9V_DRAFT_W2_RANK_ROWS R9V_CACHE_FILL_BATCH R9V_CACHE80 R9V_CACHE192 R9V_Q6K_HEAD_ROWS5 \
  R9V_DENSE_ROWS45 R9V_DENSE_ROWS2 R9V_HC_ROWS45 R9V_HC_MMQ4 R9V_HC_MIX_MMQ4 R9V_HC_MIX_MMQ45 R9V_HC_SAFE_PADDING \
  R9V_ATTENTION_OUTPUT4 R9V_SHARED_SAFE_PADDING R9V_NATIVE_CUSTOM_AR R9V_NATIVE_ALIGN_IPC R9V_Q8_MMVQ5_SHAPES
export VLLM_ROCM_USE_AITER_CUSTOM_AR=0 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=$R9V_VISIBLE_DEVICES
export VLLM_CACHE_ROOT=$root TRITON_CACHE_DIR=$root/bootstrap/triton TRITON_CACHE_AUTOTUNING=1
export TORCHINDUCTOR_CACHE_DIR=$root/bootstrap/inductor TORCH_EXTENSIONS_DIR=$root/extensions
export RADIANCE_CPU_OFFLOAD_GB_BY_DEVICE=$R9V_CPU_OFFLOAD_GB_BY_DEVICE R9V_CPU_OFFLOAD_GB_BY_DEVICE
export RADIANCE_TIERED_EXPERT_MANIFEST=$manifest_path
export RADIANCE_UVA_HOST_COHERENCE=default RADIANCE_UVA_HOST_NONCOHERENT=0
export RADIANCE_USE_R4D=0 RADIANCE_USE_R4D_AR=0 RADIANCE_USE_R4D_GDN=0 RADIANCE_USE_R4D_AR_QUANT=0
export QWEN38_USE_TIERED_IQ_MOE_HIP=1 QWEN38_TIERED_IQ_MOE_VARIANT=$R9V_TIERED_IQ_MOE_VARIANT
export QWEN38_TIERED_PREFILL_GROUP_SIZE=$R9V_TIERED_PREFILL_GROUP_SIZE
export QWEN38_TIERED_EXPERT_CACHE_SLOTS=$R9V_TIERED_EXPERT_CACHE_SLOTS QWEN38_TIERED_EXPERT_CACHE_RANKS=$R9V_TIERED_EXPERT_CACHE_RANKS
export QWEN38_TIERED_EXPERT_CACHE_POLICY=$R9V_TIERED_EXPERT_CACHE_POLICY QWEN38_TIERED_EXPERT_CACHE_ASYNC=$R9V_TIERED_EXPERT_CACHE_ASYNC
export QWEN38_TIERED_STREAM_COMPACTION=${R9V_STREAM_EXPERT_COMPACTION:-0}
export QWEN38_USE_DENSE_MMVQ_HIP=1 QWEN38_USE_DENSE_MMVQ_REUSE2=1 QWEN38_USE_DENSE_MMVQ_Q8_REUSE2=1
export QWEN38_USE_DENSE_MMVQ_REUSE3=1 QWEN38_USE_DENSE_MMVQ_REUSE4=0
export QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3=$R9V_ENABLE_DENSE_Q8_ATTN_M3 QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT=$R9V_DENSE_Q8_ATTN_M3_VARIANT
export QWEN38_USE_DENSE_HC_DOWN_BF16_M3=$R9V_ENABLE_DENSE_HC_DOWN_BF16_M3 QWEN38_FUSED_HC_UP_MIX=$R9V_ENABLE_FUSED_HC_UP_MIX
export VLLM_GGUF_FUSED_MOE_SHARED_EPILOGUE=$R9V_ENABLE_FUSED_MOE_SHARED_EPILOGUE QWEN38_USE_HIP_FUSED_GDN_MTP=$R9V_ENABLE_FUSED_GDN_MTP
export VLLM_QWEN4_EXP_RDNA4_QSA_STRIDED=$R9V_ENABLE_RDNA4_QSA_STRIDED
export VLLM_GGUF_NATIVE_SAFE_MOE_IDS=1 VLLM_GGUF_QWEN4_EXP_MULTIMODAL=1 VLLM_QWEN4_EXP_MTP_FP8_EXPERT_ONLY=0
export VLLM_QWEN4_EXP_MTP_FUSED_FC_GATHER=0 VLLM_KV_CACHE_LAYOUT=BLHNC VLLM_ROCM_MOE_PADDING=0 NCCL_ALGO=Ring NCCL_PROTO=Simple
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0 VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_PLE_CPU_OFFLOAD=1 R9V_PLE_HOST_FENCE=${R9V_PLE_HOST_FENCE:-1} VLLM_PLE_RESIDENCY_MODE=$R9V_PLE_RESIDENCY_MODE
export VLLM_PLE_MMAP_HOST_REGISTER=$ple_reg VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES=28800138240
export VLLM_PLE_PINNED_RESERVE_BYTES=$R9V_PLE_PINNED_RESERVE_BYTES VLLM_PLE_BOUNDED_BYTES=4294967296 VLLM_PLE_BOUNDED_CHUNK_BYTES=4096
export VLLM_PLE_MMAP_READAHEAD=$R9V_PLE_MMAP_READAHEAD VLLM_PLE_RSS_LOG_ROWS=$R9V_PLE_RSS_LOG_ROWS VLLM_PLE_WORKER_TIMING=$R9V_PLE_WORKER_TIMING
export R9V_WORKER_DIAGNOSTICS=1 R9V_CONTAINER_NAME=r9v-qwen38-flash-next R9V_OBSERVABILITY_TARGET=
export R9V_STAGE_DIAGNOSTICS=${R9V_STAGE_DIAGNOSTICS:-0} R9V_EXPECTED_GPU_BDFS=${R9V_EXPECTED_GPU_BDFS:-}
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=0 QWEN38_PROFILE_DENSE_SHAPES=0
# R9V_TORCH_PROFILE_DIR: vLLM's torch profiler (launch.sh's R9V_PROFILER_DIR, without stacks or shapes), started and
# stopped with POST /start_profile and /stop_profile; each worker writes a Chrome trace there
prof=()
if [ -n "${R9V_TORCH_PROFILE_DIR:-}" ]; then
  mkdir -p "$R9V_TORCH_PROFILE_DIR"
  prof=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R9V_TORCH_PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"wait_iterations\":0,\"warmup_iterations\":1,\"active_iterations\":${R9V_TORCH_PROFILE_STEPS:-12},\"max_iterations\":${R9V_TORCH_PROFILE_STEPS:-12}}")
fi
export GGUF_PLE_MMAP_PATH=$ple_path GGUF_PLE_MMAP_TRIM_ROWS=$R9V_PLE_MMAP_TRIM_ROWS

# docker ran it with --ulimit memlock=-1. This rig is a Proxmox container capped at 8 MiB locked memory even for
# systemd units (LimitMEMLOCK=infinity is not honoured), and llama.cpp runs under the same cap: GPU-driver pinning
# (hipHostMalloc / hipHostRegister) is not charged to RLIMIT_MEMLOCK. Logged, not enforced.
echo "locked memory limit: $(ulimit -l)"
cd "$B"
exec "$RV/bin/python" -m vllm.entrypoints.cli.main serve \
  "$model_dir/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" \
  --tokenizer "$metadata" --hf-config-path "$metadata" --served-model-name "$R9V_SERVED_MODEL_NAME" \
  "${prefix[@]}" --load-format gguf --quantization gguf \
  --tensor-parallel-size "$R9V_TENSOR_PARALLEL_SIZE" --pipeline-parallel-size 1 \
  --cpu-offload-gb "$R9V_CPU_OFFLOAD_GB" --cpu-offload-params experts \
  --kv-cache-memory-bytes "$R9V_KV_CACHE_MEMORY_BYTES" \
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$mtp\",\"num_speculative_tokens\":$R9V_MTP_SPEC_TOKENS,\"draft_tensor_parallel_size\":$R9V_MTP_DRAFT_TP_SIZE,\"quantization\":\"$R9V_MTP_QUANTIZATION\",\"use_local_argmax_reduction\":$local_argmax,\"draft_load_config\":{\"load_format\":\"auto\"}}" \
  --max-model-len "$R9V_MAX_MODEL_LEN" --max-num-seqs "$R9V_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$R9V_MAX_NUM_BATCHED_TOKENS" --compilation-config "$sizes" \
  --model-loader-extra-config "{\"mm_proj\":\"$model_dir/vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf\"}" \
  --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":262144}' \
  --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights \
  "${auto_tool[@]}" --tool-call-parser "$R9V_TOOL_CALL_PARSER" --reasoning-parser "$R9V_REASONING_PARSER" \
  "${prof[@]}" --trust-remote-code --host 127.0.0.1 --port "$R9V_HOST_PORT"

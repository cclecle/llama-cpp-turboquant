#pragma once

#include "common.cuh"

// Tiered MoE expert tensors: buffer type <device>_TIERED (ROCm0_TIERED, ...), selected per tensor with -ot.
//
// Each expert of a [ne0, ne1, n_expert] quantized weight lives in exactly one place: the hot experts (listed per
// layer in the file named by GGML_CUDA_MOE_HOT_FILE, lines "blk.<layer> <id> <id> ...") compacted in VRAM, all
// others compacted in pinned host memory mapped into the device address space and read in place over the bus.
// A device-side table of the n_expert base addresses follows the hot experts in the tensor's VRAM allocation;
// the MUL_MAT_ID kernels read it instead of computing data + expert*nb[2]. The addresses never change after
// load, so CUDA/HIP graphs stay valid. GGML_CUDA_MOE_TIERED=0 makes every expert cold (the same placement as the
// _UVA buffer type), for an A/B with identical arguments. Under -sm tensor each device shard keeps all experts
// (the expert dim is never split), so the same hot list applies to every shard.

ggml_backend_buffer_type_t ggml_backend_cuda_tiered_buffer_type(int device);
bool ggml_backend_buft_is_cuda_tiered(ggml_backend_buffer_type_t buft);

// Device table of per-expert base addresses of a tiered expert tensor; nullptr for every other tensor.
const void * const * ggml_cuda_tiered_table(const ggml_tensor * t);

// Device address of one expert (host copy of the table), for the synchronous MUL_MAT_ID fallback.
const void * ggml_cuda_tiered_expert(const ggml_tensor * t, int64_t expert);

// 1 if the expert is in VRAM, 0 if it is in mapped host memory, -1 if t is not a tiered expert tensor
int ggml_cuda_tiered_is_hot(const ggml_tensor * t, int64_t expert);

// Prefill staging (GGML_CUDA_MOE_STAGE=1: on, off by default; GGML_CUDA_MOE_STAGE_MIN_TOKENS: the smallest ubatch,
// 256; GGML_CUDA_MOE_STAGE_AREAS: staging areas, 2). The cold experts stay in host memory; for a MUL_MAT_ID of at least
// that many tokens, the cold experts of the next layers are copied on a copy stream into VRAM staging areas (each the
// largest layer's cold bytes) while the current layer computes, and those layers' expert matmuls read them there.
// It loses on 2x R9700 (v19 study, step 28): the copies take all cold experts, twice the bytes of the in-place reads
// of the routed ones, and while they fill the PCIe link, kernel dispatch and the all-reduce that cross it slow down.
// The table to read for a MUL_MAT_ID on t with n_tokens tokens on stream: the staged one when t's layer is in the
// staging area (stream waits for the copy), else ggml_cuda_tiered_table(t).
const void * const * ggml_cuda_tiered_table_prefill(const ggml_tensor * t, int64_t n_tokens, cudaStream_t stream);
// Call after launching every MUL_MAT_ID on a tiered t: after the down projection (the layer's last expert matmul),
// the staging area is refilled with the next layer's cold experts once that matmul is done.
void ggml_cuda_tiered_prefill_launched(const ggml_tensor * t, int64_t n_tokens, cudaStream_t stream);

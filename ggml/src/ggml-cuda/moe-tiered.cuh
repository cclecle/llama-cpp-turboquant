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

#pragma once

#include "common.cuh"

// Float-dequantising mat-vec for K-quants at batch size 1 on RDNA4.
//
// The stock path (mmvq) quantises the activations to q8_1 and uses integer dot products. That is
// the right trade for the simple quants - ROCm beats Vulkan on q4_0/q8_0 - but it loses badly on
// the multi-level K-quant layouts, where vec_dot_q6_K_q8_1 covers only 8 weights per call and
// issues ~6 small, partly strided loads to do it. Measured on gfx1201 (MUL_MAT_ID, n=1, m=768,
// k=2048, 8 of 128 experts): ROCm 25.34 us, Vulkan's float shader 18.07 us, this kernel 14.58 us.
//
// Returns true if ggml_cuda_mul_mat_vec_k can serve this case.
bool ggml_cuda_should_use_mmvk(enum ggml_type type, int cc, int64_t ncols_dst);

void ggml_cuda_mul_mat_vec_k(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_device & fusion);

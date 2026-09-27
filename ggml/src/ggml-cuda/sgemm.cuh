#pragma once

#include "common.cuh"

// F32 x F32 matrix multiplication for batches past the mat-vec kernels on RDNA4, where hipBLASLt runs these shapes on
// 8x8 and 16x16 macro tiles (the qwen4exp router, 512 x 2560 x 1024, at 2.7 TFLOPS). GGML_CUDA_SGEMM=0: hipBLAS.
bool ggml_cuda_should_use_sgemm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_sgemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

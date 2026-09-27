#pragma once

#include "common.cuh"

// A greedy draft head (GGML_HINT_ARGMAX_ONLY) through a Q6_K output matrix, one column: every row is ranked with 2 of
// its Q6_K blocks (a partial dot product over 512 of the inputs), the best rows of the ranking get the exact dot
// product, and every other row gets -INFINITY. Under a tensor split each device ranks its own rows, so the argmax
// over the gathered logits sees each device's best candidates. Rows ranked out of the candidates can only change a
// draft token, never the verified text. RDNA4 only; GGML_CUDA_COARSE_HEAD=0: the exact mat-vec,
// GGML_CUDA_COARSE_HEAD_ROWS: candidates per device (8192), GGML_CUDA_COARSE_HEAD_BLOCKS: the ranking blocks ("2,7").
bool ggml_cuda_should_use_argmax_head(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_argmax_head(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// The q8_1 copy of t (contiguous F32, rows of a multiple of 32 values) that the next quantized mat-vecs reading t
// through the view `as` (t itself, or a reshape of all of it with rows of a multiple of 32, at most
// MMVQ_MAX_BATCH_SIZE rows) take from ctx.q8_1_reuse in this graph evaluation instead of quantizing: the producer of
// t writes it along with t (quantize_q8_1_warp32) in the rows of `as` (ggml_cuda_q8_1_out_block). .ptr == nullptr:
// nothing to write (reuse off, another stream, an unsupported shape).
ggml_cuda_q8_1_out ggml_cuda_q8_1_reuse_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const ggml_tensor * as);

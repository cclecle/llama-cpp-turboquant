#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"
#include "ggml-moe-hybrid.h" // struct mmid_row_mapping

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// defined in ggml-cpu.c, shared with the hybrid kernel
void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata);

// true when the operator asked for the hybrid kernel, or for its routing census
bool ggml_mul_mat_id_hybrid_enabled(void);

// same result as ggml_compute_forward_mul_mat_id, but the output rows of an expert may be shared
// with a GPU backend
void ggml_compute_forward_mul_mat_id_hybrid(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

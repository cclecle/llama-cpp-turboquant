#include "argmax-head.cuh"
#include "quantize.cuh"
#include "top-k.cuh"
#include "vecdotq.cuh"

// one warp per output row; a Q6_K block is 32 lanes of vec_dot_q6_K_q8_1 (QI6_K ints, VDR 1)
#define ARGMAX_HEAD_WARPS 8
static_assert(QI6_K == 32, "one warp per Q6_K block");

// the ranking of a row: its dot product over two of its blocks; rows past nrows (the padding of the last top-k
// part) rank last
static __global__ void argmax_head_rank(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ score,
        const int nrows, const int n_pad, const int blocks_per_row, const int b0, const int b1) {
    const int row  = blockIdx.x*ARGMAX_HEAD_WARPS + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= n_pad) {
        return;
    }
    if (row >= nrows) {
        if (lane == 0) {
            score[row] = -INFINITY;
        }
        return;
    }
    const int64_t kb = (int64_t) row*blocks_per_row;
    float s = vec_dot_q6_K_q8_1(vx, &y[b0*(QK_K/QK8_1)], kb + b0, lane)
            + vec_dot_q6_K_q8_1(vx, &y[b1*(QK_K/QK8_1)], kb + b1, lane);
    s = warp_reduce_sum<32>(s);
    if (lane == 0) {
        score[row] = s;
    }
}

static __global__ void argmax_head_fill(float * __restrict__ dst, const int nrows) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nrows) {
        dst[i] = -INFINITY;
    }
}

// the exact dot product of each candidate row (cand holds the column within its top-k part)
static __global__ void argmax_head_exact(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int * __restrict__ cand,
        float * __restrict__ dst, const int n_cand, const int k_per_part, const int part_cols, const int nrows,
        const int blocks_per_row) {
    const int c    = blockIdx.x*ARGMAX_HEAD_WARPS + threadIdx.y;
    const int lane = threadIdx.x;
    if (c >= n_cand) {
        return;
    }
    const int row = (c / k_per_part)*part_cols + cand[c];
    if (row >= nrows) {
        return;
    }
    const int64_t kb = (int64_t) row*blocks_per_row;
    float s = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        s += vec_dot_q6_K_q8_1(vx, &y[b*(QK_K/QK8_1)], kb + b, lane);
    }
    s = warp_reduce_sum<32>(s);
    if (lane == 0) {
        dst[row] = s;
    }
}

struct argmax_head_params {
    bool enabled;
    int  rows;
    int  b0;
    int  b1;
};

static const argmax_head_params & argmax_head_get_params() {
    static const argmax_head_params p = [] {
        argmax_head_params r = { true, 8192, 2, 7 }; // blocks 0,5 (R9V) lost 3% acceptance here (step 37)
        if (const char * env = getenv("GGML_CUDA_COARSE_HEAD")) {
            r.enabled = atoi(env) != 0;
        }
        if (const char * env = getenv("GGML_CUDA_COARSE_HEAD_ROWS")) {
            r.rows = std::max(1, atoi(env));
        }
        if (const char * env = getenv("GGML_CUDA_COARSE_HEAD_BLOCKS")) {
            int b0 = 0;
            int b1 = 0;
            if (sscanf(env, "%d,%d", &b0, &b1) == 2 && b0 >= 0 && b1 >= 0 && b0 != b1) {
                r.b0 = b0;
                r.b1 = b1;
            }
        }
        return r;
    }();
    return p;
}

bool ggml_cuda_should_use_argmax_head(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
    const argmax_head_params & p = argmax_head_get_params();
    if (!p.enabled || ggml_get_op_params_i32(dst, 1) != GGML_HINT_ARGMAX_ONLY || !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t blocks_per_row = src0->ne[0] / QK_K;
    return src0->ne[0] % QK_K == 0 && std::max(p.b0, p.b1) < blocks_per_row &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[1] > 2048 && src0->ne[1] <= INT_MAX;
#else
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
}

void ggml_cuda_mul_mat_argmax_head(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
    const argmax_head_params & p = argmax_head_get_params();
    cudaStream_t stream = ctx.stream();

    const int ne00           = src0->ne[0];
    const int nrows          = src0->ne[1];
    const int blocks_per_row = ne00 / QK_K;

    // the ranking goes through the one-kernel top-k in parts of at most 65536 rows
    const int n_parts    = (nrows + 65535) / 65536;
    const int part_cols  = (nrows + n_parts - 1) / n_parts;
    const int n_pad      = n_parts*part_cols;
    const int k_per_part = std::min(part_cols, (p.rows + n_parts - 1) / n_parts);
    const int n_cand     = n_parts*k_per_part;

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char>  q8_1(ctx.pool(), ne10_padded*sizeof(block_q8_1)/QK8_1);
    ggml_cuda_pool_alloc<float> score(ctx.pool(), n_pad);
    ggml_cuda_pool_alloc<int>   cand(ctx.pool(), n_cand);

    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8_1.get(), src0->type, ne00, ne00, ne00, ne00,
        ne10_padded, 1, 1, 1, stream);

    const dim3 block_dims(32, ARGMAX_HEAD_WARPS, 1);
    argmax_head_rank<<<(n_pad + ARGMAX_HEAD_WARPS - 1)/ARGMAX_HEAD_WARPS, block_dims, 0, stream>>>(
        src0->data, (const block_q8_1 *) q8_1.get(), score.get(), nrows, n_pad, blocks_per_row, p.b0, p.b1);
    CUDA_CHECK(cudaGetLastError());

    GGML_ASSERT(ggml_cuda_top_k_rows(score.get(), cand.get(), part_cols, n_parts, k_per_part, stream));

    argmax_head_fill<<<(nrows + 255)/256, 256, 0, stream>>>((float *) dst->data, nrows);
    CUDA_CHECK(cudaGetLastError());

    argmax_head_exact<<<(n_cand + ARGMAX_HEAD_WARPS - 1)/ARGMAX_HEAD_WARPS, block_dims, 0, stream>>>(
        src0->data, (const block_q8_1 *) q8_1.get(), cand.get(), (float *) dst->data, n_cand, k_per_part, part_cols,
        nrows, blocks_per_row);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("the argmax head needs the HIP top-k");
#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
}

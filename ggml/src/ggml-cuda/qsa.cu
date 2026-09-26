#include "qsa.cuh"
#include "convert.cuh"

// ggml_qsa_pool: one warp per block, each lane owns n_embd/WARP_SIZE elements at a stride of WARP_SIZE, so the lanes
// of a q8_0 row read one 32-element block together (one scale, 32 consecutive bytes)
template <ggml_type type, int n_per_lane>
static __global__ void qsa_pool_kernel(
        const char * __restrict__ k, const int32_t * __restrict__ cells, const float * __restrict__ w,
        float * __restrict__ dst, const int r, const int n_blocks,
        const int64_t nbk1, const int64_t nbk2, const int64_t s_cells1, const int64_t s_dst1, const int64_t s_dst2,
        const float eps) {
    constexpr int n_embd = n_per_lane*WARP_SIZE;

    const int b    = blockIdx.x*blockDim.y + threadIdx.y;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;

    if (b >= n_blocks) {
        return;
    }

    const int32_t * c = cells + s*s_cells1 + (int64_t) b*r;

    float acc[n_per_lane];
#pragma unroll
    for (int i = 0; i < n_per_lane; ++i) {
        acc[i] = 0.0f;
    }

    for (int m = 0; m < r; ++m) {
        const char * row = k + s*nbk2 + (int64_t) c[m]*nbk1;
#pragma unroll
        for (int i = 0; i < n_per_lane; ++i) {
            const int j = i*WARP_SIZE + lane;
            float x;
            if constexpr (type == GGML_TYPE_F32) {
                x = ((const float *) row)[j];
            } else if constexpr (type == GGML_TYPE_F16) {
                x = __half2float(((const half *) row)[j]);
            } else {
                static_assert(type == GGML_TYPE_Q8_0, "bad type");
                const block_q8_0 * q = (const block_q8_0 *) row + j/QK8_0;
                x = __half2float(q->d) * (float) q->qs[j % QK8_0];
            }
            acc[i] += x;
        }
    }

    const float scale = 1.0f/(float) r;

    float sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < n_per_lane; ++i) {
        acc[i] *= scale;
        sumsq  += acc[i]*acc[i];
    }
    sumsq = warp_reduce_sum(sumsq);

    const float rms = rsqrtf(sumsq/n_embd + eps);

    float * y = dst + s*s_dst2 + (int64_t) b*s_dst1;
#pragma unroll
    for (int i = 0; i < n_per_lane; ++i) {
        const int j = i*WARP_SIZE + lane;
        y[j] = acc[i]*rms*(w ? w[j] : 1.0f);
    }
}

template <ggml_type type>
static void qsa_pool_cuda(const ggml_tensor * k, const ggml_tensor * cells, const ggml_tensor * w, ggml_tensor * dst,
        const int r, const float eps, cudaStream_t stream) {
    const int n_embd   = k->ne[0];
    const int n_blocks = dst->ne[1];
    const int ns       = dst->ne[2];

    constexpr int rows_per_block = 4;
    const dim3 block_dims(WARP_SIZE, rows_per_block, 1);
    const dim3 block_nums((n_blocks + rows_per_block - 1)/rows_per_block, ns, 1);

    const char    * k_d = (const char *) k->data;
    const int32_t * c_d = (const int32_t *) cells->data;
    const float   * w_d = w ? (const float *) w->data : nullptr;
    float         * y_d = (float *) dst->data;

    const int64_t s_cells1 = cells->nb[1]/sizeof(int32_t);
    const int64_t s_dst1   = dst->nb[1]/sizeof(float);
    const int64_t s_dst2   = dst->nb[2]/sizeof(float);

    switch (n_embd) {
        case  64: qsa_pool_kernel<type, 2><<<block_nums, block_dims, 0, stream>>>(k_d, c_d, w_d, y_d, r, n_blocks, k->nb[1], k->nb[2], s_cells1, s_dst1, s_dst2, eps); break;
        case 128: qsa_pool_kernel<type, 4><<<block_nums, block_dims, 0, stream>>>(k_d, c_d, w_d, y_d, r, n_blocks, k->nb[1], k->nb[2], s_cells1, s_dst1, s_dst2, eps); break;
        case 256: qsa_pool_kernel<type, 8><<<block_nums, block_dims, 0, stream>>>(k_d, c_d, w_d, y_d, r, n_blocks, k->nb[1], k->nb[2], s_cells1, s_dst1, s_dst2, eps); break;
        default: GGML_ABORT("qsa_pool: unsupported n_embd %d", n_embd);
    }
}

bool ggml_cuda_qsa_pool_supported(const ggml_tensor * op) {
    const ggml_tensor * k = op->src[0];
    return (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
        (k->ne[0] == 64 || k->ne[0] == 128 || k->ne[0] == 256);
}

void ggml_cuda_op_qsa_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k     = dst->src[0];
    const ggml_tensor * cells = dst->src[1];
    const ggml_tensor * w     = dst->src[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int   r   = ggml_get_op_params_i32(dst, 0);
    const float eps = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();

    switch (k->type) {
        case GGML_TYPE_F32:  qsa_pool_cuda<GGML_TYPE_F32> (k, cells, w, dst, r, eps, stream); break;
        case GGML_TYPE_F16:  qsa_pool_cuda<GGML_TYPE_F16> (k, cells, w, dst, r, eps, stream); break;
        case GGML_TYPE_Q8_0: qsa_pool_cuda<GGML_TYPE_Q8_0>(k, cells, w, dst, r, eps, stream); break;
        default: GGML_ABORT("qsa_pool: unsupported type %s", ggml_type_name(k->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

// ggml_qsa_expand: one thread per cell
template <typename T_add>
static __global__ void qsa_expand_kernel(
        const float * __restrict__ score, const int32_t * __restrict__ blk, const T_add * __restrict__ add,
        float * __restrict__ dst, const int n_kv,
        const int64_t s_score1, const int64_t s_score2, const int64_t s_blk1,
        const int64_t s_add1, const int64_t s_add2, const int64_t s_dst1, const int64_t s_dst2) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;

    if (j >= n_kv) {
        return;
    }

    const int b = blk[s*s_blk1 + j];

    dst[s*s_dst2 + t*s_dst1 + j] = score[s*s_score2 + t*s_score1 + b] + ggml_cuda_cast<float>(add[s*s_add2 + t*s_add1 + j]);
}

void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score = dst->src[0];
    const ggml_tensor * blk   = dst->src[1];
    const ggml_tensor * add   = dst->src[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int n_kv = dst->ne[0];
    const int n_tk = dst->ne[1];
    const int ns   = dst->ne[2];

    GGML_ASSERT(n_tk <= 65535 && ns <= 65535);

    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 block_nums((n_kv + block_size - 1)/block_size, n_tk, ns);

    const size_t ts_add = ggml_type_size(add->type);

    cudaStream_t stream = ctx.stream();

    const int64_t s_score1 = score->nb[1]/sizeof(float);
    const int64_t s_score2 = score->nb[2]/sizeof(float);
    const int64_t s_blk1   = blk->nb[1]/sizeof(int32_t);
    const int64_t s_add1   = add->nb[1]/ts_add;
    const int64_t s_add2   = add->nb[2]/ts_add;
    const int64_t s_dst1   = dst->nb[1]/sizeof(float);
    const int64_t s_dst2   = dst->nb[2]/sizeof(float);

    if (add->type == GGML_TYPE_F16) {
        qsa_expand_kernel<half><<<block_nums, block_dims, 0, stream>>>(
            (const float *) score->data, (const int32_t *) blk->data, (const half *) add->data, (float *) dst->data, n_kv,
            s_score1, s_score2, s_blk1, s_add1, s_add2, s_dst1, s_dst2);
    } else {
        GGML_ASSERT(add->type == GGML_TYPE_F32);
        qsa_expand_kernel<float><<<block_nums, block_dims, 0, stream>>>(
            (const float *) score->data, (const int32_t *) blk->data, (const float *) add->data, (float *) dst->data, n_kv,
            s_score1, s_score2, s_blk1, s_add1, s_add2, s_dst1, s_dst2);
    }
    CUDA_CHECK(cudaGetLastError());
}

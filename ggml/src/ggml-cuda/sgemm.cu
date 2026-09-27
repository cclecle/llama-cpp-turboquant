#include "sgemm.cuh"

// dst[m, n] = sum_k src0[k, m] * src1[k, n], both operands contiguous along k (ggml's mul_mat layout).
// A workgroup of 256 threads owns a bm x 64 tile of dst (bm 64, or 32 / 16 for matrices with few rows) and walks k in
// slices of 16: each slice of both operands is loaded as float4 along k (the next slice into registers while the
// current one is in use) and stored transposed in LDS, so that a thread reads its bm/16 rows and its 4 columns of a k
// step from LDS and does 4*bm/16 FMAs.
// Short, wide problems (few tiles, long k: the hyper-connection projections, 24 x 10240 x 1024) split k over
// gridDim.z; every split writes its own partial tile and sgemm_reduce sums them in split order, so the result does
// not depend on scheduling.

#define SGEMM_NT  256
#define SGEMM_BN  64
#define SGEMM_BK  16
#define SGEMM_PAD 4 // keeps the LDS rows 16-byte aligned for the float4 reads

template <int bm>
__launch_bounds__(SGEMM_NT, 2)
static __global__ void sgemm_tn(
        const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
        const int M, const int N, const int K, const int64_t lda, const int64_t ldb, const int64_t ldc,
        const int k_split, const int64_t s_split) {
    constexpr int tm = bm/16; // rows per thread

    __shared__ float As[2][SGEMM_BK][bm + SGEMM_PAD];
    __shared__ float Bs[2][SGEMM_BK][SGEMM_BN + SGEMM_PAD];

    const int tid = threadIdx.x;
    const int m0  = blockIdx.x*bm;
    const int n0  = blockIdx.y*SGEMM_BN;

    const int k_begin = blockIdx.z*k_split;
    const int k_end   = min(K, k_begin + k_split);

    // loads: 64 rows x 16 k of B = 256 float4, one per thread; bm rows of A, by the first 4*bm threads
    const int lr = tid / (SGEMM_BK/4);
    const int lk = (tid % (SGEMM_BK/4))*4;
    const bool a_ld = lr < bm;
    const bool a_ok = a_ld && m0 + lr < M;
    const bool b_ok = n0 + lr < N;
    const float * a_ptr = A + (int64_t) (a_ok ? m0 + lr : 0)*lda + lk;
    const float * b_ptr = B + (int64_t) (b_ok ? n0 + lr : 0)*ldb + lk;

    // K and k_split are multiples of 4, so a float4 is either all in [k_begin, k_end) or all out
    const auto load = [&](const int k0, float4 & ra, float4 & rb) {
        const bool in_k = k0 + lk < k_end;
        ra = a_ok && in_k ? *(const float4 *) (a_ptr + k0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        rb = b_ok && in_k ? *(const float4 *) (b_ptr + k0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    const auto store = [&](const int buf, const float4 & ra, const float4 & rb) {
        if (a_ld) {
            As[buf][lk + 0][lr] = ra.x; As[buf][lk + 1][lr] = ra.y; As[buf][lk + 2][lr] = ra.z; As[buf][lk + 3][lr] = ra.w;
        }
        Bs[buf][lk + 0][lr] = rb.x; Bs[buf][lk + 1][lr] = rb.y; Bs[buf][lk + 2][lr] = rb.z; Bs[buf][lk + 3][lr] = rb.w;
    };

    // compute: thread (tx, ty) owns rows m0 + tm*tx .. + tm-1 and columns n0 + 4*ty .. +3
    const int tx = tid % 16;
    const int ty = tid / 16;

    float acc[tm][4] = {{0.0f}};

    float4 ra;
    float4 rb;
    load(k_begin, ra, rb);
    store(0, ra, rb);
    __syncthreads();

    int buf = 0;
    for (int k0 = k_begin; k0 < k_end; k0 += SGEMM_BK) {
        const bool more = k0 + SGEMM_BK < k_end;
        if (more) {
            load(k0 + SGEMM_BK, ra, rb);
        }
#pragma unroll
        for (int k = 0; k < SGEMM_BK; ++k) {
            float av[tm];
            if constexpr (tm == 4) {
                const float4 a = *(const float4 *) &As[buf][k][4*tx];
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
            } else if constexpr (tm == 2) {
                const float2 a = *(const float2 *) &As[buf][k][2*tx];
                av[0] = a.x; av[1] = a.y;
            } else {
                av[0] = As[buf][k][tx];
            }
            const float4 b = *(const float4 *) &Bs[buf][k][4*ty];
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < tm; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1, ra, rb);
        }
        __syncthreads();
        buf ^= 1;
    }

    float * c = C + blockIdx.z*s_split;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n0 + 4*ty + j;
        if (n >= N) {
            break;
        }
#pragma unroll
        for (int i = 0; i < tm; ++i) {
            const int m = m0 + tm*tx + i;
            if (m < M) {
                c[(int64_t) n*ldc + m] = acc[i][j];
            }
        }
    }
}

// dst[m, n] = partial 0 + partial 1 + ... in split order
static __global__ void sgemm_reduce(
        const float * __restrict__ tmp, float * __restrict__ dst, const int M, const int N, const int n_split, const int64_t ldc) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t MN = (int64_t) M*N;
    if (i >= MN) {
        return;
    }
    float sum = tmp[i];
    for (int s = 1; s < n_split; ++s) {
        sum += tmp[s*MN + i];
    }
    dst[(i / M)*ldc + i % M] = sum;
}

bool ggml_cuda_should_use_sgemm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_SGEMM");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_AMD(cc) || !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // one matrix each; batches keep the batched hipBLAS path
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // float4 loads along k: rows contiguous, k and the row strides multiples of 4 floats, 16-byte aligned data
    const int64_t K = src0->ne[0];
    return src1->ne[1] > 8 && K % 4 == 0 && K <= INT_MAX && src0->ne[1] <= INT_MAX && src1->ne[1] <= INT_MAX &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src0->nb[1] % 16 == 0 && src1->nb[1] % 16 == 0 && dst->nb[1] % sizeof(float) == 0 &&
        (uintptr_t) src0->data % 16 == 0 && (uintptr_t) src1->data % 16 == 0;
}

void ggml_cuda_mul_mat_sgemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    const int64_t lda = src0->nb[1]/sizeof(float);
    const int64_t ldb = src1->nb[1]/sizeof(float);
    const int64_t ldc = dst->nb[1]/sizeof(float);

    const float * A = (const float *) src0->data;
    const float * B = (const float *) src1->data;
    float       * C = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // few rows (the hyper-connection projections): a shorter tile wastes fewer FMAs
    const int bm      = M <= 16 ? 16 : M <= 32 ? 32 : 64;
    const int tiles_m = (M + bm - 1)/bm;
    const int tiles_n = (N + SGEMM_BN - 1)/SGEMM_BN;
    const int nsm     = ggml_cuda_info().devices[ctx.device].nsm;

    // split k until there are about two workgroups per CU, keeping at least 256 of k per split
    int n_split = 1;
    if (tiles_m*tiles_n < nsm) {
        n_split = std::min((2*nsm + tiles_m*tiles_n - 1)/(tiles_m*tiles_n), std::max(1, K/256));
        // the partials stay small: at most 64 MiB
        while (n_split > 1 && (size_t) n_split*M*N*sizeof(float) > (64u << 20)) {
            n_split--;
        }
    }
    const int k_split = GGML_PAD((K + n_split - 1)/n_split, SGEMM_BK);
    n_split = (K + k_split - 1)/k_split;

    const dim3 block_dims(SGEMM_NT, 1, 1);
    const dim3 block_nums(tiles_m, tiles_n, n_split);

    const auto launch = [&](float * c, const int64_t ldc_c, const int64_t s_split) {
        switch (bm) {
            case 16: sgemm_tn<16><<<block_nums, block_dims, 0, stream>>>(A, B, c, M, N, K, lda, ldb, ldc_c, k_split, s_split); break;
            case 32: sgemm_tn<32><<<block_nums, block_dims, 0, stream>>>(A, B, c, M, N, K, lda, ldb, ldc_c, k_split, s_split); break;
            default: sgemm_tn<64><<<block_nums, block_dims, 0, stream>>>(A, B, c, M, N, K, lda, ldb, ldc_c, k_split, s_split); break;
        }
        CUDA_CHECK(cudaGetLastError());
    };

    if (n_split == 1) {
        launch(C, ldc, 0);
        return;
    }

    const int64_t MN = (int64_t) M*N;
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n_split*MN);
    launch(tmp.get(), M, MN);

    constexpr int reduce_block = 256;
    sgemm_reduce<<<(MN + reduce_block - 1)/reduce_block, reduce_block, 0, stream>>>(tmp.get(), C, M, N, n_split, ldc);
    CUDA_CHECK(cudaGetLastError());
}

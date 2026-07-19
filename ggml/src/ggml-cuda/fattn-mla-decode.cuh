#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdlib>

// Decode-only FlashAttention kernel for MLA head shapes (DKQ == 576, DV == 512).
//
// At decode Q has a single column, so this is a streaming reduction rather than a tiled GEMM.
// V aliases the first DV dims of K, so a KV row is fetched from global memory exactly once and
// serves both the KQ dot products and the VKQ accumulation.
//
// Work is split in two phases with different thread mappings, connected through the LDS tile:
//   phase 1: warp w handles heads [w*ncols2/nwarps, (w+1)*ncols2/nwarps), lanes split the DKQ dims.
//            Q stays in registers and is reused for every KV position.
//   phase 2: warp w handles the DV dim slice [w*DV/nwarps, (w+1)*DV/nwarps) for all heads, so each
//            KV row is read from LDS exactly once per block.

#define MLA_DEC_DKQ     576
#define MLA_DEC_DV      512
#define MLA_DEC_KQ_TILE  16 // largest supported KV tile; the gate only needs K->ne[1] to divide by it

// Heads are spread over warps by even division; for counts that do not divide evenly (e.g. 20 heads
// over 8 warps) the remainder is distributed, giving per-warp counts of 2,3,2,3,...
static constexpr __device__ __host__ int mla_dec_head_begin(int ncols2, int nwarps, int w) {
    return (ncols2*w) / nwarps;
}
static constexpr __device__ __host__ int mla_dec_head_max(int ncols2, int nwarps) {
    return (ncols2 + nwarps - 1) / nwarps;
}

template <int ncols2, int nthreads_t, int nbatch_fa_t, ggml_type type_K, bool use_logit_softcap>
__launch_bounds__(nthreads_t, 1)
static __global__ void flash_attn_ext_mla_decode(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const char * __restrict__ sinks_ptr,
        const int  * __restrict__ KV_max_ptr,
        float      * __restrict__ dst_ptr,
        float2     * __restrict__ dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP) && defined(RDNA)
    constexpr int DKQ       = MLA_DEC_DKQ;
    constexpr int DV        = MLA_DEC_DV;
    constexpr int nthreads  = nthreads_t;
    constexpr int nbatch_fa = nbatch_fa_t;
    constexpr int nwarps    = nthreads / WARP_SIZE;

    constexpr int hpw   = mla_dec_head_max(ncols2, nwarps); // max heads per warp in phase 1
    constexpr int nh2_A = DKQ / 2 / WARP_SIZE;              // half2 per lane in phase 1
    constexpr int dv_pw = DV / nwarps;                      // DV dims per warp in phase 2
    constexpr int dpl_B = dv_pw / WARP_SIZE;                // DV dims per lane in phase 2

    static_assert(DKQ % (2*WARP_SIZE) == 0, "DKQ must be a multiple of 2*WARP_SIZE");
    static_assert(DV  % (2*nthreads) == 0, "DV must be a multiple of 2*nthreads");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int h_beg = mla_dec_head_begin(ncols2, nwarps, warp);
    const int h_cnt = mla_dec_head_begin(ncols2, nwarps, warp + 1) - h_beg;

    // ncols2 Q heads per block. Splitting the heads over several blocks costs extra passes over K
    // but buys parallelism, which on RDNA4 is the scarcer resource. ne02 % ncols2 == 0 is required.
    const int ngroups  = ne02 / ncols2;
    const int sequence = blockIdx.z / ngroups;
    const int head0    = (blockIdx.z - sequence*ngroups) * ncols2;

    const char * __restrict__ Q    = Q_ptr    + nb03*sequence;
    const char * __restrict__ K    = K_ptr    + nb13*sequence;
    const half * __restrict__ maskh = (const half *) (mask_ptr + nb33*(sequence % ne33));

    // P is indexed [head][k] so that each warp owns a contiguous, naturally aligned region for its
    // own heads. A transposed [k][head] half layout was tried and is faster on paper, but it puts
    // two warps' heads in the same 32-bit LDS word, which races when the compiler merges the 16-bit
    // stores into a read-modify-write.
    __shared__ half  KV_tile[nbatch_fa * DKQ];
    __shared__ float KQ_p[ncols2 * nbatch_fa];
    __shared__ float KQ_scale_s[ncols2];
    __shared__ float KQ_sum_s[ncols2];
    __shared__ float KQ_max_s[ncols2];

    // Q for this warp's heads, scaled, kept in registers for the whole KV loop.
    // Lanes are interleaved over the dim axis (lane + WARP_SIZE*j) so that both the global Q loads
    // and the LDS K loads below are naturally aligned and fully coalesced.
    half2 Q_reg[hpw][nh2_A];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int h0 = 0; h0 < hpw; ++h0) {
        if (h0 >= h_cnt) {
            break;
        }
        const float2 * Q_h = (const float2 *) (Q + nb02*(head0 + h_beg + h0));
#pragma unroll
        for (int j = 0; j < nh2_A; ++j) {
            const float2 tmp = Q_h[lane + WARP_SIZE*j];
            Q_reg[h0][j] = make_half2(tmp.x*scale, tmp.y*scale);
        }
    }

    float VKQ[ncols2][dpl_B] = {{0.0f}};

    float KQ_max[hpw];
    float KQ_sum[hpw];
#pragma unroll
    for (int h0 = 0; h0 < hpw; ++h0) {
        KQ_max[h0] = -FLT_MAX/2.0f;
        KQ_sum[h0] = 0.0f;
    }

    const int k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;

    const int d_base_B = warp*dv_pw + lane*dpl_B; // phase 2 dim slice owned by this thread

    // The KV tile is double buffered through registers. Flattened over (row, 16-byte chunk) so every
    // thread does the same number of loads and consecutive threads read consecutive addresses. The
    // loads for the next tile are issued before the current one is processed, so the HBM latency
    // overlaps with compute instead of stalling the whole block at the barrier after the loads.
    constexpr int chunks_per_row = DKQ/8;
    constexpr int nchunks        = nbatch_fa * chunks_per_row;
    constexpr int niter          = (nchunks + nthreads - 1) / nthreads;
    constexpr bool exact         = niter*nthreads == nchunks;

    const int k_VKQ_stride = gridDim.y*nbatch_fa;

    // q8_0 K is dequantized straight into the LDS tile. That avoids the generic path, which
    // materializes the whole KV cache as f16 in global memory on every call: per KV row that costs
    // read 612 B + write 1152 B + read back 1152 B, against 612 B read here.
    constexpr int blocks_per_row = DKQ/QK8_0;
    constexpr int nblk           = nbatch_fa * blocks_per_row;

    int4 kv_stage[niter]; // unused (and eliminated) on the q8_0 path

    auto load_tile = [&](const int k0) {
#pragma unroll
        for (int n = 0; n < niter; ++n) {
            const int c = n*nthreads + tid;
            if (exact || c < nchunks) {
                const int row = c / chunks_per_row;
                const int i   = c % chunks_per_row;
                kv_stage[n] = *(const int4 *) (K + (k0 + row)*nb11 + (size_t)16*i);
            }
        }
    };

    if (type_K == GGML_TYPE_F16 && blockIdx.y*nbatch_fa < k_VKQ_max) {
        load_tile(blockIdx.y*nbatch_fa);
    }

    for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += k_VKQ_stride) {
        const int k_VKQ_nxt = k_VKQ_0 + k_VKQ_stride;

        if constexpr (type_K == GGML_TYPE_F16) {
            // Publish the tile prefetched during the previous iteration.
#pragma unroll
            for (int n = 0; n < niter; ++n) {
                const int c = n*nthreads + tid;
                if (exact || c < nchunks) {
                    const int row = c / chunks_per_row;
                    const int i   = c % chunks_per_row;
                    *(int4 *) &KV_tile[row*DKQ + 8*i] = kv_stage[n];
                }
            }
        } else {
            for (int b = tid; b < nblk; b += nthreads) {
                const int row = b / blocks_per_row;
                const int ib  = b % blocks_per_row;

                const block_q8_0 * bq = (const block_q8_0 *) (K + (k_VKQ_0 + row)*nb11) + ib;

                // qs sits at a 2-byte offset inside the 34-byte block, so it is only 2-byte aligned.
                int qs[QK8_0/4];
#pragma unroll
                for (int j = 0; j < QK8_0/4; ++j) {
                    qs[j] = get_int_b2(bq->qs, j);
                }
                const float d = __half2float(bq->d);

                half * dstp = &KV_tile[row*DKQ + ib*QK8_0];
#pragma unroll
                for (int j = 0; j < QK8_0/4; ++j) {
                    const int v = qs[j];
                    ((half2 *) dstp)[2*j + 0] = make_half2(d*float(int8_t(v      )), d*float(int8_t(v >>  8)));
                    ((half2 *) dstp)[2*j + 1] = make_half2(d*float(int8_t(v >> 16)), d*float(int8_t(v >> 24)));
                }
            }
        }

        __syncthreads();

        // Issue the next tile's loads now; they complete while this tile is being processed.
        if (type_K == GGML_TYPE_F16 && k_VKQ_nxt < k_VKQ_max) {
            load_tile(k_VKQ_nxt);
        }

        // Phase 1: KQ dot products. Every lane redundantly ends up with the full S value for the
        // current KV position, so the running maximum needs no further reduction.
        float S_own[hpw]; // S for the KV position owned by this lane (lane < nbatch_fa)
        float KQ_max_new[hpw];
#pragma unroll
        for (int h0 = 0; h0 < hpw; ++h0) {
            KQ_max_new[h0] = KQ_max[h0];
            S_own[h0]      = -FLT_MAX/2.0f;
        }

        for (int k = 0; k < nbatch_fa; ++k) {
            const half2 * K_k = (const half2 *) &KV_tile[k*DKQ];

            half2 K_reg[nh2_A];
#pragma unroll
            for (int j = 0; j < nh2_A; ++j) {
                K_reg[j] = K_k[lane + WARP_SIZE*j];
            }

#pragma unroll
            for (int h0 = 0; h0 < hpw; ++h0) {
                if (h0 >= h_cnt) {
                    break;
                }
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < nh2_A; ++j) {
                    ggml_cuda_mad(s, Q_reg[h0][j], K_reg[j]);
                }
                s = warp_reduce_sum(s);

                if (use_logit_softcap) {
                    s = logit_softcap*tanhf(s);
                }

                s += __half2float(maskh[k_VKQ_0 + k]);

                KQ_max_new[h0] = fmaxf(KQ_max_new[h0], s + FATTN_KQ_MAX_OFFSET);

                if (lane == k) {
                    S_own[h0] = s;
                }
            }
        }

        // Rescale the running softmax state and publish P and the rescaling factor for phase 2.
#pragma unroll
        for (int h0 = 0; h0 < hpw; ++h0) {
            if (h0 >= h_cnt) {
                break;
            }
            const int h = h_beg + h0;

            const float KQ_max_scale = expf(KQ_max[h0] - KQ_max_new[h0]);
            KQ_max[h0] = KQ_max_new[h0];

            const float p = lane < nbatch_fa ? expf(S_own[h0] - KQ_max[h0]) : 0.0f;
            if (lane < nbatch_fa) {
                KQ_p[h*nbatch_fa + lane] = p;
            }

            KQ_sum[h0] = KQ_sum[h0]*KQ_max_scale + warp_reduce_sum(p);

            if (lane == 0) {
                KQ_scale_s[h] = KQ_max_scale;
            }
        }

        __syncthreads();

        // Phase 2: VKQ accumulation. V is the first DV dims of the KV rows already in LDS.
#pragma unroll
        for (int h = 0; h < ncols2; ++h) {
            const float s = KQ_scale_s[h];
#pragma unroll
            for (int i = 0; i < dpl_B; ++i) {
                VKQ[h][i] *= s;
            }
        }

        for (int k = 0; k < nbatch_fa; ++k) {
            half2 V_reg[dpl_B/2];
            ggml_cuda_memcpy_1<dpl_B*sizeof(half)>(V_reg, &KV_tile[k*DKQ + d_base_B]);

            float2 V_f[dpl_B/2];
#pragma unroll
            for (int i = 0; i < dpl_B/2; ++i) {
                V_f[i] = __half22float2(V_reg[i]);
            }

#pragma unroll
            for (int h = 0; h < ncols2; ++h) {
                const float p = KQ_p[h*nbatch_fa + k];
#pragma unroll
                for (int i = 0; i < dpl_B/2; ++i) {
                    VKQ[h][2*i + 0] += p*V_f[i].x;
                    VKQ[h][2*i + 1] += p*V_f[i].y;
                }
            }
        }

        __syncthreads();
    }

    // Publish the per-head softmax denominators so every thread can normalize its own dim slice.
#pragma unroll
    for (int h0 = 0; h0 < hpw; ++h0) {
        if (h0 >= h_cnt) {
            break;
        }
        if (lane == 0) {
            KQ_sum_s[h_beg + h0] = KQ_sum[h0];
            KQ_max_s[h_beg + h0] = KQ_max[h0];
        }
    }

    __syncthreads();

#pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        const int head = head0 + h;

        float dst_val[dpl_B];
#pragma unroll
        for (int i = 0; i < dpl_B; ++i) {
            dst_val[i] = VKQ[h][i];
        }

        if (gridDim.y == 1) {
            const float inv_sum = 1.0f / KQ_sum_s[h];
#pragma unroll
            for (int i = 0; i < dpl_B; ++i) {
                dst_val[i] *= inv_sum;
            }
        }

        float * dst_h = dst_ptr + (size_t)(((sequence*int(ne01.z) + 0)*ne02 + head)*gridDim.y + blockIdx.y)*DV;
#pragma unroll
        for (int i = 0; i < dpl_B; ++i) {
            dst_h[d_base_B + i] = dst_val[i];
        }
    }

    if (gridDim.y != 1 && tid < ncols2) {
        dst_meta_ptr[((sequence*int(ne01.z) + 0)*ne02 + head0 + tid)*gridDim.y + blockIdx.y] =
            make_float2(KQ_max_s[tid], KQ_sum_s[tid]);
    }


    GGML_UNUSED_VARS(V_ptr, sinks_ptr, max_bias, m0, m1, n_head_log2,
        ne00, ne10, ne12, ne13, ne31, ne32, nb01, nb12, nb21, nb22, nb23, nb31, nb32);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE && GGML_USE_HIP && RDNA
}

template <int ncols2, int nthreads_t, int nbatch_fa_t, ggml_type type_K>
static void ggml_cuda_flash_attn_ext_mla_decode_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr int nwarps = nthreads_t / WARP_SIZE;
    constexpr size_t nbytes_shared = 0;
    // q8_0 K is consumed in its native form, so the f16 materialization is not needed.
    constexpr bool need_f16_K = type_K == GGML_TYPE_F16;

    if (logit_softcap == 0.0f) {
        fattn_kernel_t fattn_kernel = flash_attn_ext_mla_decode<ncols2, nthreads_t, nbatch_fa_t, type_K, false>;
        launch_fattn<MLA_DEC_DV, 1, ncols2>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa_t, need_f16_K, false, false);
    } else {
        fattn_kernel_t fattn_kernel = flash_attn_ext_mla_decode<ncols2, nthreads_t, nbatch_fa_t, type_K, true>;
        launch_fattn<MLA_DEC_DV, 1, ncols2>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa_t, need_f16_K, false, false);
    }
}

template <int ncols2, int nthreads_t, int nbatch_fa_t>
static void ggml_cuda_flash_attn_ext_mla_decode_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_Q8_0:
            ggml_cuda_flash_attn_ext_mla_decode_case_impl<ncols2, nthreads_t, nbatch_fa_t, GGML_TYPE_Q8_0>(ctx, dst);
            break;
        default:
            ggml_cuda_flash_attn_ext_mla_decode_case_impl<ncols2, nthreads_t, nbatch_fa_t, GGML_TYPE_F16>(ctx, dst);
            break;
    }
}

// Geometry (threads per block, KV positions per tile) trades LDS footprint and register pressure
// against occupancy. GGML_MLA_DEC_CFG selects a variant for tuning; the default is the fastest
// measured configuration.
static int ggml_cuda_mla_decode_cfg() {
    static const int cfg = []() {
        const char * s = getenv("GGML_MLA_DEC_CFG");
        return s ? atoi(s) : 0;
    }();
    return cfg;
}

static void ggml_cuda_flash_attn_ext_mla_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    GGML_ASSERT(Q->ne[2] / K->ne[2] == 20);

    switch (ggml_cuda_mla_decode_cfg()) {
        //                                                    heads thr  fa
        case  1: ggml_cuda_flash_attn_ext_mla_decode_case<20, 128, 16>(ctx, dst); break;
        case  3: ggml_cuda_flash_attn_ext_mla_decode_case<20, 128,  4>(ctx, dst); break;
        case  2:
        default: ggml_cuda_flash_attn_ext_mla_decode_case<20, 128,  8>(ctx, dst); break;
    }
}

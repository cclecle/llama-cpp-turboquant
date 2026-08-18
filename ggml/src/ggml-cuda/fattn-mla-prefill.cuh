#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdlib>

// Prefill FlashAttention kernel for MLA head shapes (DKQ == 576, DV == 512) on RDNA4.
//
// The tile kernel that otherwise serves this shape uses no matrix cores and reaches ~13 TFLOPS on
// gfx1201, against ~36 TFLOPS for the VALU dot2 rate and ~142 for WMMA. This kernel uses WMMA and
// measures ~92 TFLOPS standalone on the GLM-4.7-Flash shape (512 queries x 32768 KV x 20 heads).
//
// The WMMA contract below was established by measurement on gfx1201, not by reading the ISA docs
// (see the note on the accumulator - getting it wrong yields a silently transposed S):
//     a[l]   = A[lane%16][(lane/16)*8 + l]
//     b[l]   = B[lane%16][(lane/16)*8 + l]
//     acc[l] = D[(lane/16)*8 + l][lane%16]     with  D[i][j] = sum_k A[i][k]*B[j][k]
// so the accumulator is transposed with respect to the A/B fragments.
//
// Feeding A = K and B = Q makes acc hold S[kv][query], which means each lane owns exactly one
// query row and 8 of its KV values. Two things fall out of that:
//   - the softmax max/sum reduce inside registers (8 values plus one shfl_xor); only NBC partials
//     per query row ever cross warps,
//   - the resulting P fragment is already in A/B layout, so it feeds the second GEMM with no
//     transpose and no LDS round trip.
//
// Phase 2 computes O[dv][query] = sum_kv V^T[dv][kv] * P[query][kv]. WMMA contracts over the k
// axis with k contiguous per lane, but V stores kv as the slow axis, so the V fragment needs a
// transposed gather whichever operand slot it takes. That gather is served straight from global
// memory: staging V transposed in LDS was measured at 18 TFLOPS against 45 for the global gather,
// because the scattered LDS *writes* cost as much as the gather saves. Only Q is staged.
//
// Measured variants on the GLM-4.7-Flash shape (TFLOPS, higher is better):
//     Q from global f32, converted per fragment      60.4
//     Q already f16 in global (not reachable in ggml) 74.2
//     Q staged to LDS as f16 once per block           92.1  <- what this kernel does
// Staging Q wins because every KV tile would otherwise re-read it, once per bc tile.

// Set GGML_MLA_PREFILL=0 to fall back to the tile kernel, so the same binary can be A/B'd for
// perplexity and speed without a rebuild.
static bool ggml_cuda_mla_prefill_enabled() {
    static const bool enabled = []() {
        const char * s = getenv("GGML_MLA_PREFILL");
        return s == nullptr || atoi(s) != 0;
    }();
    return enabled;
}

#define MLA_PRE_DKQ 576
#define MLA_PRE_DV  512
#define MLA_PRE_BR   32 // query rows per block
#define MLA_PRE_BC  128 // KV positions per tile
#define MLA_PRE_NW   16 // warps per block

#if defined(GGML_USE_HIP) && defined(RDNA4)
typedef __attribute__((ext_vector_type(8))) _Float16 mla_pre_half8;
typedef __attribute__((ext_vector_type(8))) float    mla_pre_float8;
#endif // GGML_USE_HIP && RDNA4

template <int Br, int Bc, int NW>
struct mla_pre_cfg {
    static constexpr int NBR   = Br / 16;          // query tiles
    static constexpr int NBC   = Bc / 16;          // KV tiles
    static constexpr int NDV   = MLA_PRE_DV / 16 / NW; // dv tiles per warp in phase 2
    static constexpr int DKQ_S = MLA_PRE_DKQ + 8;  // padded so the 16-row Q gather changes banks
    static constexpr int Bc_S  = Bc + 8;           // stays a multiple of 8 halves => 16 B aligned

    // Phase 1 gives every warp one (query tile, KV tile) pair and the full 576-dim reduction, so
    // no partial sums cross warps.
    static_assert(NBR*NBC == NW, "warps must factor into query tiles x KV tiles");
    static_assert(NDV*16*NW == MLA_PRE_DV, "dv must divide evenly over warps");
    static_assert(Br % 16 == 0 && Bc % 16 == 0, "tile sizes must be multiples of the WMMA tile");
};

template <int Br, int Bc, int NW, bool use_logit_softcap>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void flash_attn_ext_mla_prefill(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP) && defined(RDNA4)
    using C = mla_pre_cfg<Br, Bc, NW>;
    constexpr int DKQ   = MLA_PRE_DKQ;
    constexpr int DV    = MLA_PRE_DV;
    constexpr int DKQ_S = C::DKQ_S;
    constexpr int Bc_S  = C::Bc_S;

    const int lane  = threadIdx.x;
    const int w     = threadIdx.y;
    const int tid   = w*WARP_SIZE + lane;
    const int row   = lane % 16;   // query row inside a fragment
    const int khalf = lane / 16;   // which half of the 16-wide k axis this lane holds

    const int br_t = w % C::NBR;
    const int bc_t = w / C::NBR;

    // ncols2 == 1: one Q head per block.
    const int col_Q_0   = blockIdx.x * Br;
    const int sequence  = blockIdx.z / ne02;
    const int head0     = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    // The col_Q_0 term is load bearing: without it every tile past the first stages the batch's
    // first Br queries instead of its own.
    const char * Q_c = Q_ptr + nb03*sequence + nb02*head0 + (size_t)nb01*col_Q_0;
    const char * K_c = K_ptr + nb13*sequence + nb12*(head0 / gqa_ratio);
    const char * V_c = V_ptr + nb23*sequence + nb22*(head0 / gqa_ratio);

    const half * maskh = mask_ptr ? (const half *) (mask_ptr + nb33*(sequence % ne33)) : nullptr;
    const int stride_mask = nb31 / sizeof(half);
    const float slope = get_alibi_slope(max_bias, head0, n_head_log2, m0, m1);

    __shared__ _Float16 Qs[Br*DKQ_S];
    __shared__ _Float16 Ps[Br*Bc_S];
    __shared__ float Mpart[C::NBC*Br];
    __shared__ float Spart[C::NBC*Br];
    __shared__ float Mx[Br];
    __shared__ float Sm[Br];
    __shared__ float Rs[Br];

    ggml_cuda_pdl_sync();

    for (int i = tid; i < Br; i += NW*WARP_SIZE) {
        Mx[i] = -FLT_MAX/2.0f;
        Sm[i] = 0.0f;
    }

    // Q does not change across KV tiles, so convert it to f16 once. Rows past the end of the
    // batch are zero filled rather than read.
#pragma unroll 1
    for (int idx = tid; idx < Br*(DKQ/4); idx += NW*WARP_SIZE) {
        const int r  = idx / (DKQ/4);
        const int ch = idx % (DKQ/4);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (col_Q_0 + r < int(ne01.z)) {
            v = *(const float4 *) (Q_c + (size_t)nb01*r + (size_t)ch*16);
        }
        _Float16 * d = &Qs[r*DKQ_S + ch*4];
        d[0] = (_Float16) v.x;
        d[1] = (_Float16) v.y;
        d[2] = (_Float16) v.z;
        d[3] = (_Float16) v.w;
    }

    mla_pre_float8 o_acc[C::NBR][C::NDV];
#pragma unroll
    for (int b = 0; b < C::NBR; ++b) {
#pragma unroll
        for (int t = 0; t < C::NDV; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) o_acc[b][t][l] = 0.0f;
        }
    }

    const int q_own  = br_t*16 + row;              // query row inside the block's Br rows
    const int q_glob = col_Q_0 + q_own;
    const int q_mask = q_glob < int(ne01.z) ? q_glob : int(ne01.z) - 1;

    const int k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;

    __syncthreads();

    for (int kv0 = blockIdx.y*Bc; kv0 < k_VKQ_max; kv0 += gridDim.y*Bc) {
        // ---- phase 1: S = K * Q^T, all 576 dims inside one warp ----
        mla_pre_float8 s_acc;
#pragma unroll
        for (int l = 0; l < 8; ++l) s_acc[l] = 0.0f;

        const char * krow = K_c + (size_t)(kv0 + bc_t*16 + row)*nb11;

#pragma unroll 4
        for (int k0 = 0; k0 < DKQ; k0 += 16) {
            const mla_pre_half8 a = *(const mla_pre_half8 *) (krow + (size_t)(k0 + khalf*8)*sizeof(half));
            const mla_pre_half8 b = *(const mla_pre_half8 *) &Qs[q_own*DKQ_S + k0 + khalf*8];
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, s_acc);
        }

        // s_acc[l] = S[kv = bc_t*16 + khalf*8 + l][query = q_own]
        float sv[8];
        float m = -FLT_MAX/2.0f;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            float s = s_acc[l]*scale;
            if (use_logit_softcap) {
                s = logit_softcap*tanhf(s);
            }
            if (maskh) {
                s += slope*__half2float(maskh[(size_t)q_mask*stride_mask + kv0 + bc_t*16 + khalf*8 + l]);
            }
            sv[l] = s;
            m = fmaxf(m, s + FATTN_KQ_MAX_OFFSET);
        }
        m = fmaxf(m, __shfl_xor(m, 16));

        if (khalf == 0) {
            Mpart[bc_t*Br + q_own] = m;
        }

        __syncthreads();

        float mtile = -FLT_MAX/2.0f;
#pragma unroll
        for (int c = 0; c < C::NBC; ++c) mtile = fmaxf(mtile, Mpart[c*Br + q_own]);

        const float m_new = fmaxf(Mx[q_own], mtile);

        float ssum = 0.0f;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float p = expf(sv[l] - m_new);
            ssum += p;
            Ps[q_own*Bc_S + bc_t*16 + khalf*8 + l] = (_Float16) p;
        }
        ssum += __shfl_xor(ssum, 16);
        if (khalf == 0) {
            Spart[bc_t*Br + q_own] = ssum;
        }

        __syncthreads();

        // One writer per query row folds this tile into the running softmax state.
        if (tid < Br) {
            const float mo = Mx[tid];
            float mt = -FLT_MAX/2.0f;
#pragma unroll
            for (int c = 0; c < C::NBC; ++c) mt = fmaxf(mt, Mpart[c*Br + tid]);
            const float mn = fmaxf(mo, mt);
            float sacc = 0.0f;
#pragma unroll
            for (int c = 0; c < C::NBC; ++c) sacc += Spart[c*Br + tid];
            const float r = expf(mo - mn);
            Rs[tid] = r;
            Sm[tid] = Sm[tid]*r + sacc;
            Mx[tid] = mn;
        }

        __syncthreads();

        // ---- phase 2: O += V^T * P^T; warps split by dv so each V fragment serves every query tile ----
#pragma unroll
        for (int b = 0; b < C::NBR; ++b) {
            const float r = Rs[b*16 + row];
#pragma unroll
            for (int t = 0; t < C::NDV; ++t) {
#pragma unroll
                for (int l = 0; l < 8; ++l) o_acc[b][t][l] *= r;
            }
        }

#pragma unroll
        for (int bs = 0; bs < Bc/16; ++bs) {
            mla_pre_half8 pb[C::NBR];
#pragma unroll
            for (int b = 0; b < C::NBR; ++b) {
                pb[b] = *(const mla_pre_half8 *) &Ps[(b*16 + row)*Bc_S + bs*16 + khalf*8];
            }
#pragma unroll
            for (int t = 0; t < C::NDV; ++t) {
                const int dv0 = (w*C::NDV + t)*16;
                mla_pre_half8 va;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    va[l] = *(const _Float16 *) (V_c + (size_t)(kv0 + bs*16 + khalf*8 + l)*nb21
                                                     + (size_t)(dv0 + row)*sizeof(half));
                }
#pragma unroll
                for (int b = 0; b < C::NBR; ++b) {
                    o_acc[b][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, pb[b], o_acc[b][t]);
                }
            }
        }

        __syncthreads();
    }

    // acc[l] = O[dv = dv0 + khalf*8 + l][query = b*16 + row]
#pragma unroll
    for (int b = 0; b < C::NBR; ++b) {
        const int q_l = b*16 + row;
        const int q_g = col_Q_0 + q_l;
        if (q_g >= int(ne01.z)) {
            continue;
        }
        const float norm = dst_meta_ptr == nullptr ? 1.0f/Sm[q_l] : 1.0f;
        const int j_dst = ((sequence*int(ne01.z) + q_g)*ne02 + head0)*gridDim.y + blockIdx.y;
#pragma unroll
        for (int t = 0; t < C::NDV; ++t) {
            const int dv0 = (w*C::NDV + t)*16;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                dst_ptr[(size_t)j_dst*DV + dv0 + khalf*8 + l] = o_acc[b][t][l]*norm;
            }
        }
    }

    if (dst_meta_ptr != nullptr && tid < Br && col_Q_0 + tid < int(ne01.z)) {
        const int j_dst = ((sequence*int(ne01.z) + col_Q_0 + tid)*ne02 + head0)*gridDim.y + blockIdx.y;
        dst_meta_ptr[j_dst] = make_float2(Mx[tid], Sm[tid]);
    }

    GGML_UNUSED_VARS(sinks_ptr, ne00, ne03, ne10, ne13, ne31, ne32, nb32);
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
#endif // FLASH_ATTN_AVAILABLE && GGML_USE_HIP && RDNA4
}

template <int Br, int Bc, int NW>
static void ggml_cuda_flash_attn_ext_mla_prefill_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // All shared memory is statically sized, so nothing extra is requested at launch.
    constexpr size_t nbytes_shared = 0;

    if (logit_softcap == 0.0f) {
        fattn_kernel_t fattn_kernel = flash_attn_ext_mla_prefill<Br, Bc, NW, false>;
        launch_fattn<MLA_PRE_DV, Br, 1>
            (ctx, dst, fattn_kernel, NW, nbytes_shared, Bc, true, true, false);
    } else {
        fattn_kernel_t fattn_kernel = flash_attn_ext_mla_prefill<Br, Bc, NW, true>;
        launch_fattn<MLA_PRE_DV, Br, 1>
            (ctx, dst, fattn_kernel, NW, nbytes_shared, Bc, true, true, false);
    }
}

static void ggml_cuda_flash_attn_ext_mla_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_mla_prefill_case<MLA_PRE_BR, MLA_PRE_BC, MLA_PRE_NW>(ctx, dst);
}

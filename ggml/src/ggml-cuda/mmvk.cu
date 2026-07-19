#include "mmvk.cuh"
#include "unary.cuh"

// See mmvk.cuh for why this path exists. Structure follows the Vulkan mul_mat_vec_q6_k shader:
// 16 threads cooperate on one 256-weight superblock and BS/16 superblocks are in flight per row.
// Two deliberate departures from Vulkan, both measured on gfx1201:
//   - the superblock scales are NOT staged in LDS. Each thread needs only a few of the scale
//     bytes and all 16 threads read the same small region, so plain loads broadcast out of L1 for
//     less than the two __syncthreads that staging costs (BS64/NR2: 16.6 us direct vs 17.7 LDS).
//   - small blocks win: BS32 14.6 us, BS64 ~15.5, BS128 ~17, BS256 18.7.

// Block geometry. The synthetic MUL_MAT_ID shape used during development (m=768, k=2048) and the
// shapes the model actually issues do not favour the same config, so this is env-selectable and
// tuned against llama-bench rather than against a microbenchmark.
// BS64/NR1 measured best on GLM-4.7-Flash (tg64@d0 83.16 t/s vs 80.55 for the BS32/NR2 that won
// the microbenchmark) - the model's shape mix is what matters, not the synthetic one.
#define MMVK_BS 64   // threads per block (default)
#define MMVK_NR 1    // rows per block (default)

static int mmvk_cfg() {
    static const int cfg = []() {
        const char * s = getenv("GGML_MMVK_CFG");
        return s ? atoi(s) : 0;
    }();
    return cfg;
}

// block_q6_K is 210 B and block_q4_K 144 B, neither a multiple of 4, so dword loads inside a
// block are not naturally aligned. Go through memcpy so the compiler emits a legal access.
static __device__ __forceinline__ uint32_t mmvk_ld_u32(const void * p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

// Accumulate one superblock for one row. Returns the contribution to the dot product.
template <ggml_type type>
static __device__ __forceinline__ float mmvk_dot_superblock(
        const void * __restrict__ vx, const float * __restrict__ y, const int itid);

template <>
__device__ __forceinline__ float mmvk_dot_superblock<GGML_TYPE_Q6_K>(
        const void * __restrict__ vx, const float * __restrict__ y, const int itid) {
    const block_q6_K * b = (const block_q6_K *) vx;

    const int v_im = itid / 8;          // 0 -> weights 0..127, 1 -> 128..255
    const int v_in = itid % 8;
    const int l0   = 4 * v_in;
    const int is   = v_in / 4;

    const int ql_offset = 64*v_im + l0;
    const int qh_offset = 32*v_im + l0;
    const int s_offset  =  8*v_im + is;
    const int y_offset  = 128*v_im + l0;

    const float sc0 = (float) b->scales[s_offset + 0];
    const float sc1 = (float) b->scales[s_offset + 2];
    const float sc2 = (float) b->scales[s_offset + 4];
    const float sc3 = (float) b->scales[s_offset + 6];

    const uint32_t ql0  = mmvk_ld_u32(b->ql + ql_offset);
    const uint32_t ql32 = mmvk_ld_u32(b->ql + ql_offset + 32);
    const uint32_t qh   = mmvk_ld_u32(b->qh + qh_offset);

    const uint32_t q0u = ( ql0        & 0x0F0F0F0F) | ((qh & 0x03030303) << 4);
    const uint32_t q1u = ( ql32       & 0x0F0F0F0F) | ((qh & 0x0C0C0C0C) << 2);
    const uint32_t q2u = ((ql0  >> 4) & 0x0F0F0F0F) | ( qh & 0x30303030);
    const uint32_t q3u = ((ql32 >> 4) & 0x0F0F0F0F) | ((qh & 0xC0C0C0C0) >> 2);

    const uint8_t * q0 = (const uint8_t *) &q0u;
    const uint8_t * q1 = (const uint8_t *) &q1u;
    const uint8_t * q2 = (const uint8_t *) &q2u;
    const uint8_t * q3 = (const uint8_t *) &q3u;

    const float4 by0  = *(const float4 *) (y + y_offset);
    const float4 by32 = *(const float4 *) (y + y_offset + 32);
    const float4 by64 = *(const float4 *) (y + y_offset + 64);
    const float4 by96 = *(const float4 *) (y + y_offset + 96);
    const float * b0 = (const float *) &by0;
    const float * b1 = (const float *) &by32;
    const float * b2 = (const float *) &by64;
    const float * b3 = (const float *) &by96;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        s0 = fmaf(b0[l], (float) q0[l] - 32.0f, s0);
        s1 = fmaf(b1[l], (float) q1[l] - 32.0f, s1);
        s2 = fmaf(b2[l], (float) q2[l] - 32.0f, s2);
        s3 = fmaf(b3[l], (float) q3[l] - 32.0f, s3);
    }
    return (float) b->d * (s0*sc0 + s1*sc1 + s2*sc2 + s3*sc3);
}

template <>
__device__ __forceinline__ float mmvk_dot_superblock<GGML_TYPE_Q4_K>(
        const void * __restrict__ vx, const float * __restrict__ y, const int itid) {
    const block_q4_K * b = (const block_q4_K *) vx;

    // 4 groups of 64 weights; each group uses two 6-bit scale/min pairs (low and high nibbles).
    const int g = itid / 4;              // 0..3, which 64-weight group
    const int t = itid % 4;              // 0..3, which 8 bytes of that group's 32
    const int qs_off = 32*g + 8*t;
    const int y_lo   = 64*g + 8*t;
    const int y_hi   = y_lo + 32;

    // get_scale_min_k4 for scale indices 2g (low nibbles) and 2g+1 (high nibbles)
    const uint8_t * sc = b->scales;
    uint8_t d_lo, m_lo, d_hi, m_hi;
    {
        const int j = 2*g;
        if (j < 4) { d_lo = sc[j] & 63; m_lo = sc[j + 4] & 63; }
        else       { d_lo = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                     m_lo = (sc[j + 4] >>   4) | ((sc[j    ] >> 6) << 4); }
    }
    {
        const int j = 2*g + 1;
        if (j < 4) { d_hi = sc[j] & 63; m_hi = sc[j + 4] & 63; }
        else       { d_hi = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                     m_hi = (sc[j + 4] >>   4) | ((sc[j    ] >> 6) << 4); }
    }

    const uint32_t q0 = mmvk_ld_u32(b->qs + qs_off);
    const uint32_t q1 = mmvk_ld_u32(b->qs + qs_off + 4);
    const uint8_t * qa = (const uint8_t *) &q0;
    const uint8_t * qb = (const uint8_t *) &q1;

    const float4 yl0 = *(const float4 *) (y + y_lo);
    const float4 yl1 = *(const float4 *) (y + y_lo + 4);
    const float4 yh0 = *(const float4 *) (y + y_hi);
    const float4 yh1 = *(const float4 *) (y + y_hi + 4);
    const float * pl0 = (const float *) &yl0;
    const float * pl1 = (const float *) &yl1;
    const float * ph0 = (const float *) &yh0;
    const float * ph1 = (const float *) &yh1;

    // sum of y*q and sum of y, per scale group: value = d*sc*q - dmin*m, so the min term needs sum(y)
    float slo = 0.0f, shi = 0.0f, ylo = 0.0f, yhi = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        slo = fmaf(pl0[l], (float) (qa[l] & 0x0F), slo);
        shi = fmaf(ph0[l], (float) (qa[l] >>   4), shi);
        ylo += pl0[l];
        yhi += ph0[l];
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        slo = fmaf(pl1[l], (float) (qb[l] & 0x0F), slo);
        shi = fmaf(ph1[l], (float) (qb[l] >>   4), shi);
        ylo += pl1[l];
        yhi += ph1[l];
    }

    const float2 dm = __half22float2(b->dm);
    return dm.x*((float) d_lo*slo + (float) d_hi*shi) - dm.y*((float) m_lo*ylo + (float) m_hi*yhi);
}

template <ggml_type type, int BS, int NR, bool has_fusion>
__launch_bounds__(BS, 1)
static __global__ void mul_mat_vec_k(
        const void * __restrict__ vx, const float * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x) {
    constexpr int WS = 32;               // gfx1201 runs HIP compute in wave32
    constexpr int it_size = BS / 16;     // superblocks in flight per row

    const int tid  = threadIdx.x;
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int nsb  = ncols_x / QK_K;
    const int row0 = NR * blockIdx.x;

    const uint32_t channel_dst = blockIdx.y;
    ggml_cuda_pdl_sync();
    const uint32_t channel_x  = ids ? (uint32_t) ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y  = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t sample_x   = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y   = sample_dst;

    const size_t bs_x = sizeof(block_q6_K); // unused; strides below are in blocks
    GGML_UNUSED(bs_x);

    typedef typename std::conditional<type == GGML_TYPE_Q6_K, block_q6_K, block_q4_K>::type blk_t;
    const blk_t * x    = (const blk_t *) vx + (size_t) sample_x*stride_sample_x + (size_t) channel_x*stride_channel_x;
    const blk_t * xg   = has_fusion && fusion.gate
                       ? (const blk_t *) fusion.gate + (size_t) sample_x*stride_sample_x + (size_t) channel_x*stride_channel_x
                       : nullptr;
    const float * y    = vy + (size_t) sample_y*stride_sample_y + (size_t) channel_y*stride_channel_y;

    float temp[NR]      = {0.0f};
    float temp_gate[NR] = {0.0f};

    for (int i = ix; i < nsb; i += it_size) {
        const float * yb = y + (size_t) i*QK_K;
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            const uint32_t row = row0 + n;
            if (row >= nrows_x) {
                break;
            }
            const size_t off = (size_t) row*stride_row_x + i;
            temp[n] += mmvk_dot_superblock<type>(x + off, yb, itid);
            if constexpr (has_fusion) {
                if (xg) {
                    temp_gate[n] += mmvk_dot_superblock<type>(xg + off, yb, itid);
                }
            }
        }
    }

    // wave32 shuffle reduction, then at most BS/WS partials through LDS
    __shared__ float red[(BS/WS > 0 ? BS/WS : 1)][NR];
    __shared__ float red_g[(BS/WS > 0 ? BS/WS : 1)][NR];
    const int lane = tid % WS;
    const int warp = tid / WS;

#pragma unroll
    for (int n = 0; n < NR; ++n) {
#pragma unroll
        for (int s = WS/2; s > 0; s >>= 1) {
            temp[n] += __shfl_xor(temp[n], s, WS);
            if constexpr (has_fusion) {
                temp_gate[n] += __shfl_xor(temp_gate[n], s, WS);
            }
        }
        if constexpr (BS > WS) {
            if (lane == 0) {
                red[warp][n] = temp[n];
                if constexpr (has_fusion) {
                    red_g[warp][n] = temp_gate[n];
                }
            }
        }
    }
    if constexpr (BS > WS) {
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int n = 0; n < NR; ++n) {
                float v = 0.0f, vg = 0.0f;
#pragma unroll
                for (int w = 0; w < BS/WS; ++w) {
                    v += red[w][n];
                    if constexpr (has_fusion) {
                        vg += red_g[w][n];
                    }
                }
                temp[n] = v;
                if constexpr (has_fusion) {
                    temp_gate[n] = vg;
                }
            }
        }
    }
    if (tid != 0) {
        return;
    }

    float * dst_row = dst + (size_t) sample_dst*stride_sample_dst + (size_t) channel_dst*stride_channel_dst + row0;

    const float * x_bias    = nullptr;
    const float * gate_bias = nullptr;
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (fusion.x_bias) {
            x_bias = (const float *) fusion.x_bias
                   + (size_t) sample_dst*stride_sample_dst + (size_t) channel_bias*stride_channel_dst + row0;
        }
        if (fusion.gate_bias) {
            gate_bias = (const float *) fusion.gate_bias
                      + (size_t) sample_dst*stride_sample_dst + (size_t) channel_bias*stride_channel_dst + row0;
        }
    }

#pragma unroll
    for (int n = 0; n < NR; ++n) {
        if ((uint32_t) (row0 + n) >= nrows_x || (uint32_t) (row0 + n) >= stride_col_dst) {
            break;
        }
        float result = temp[n];
        if constexpr (has_fusion) {
            if (x_bias) {
                result += x_bias[n];
            }
            if (fusion.gate) {
                float gate_value = temp_gate[n];
                if (gate_bias) {
                    gate_value += gate_bias[n];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    default:
                        break;
                }
            }
        }
        dst_row[n] = result;
    }
}

bool ggml_cuda_should_use_mmvk(enum ggml_type type, int cc, int64_t ncols_dst) {
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || ncols_dst != 1) {
        return false;
    }
    // Only the K-quants where the integer-dot path loses; q4_0/q8_0 stay on mmvq, which is faster
    // there than both this kernel and Vulkan.
    return type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q4_K;
}

template <ggml_type type>
static void mmvk_launch(
        const void * vx, const float * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t stride_row_x,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t nsamples_x, const int64_t nsamples_dst,
        const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    // Same guard as mmvq: with ids the channel comes from the id table so channel_ratio is unused
    // and would be a division by zero; without ids nchannels_y is unused. init_fastdiv_values
    // asserts on 0, so only build the one that is actually read.
    const uint3 nchannels_y_fd = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio  = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio   = init_fastdiv_values(nsamples_dst / nsamples_x);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

#define MMVK_DISPATCH(BS, NR)                                                                        \
    do {                                                                                             \
        const dim3 bn((nrows_x + (NR) - 1) / (NR), nchannels_dst, nsamples_dst);                     \
        const dim3 bd((BS), 1, 1);                                                                   \
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(bn, bd, 0, stream); \
        if (has_fusion) {                                                                            \
            ggml_cuda_kernel_launch(mul_mat_vec_k<type, (BS), (NR), true>, lp,                       \
                vx, vy, ids, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd,                        \
                (uint32_t) stride_row_x, (uint32_t) stride_col_dst, channel_ratio,                   \
                (uint32_t) stride_channel_x, (uint32_t) stride_channel_y,                            \
                (uint32_t) stride_channel_dst, sample_ratio, (uint32_t) stride_sample_x,             \
                (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst, (uint32_t) nrows_x);       \
        } else {                                                                                     \
            ggml_cuda_kernel_launch(mul_mat_vec_k<type, (BS), (NR), false>, lp,                      \
                vx, vy, ids, fusion, dst, (uint32_t) ncols_x, nchannels_y_fd,                        \
                (uint32_t) stride_row_x, (uint32_t) stride_col_dst, channel_ratio,                   \
                (uint32_t) stride_channel_x, (uint32_t) stride_channel_y,                            \
                (uint32_t) stride_channel_dst, sample_ratio, (uint32_t) stride_sample_x,             \
                (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst, (uint32_t) nrows_x);       \
        }                                                                                            \
    } while (0)

    switch (mmvk_cfg()) {
        case  1: MMVK_DISPATCH( 32, 1); break;
        case  2: MMVK_DISPATCH( 64, 1); break;
        case  3: MMVK_DISPATCH( 64, 2); break;
        case  4: MMVK_DISPATCH( 64, 4); break;
        case  5: MMVK_DISPATCH(128, 1); break;
        case  6: MMVK_DISPATCH(128, 2); break;
        case  7: MMVK_DISPATCH(128, 4); break;
        case  8: MMVK_DISPATCH(256, 2); break;
        case  9: MMVK_DISPATCH(256, 4); break;
        case 10: MMVK_DISPATCH( 32, 4); break;
        default: MMVK_DISPATCH(MMVK_BS, MMVK_NR); break;
    }
#undef MMVK_DISPATCH
}

void ggml_cuda_mul_mat_vec_k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_device & fusion) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    // src0 strides are in blocks; src1/dst strides in elements.
    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s3  = dst->nb[3] / ts_dst;

    // MUL_MAT_ID lays memory out differently from MUL_MAT, exactly as in mmvq.
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    switch (src0->type) {
        case GGML_TYPE_Q6_K:
            mmvk_launch<GGML_TYPE_Q6_K>(src0->data, src1_d, ids_d, fusion, dst_d, ne00, ne01, s01,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03, s13, s3, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            mmvk_launch<GGML_TYPE_Q4_K>(src0->data, src1_d, ids_d, fusion, dst_d, ne00, ne01, s01,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03, s13, s3, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("mmvk: unsupported type %s (ne0=%ld ne1=%ld ne2=%ld ids=%d)",
                       ggml_type_name(src0->type), (long) ne0, (long) ne1, (long) ne2, ids ? 1 : 0);
    }
}

#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static __device__ __forceinline__ uint64_t top_k_ballot(const bool pred) {
    return __ballot(pred);
}

// One block per row for rows of up to TOP_K_ROW_THREADS*e_max columns. The row is loaded once into registers as
// order-preserving keys, four 8-bit radix passes find the k-th largest key with shared-memory histograms (one per
// warp), and the selected columns are written in ascending column order, the ties at the threshold taking the lowest
// columns. One launch instead of the eleven of the multi-block path, and the same list on every run: that path
// writes through atomic counters, so the order of the list (and which of the tied columns are kept) changed from run
// to run, and the sparse QSA attention accumulates its cells in list order (a different greedy text on every load).
// Warp w owns the columns [w*e*warp_size, (w+1)*e*warp_size), element i of a lane is column
// w*e*warp_size + i*warp_size + lane (e = ceil(ncols / TOP_K_ROW_THREADS)), so the reads stay coalesced and column
// order is warp, then element, then lane.
#define TOP_K_ROW_THREADS 1024

template <int e_max>
static __global__ void __launch_bounds__(TOP_K_ROW_THREADS, 1) top_k_row(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = TOP_K_ROW_THREADS / warp_size;
    constexpr int NBINS     = 256;

    const int tid  = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int         * row_dst = dst + (size_t) blockIdx.x * k;

    __shared__ int hist[nwarps][NBINS];
    __shared__ int bins[NBINS];
    __shared__ int warp_counts[nwarps];
    __shared__ int s_bin;
    __shared__ int s_rank;

    const int e    = (ncols + TOP_K_ROW_THREADS - 1) / TOP_K_ROW_THREADS;
    const int base = warp*e*warp_size + lane;

    uint32_t key[e_max];
#pragma unroll
    for (int i = 0; i < e_max; ++i) {
        const int col = base + i*warp_size;
        key[i] = i < e && col < ncols ? top_k_float_to_ordered(row_src[col]) : 0;
    }
    const auto valid = [&](const int i) {
        return i < e && base + i*warp_size < ncols;
    };

    // the k-th largest key, 8 bits per pass; rank = how many keys equal to the prefix so far are still needed
    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int      rank   = k;
    for (int shift = 32 - 8; shift >= 0; shift -= 8) {
        for (int j = tid; j < nwarps*NBINS; j += TOP_K_ROW_THREADS) {
            (&hist[0][0])[j] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < e_max; ++i) {
            if (valid(i) && (key[i] & pmask) == prefix) {
                atomicAdd(&hist[warp][(key[i] >> shift) & (NBINS - 1)], 1);
            }
        }
        __syncthreads();
        if (tid < NBINS) {
            int c = 0;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                c += hist[w][tid];
            }
            bins[tid] = c;
        }
        __syncthreads();
        if (warp == 0) {
            // lane l holds the bins 255 - l*per_lane ... from the top; the crossing lane walks its bins
            constexpr int per_lane = NBINS / warp_size;
            int local[per_lane];
            int sum = 0;
#pragma unroll
            for (int j = 0; j < per_lane; ++j) {
                local[j] = bins[NBINS - 1 - (lane*per_lane + j)];
                sum += local[j];
            }
            const int incl = warp_prefix_inclusive_sum<int, warp_size>(sum);
            const int excl = incl - sum;
            if (excl < rank && rank <= incl) {
                int acc = excl;
#pragma unroll
                for (int j = 0; j < per_lane; ++j) {
                    if (acc + local[j] >= rank) {
                        s_bin  = NBINS - 1 - (lane*per_lane + j);
                        s_rank = rank - acc;
                        break;
                    }
                    acc += local[j];
                }
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_bin << shift;
        pmask  |= (uint32_t) (NBINS - 1) << shift;
        rank    = s_rank;
        __syncthreads();
    }
    const uint32_t thr = prefix;

    const uint64_t lanes_below = (((uint64_t) 1) << lane) - 1;

    // the ties before each warp (column order), then the selected columns before each warp
    int n_eq = 0;
#pragma unroll
    for (int i = 0; i < e_max; ++i) {
        n_eq += __popcll(top_k_ballot(valid(i) && key[i] == thr));
    }
    if (lane == 0) {
        warp_counts[warp] = n_eq;
    }
    __syncthreads();
    int eq_run = 0;
    for (int w = 0; w < warp; ++w) {
        eq_run += warp_counts[w];
    }
    __syncthreads();

    const int eq_base = eq_run;
    int n_sel = 0;
#pragma unroll
    for (int i = 0; i < e_max; ++i) {
        const bool     eq    = valid(i) && key[i] == thr;
        const uint64_t m_eq  = top_k_ballot(eq);
        const bool     sel   = (valid(i) && key[i] > thr) || (eq && eq_run + __popcll(m_eq & lanes_below) < rank);
        n_sel  += __popcll(top_k_ballot(sel));
        eq_run += __popcll(m_eq);
    }
    if (lane == 0) {
        warp_counts[warp] = n_sel;
    }
    __syncthreads();
    int pos = 0;
    for (int w = 0; w < warp; ++w) {
        pos += warp_counts[w];
    }

    eq_run = eq_base;
#pragma unroll
    for (int i = 0; i < e_max; ++i) {
        const bool     eq    = valid(i) && key[i] == thr;
        const uint64_t m_eq  = top_k_ballot(eq);
        const bool     sel   = (valid(i) && key[i] > thr) || (eq && eq_run + __popcll(m_eq & lanes_below) < rank);
        const uint64_t m_sel = top_k_ballot(sel);
        if (sel) {
            row_dst[pos + __popcll(m_sel & lanes_below)] = base + i*warp_size;
        }
        pos    += __popcll(m_sel);
        eq_run += __popcll(m_eq);
    }
}

// false when the row is too long for top_k_row
static bool top_k_row_cuda(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOP_K_ROW"); // 0: always the multi-block radix select
        return env == nullptr || atoi(env) != 0;
    }();
    const int e = (ncols + TOP_K_ROW_THREADS - 1) / TOP_K_ROW_THREADS;
    if (!enabled || e > 64) {
        return false;
    }
    const dim3 grid(nrows);
    const dim3 block(TOP_K_ROW_THREADS);
    if (e <= 8) {
        top_k_row<8><<<grid, block, 0, stream>>>(src, dst, ncols, k);
    } else if (e <= 16) {
        top_k_row<16><<<grid, block, 0, stream>>>(src, dst, ncols, k);
    } else if (e <= 32) {
        top_k_row<32><<<grid, block, 0, stream>>>(src, dst, ncols, k);
    } else {
        top_k_row<64><<<grid, block, 0, stream>>>(src, dst, ncols, k);
    }
    return true;
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}


bool ggml_cuda_top_k_rows(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    return top_k_row_cuda(src, dst, ncols, nrows, k, stream);
}
#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        if (!top_k_row_cuda(src0_d, dst_d, ncols, nrows, k, stream)) {
            top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        }
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

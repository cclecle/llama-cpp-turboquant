// Hybrid CPU+GPU variant of MUL_MAT_ID.
//
// The split is per expert and per output row: src0 + a*nb02 + r*nb01 is contiguous for any row
// range, and every (slot, token) pair belongs to exactly one expert, so the CPU set and the GPU set
// write disjoint dst rows. No partial sums, no reduction.
//
// A registered backend decides, for each expert, the output row the CPU stops at. One number covers
// the three placements: 0 gives the expert to the backend (weights resident in VRAM), ne01 keeps it
// on the CPU, and anything between shares it.
//
// The stock kernel in ggml-cpu.c is left untouched; this one is selected only when the operator
// asks for it.

#include "mul-mat-id-hybrid.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"

// ggml_graph_plan sizes the MUL_MAT_ID scratch in a C translation unit, where ops.h resolves the
// cache line size to a fixed number. In C++ the same macro resolves to
// std::hardware_destructive_interference_size, which can differ. Pin the C value so the two
// layouts cannot drift apart.
#if defined(__POWER9_VECTOR__)
#define HYBRID_CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define HYBRID_CACHE_LINE_SIZE 256
#else
#define HYBRID_CACHE_LINE_SIZE 64
#endif

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// operator control
// ---------------------------------------------------------------------------

static bool hybrid_parse_on(const char * s) {
    if (s == nullptr || s[0] == '\0') {
        return false;
    }
    return strcmp(s, "off") != 0 && strcmp(s, "0") != 0 && strcmp(s, "false") != 0;
}

static bool hybrid_requested() {
    static const bool on = hybrid_parse_on(getenv("GGML_MOE_HYBRID"));
    return on;
}

// ---------------------------------------------------------------------------
// routing census (GGML_MOE_HYBRID_STATS=1)
//
// Answers the question the split depends on: how skewed is expert routing, i.e. how many bytes of
// expert weight must be reachable to cover a given share of the decode hits.
// ---------------------------------------------------------------------------

static bool hybrid_stats_enabled() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STATS");
        return s != nullptr && s[0] != '\0' && s[0] != '0';
    }();
    return on;
}

// only batches this small are counted, so prefill routing does not swamp the decode census
static int64_t hybrid_stats_max_tokens() {
    static const int64_t n = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STATS_MAX_TOKENS");
        return (s != nullptr && s[0] != '\0') ? atoll(s) : (int64_t) 8;
    }();
    return n;
}

struct hybrid_stats_entry {
    std::string          name;
    size_t               bytes_per_expert = 0;
    int64_t              n_calls          = 0;
    int64_t              us               = 0; // wall time of the node, seen by thread 0
    int64_t              weight_bytes     = 0; // expert weight touched, one matrix per used expert
    int64_t              cpu_rows         = 0; // output rows computed on the CPU
    int64_t              gpu_rows         = 0; // output rows taken by the backend
    int64_t              us_launch        = 0; // thread 0 issuing the backend's share
    int64_t              us_wait          = 0; // thread 0 waiting for it after its own rows
    std::vector<int64_t> rows;
};

struct hybrid_stats {
    std::mutex mutex;
    // keyed by tensor pointer: hashing the name on every node costs more than the census measures
    std::unordered_map<const ggml_tensor *, hybrid_stats_entry> per_tensor;

    ~hybrid_stats() {
        dump();
    }

    void dump();
};

static hybrid_stats & hybrid_stats_get() {
    static hybrid_stats stats;
    return stats;
}

static void hybrid_stats_record(
        const ggml_tensor * src0, const ggml_tensor * ids,
        const int64_t * matrix_row_counts, int64_t n_as,
        const int64_t * cpu_row_end, int64_t ne01) {
    if (ids->ne[1] > hybrid_stats_max_tokens()) {
        return;
    }

    hybrid_stats & stats = hybrid_stats_get();
    std::lock_guard<std::mutex> lock(stats.mutex);

    hybrid_stats_entry & e = stats.per_tensor[src0];
    if (e.rows.empty()) {
        e.rows.resize(n_as, 0);
        e.bytes_per_expert = src0->nb[2];
        e.name             = src0->name;
    }
    if ((int64_t) e.rows.size() != n_as) {
        return;
    }

    e.n_calls++;
    for (int64_t a = 0; a < n_as; ++a) {
        e.rows[a] += matrix_row_counts[a];
        if (matrix_row_counts[a] > 0) {
            // only the rows the CPU actually keeps: counting the backend's share here once
            // inflated the census 2.7x and its GB/s past the hardware ceiling, which sent a whole
            // debugging session chasing bytes that were never read
            e.weight_bytes += (int64_t) ((double) e.bytes_per_expert * cpu_row_end[a] / (double) ne01);
        }
    }
}

// How the output rows of this node were shared, recorded once the split is known. Gated on the same
// batch size as the row census: mixing prefill time with decode bytes would make the derived
// bandwidth meaningless.
static void hybrid_stats_record_split(
        const ggml_tensor * src0, const ggml_tensor * ids, int64_t cpu_rows, int64_t gpu_rows,
        int64_t us, int64_t us_launch, int64_t us_wait) {
    if (ids->ne[1] > hybrid_stats_max_tokens()) {
        return;
    }
    hybrid_stats & stats = hybrid_stats_get();
    std::lock_guard<std::mutex> lock(stats.mutex);

    auto it = stats.per_tensor.find(src0);
    if (it != stats.per_tensor.end()) {
        it->second.us        += us;
        it->second.cpu_rows  += cpu_rows;
        it->second.gpu_rows  += gpu_rows;
        it->second.us_launch += us_launch;
        it->second.us_wait   += us_wait;
    }
}

void hybrid_stats::dump() {
    std::lock_guard<std::mutex> lock(mutex);
    if (per_tensor.empty()) {
        return;
    }

    // the logger may already be gone at static destruction, so write straight to stderr
    fprintf(stderr, "\n[moe-hybrid] routing census, batches of at most %" PRId64 " tokens\n",
            hybrid_stats_max_tokens());

    struct expert_hits {
        int64_t rows;
        size_t  bytes;
    };

    std::vector<expert_hits> all;
    int64_t total_rows          = 0;
    size_t  total_bytes         = 0;
    int64_t total_us            = 0;
    int64_t total_touched_bytes = 0;
    int64_t total_calls         = 0;
    int64_t total_cpu_rows      = 0;
    int64_t total_gpu_rows      = 0;

    std::vector<const hybrid_stats_entry *> entries;
    for (const auto & [key, e] : per_tensor) {
        entries.push_back(&e);
    }
    std::sort(entries.begin(), entries.end(),
              [](const hybrid_stats_entry * a, const hybrid_stats_entry * b) { return a->name < b->name; });

    fprintf(stderr, "[moe-hybrid]   %-38s %7s %6s %8s %10s %9s %8s %7s\n",
            "tensor", "experts", "used", "calls", "rows", "ms", "GB/s", "gpu%");

    for (const hybrid_stats_entry * ep : entries) {
        const hybrid_stats_entry & e = *ep;

        int64_t tensor_rows = 0;
        int64_t n_touched   = 0;
        for (int64_t rows : e.rows) {
            tensor_rows += rows;
            n_touched   += rows > 0 ? 1 : 0;
            all.push_back({rows, e.bytes_per_expert});
        }
        total_rows          += tensor_rows;
        total_bytes         += e.rows.size() * e.bytes_per_expert;
        total_us            += e.us;
        total_touched_bytes += e.weight_bytes;
        total_calls         += e.n_calls;
        total_cpu_rows      += e.cpu_rows;
        total_gpu_rows      += e.gpu_rows;

        const int64_t split_rows = e.cpu_rows + e.gpu_rows;

        fprintf(stderr, "[moe-hybrid]   %-38s %7zu %6" PRId64 " %8" PRId64 " %10" PRId64 " %9.1f %8.1f %6.1f%%\n",
                e.name.c_str(), e.rows.size(), n_touched, e.n_calls, tensor_rows,
                e.us/1000.0, e.us > 0 ? e.weight_bytes/(double) e.us/1000.0 : 0.0,
                split_rows > 0 ? 100.0*e.gpu_rows/(double) split_rows : 0.0);
    }

    if (total_rows == 0) {
        return;
    }

    fprintf(stderr, "[moe-hybrid]   CPU expert compute: %.1f ms over %" PRId64 " node calls, "
            "%.2f GiB of weight read, %.1f GB/s effective\n",
            total_us/1000.0, total_calls, total_touched_bytes/1024.0/1024.0/1024.0,
            total_us > 0 ? total_touched_bytes/(double) total_us/1000.0 : 0.0);

    const int64_t split_rows = total_cpu_rows + total_gpu_rows;
    if (split_rows > 0) {
        int64_t total_launch = 0;
        int64_t total_wait   = 0;
        for (const hybrid_stats_entry * ep : entries) {
            total_launch += ep->us_launch;
            total_wait   += ep->us_wait;
        }
        fprintf(stderr, "[moe-hybrid]   output rows: %" PRId64 " CPU, %" PRId64 " GPU (%.1f%% on GPU); "
                "thread 0 spent %.1f ms issuing and %.1f ms waiting for the GPU\n",
                total_cpu_rows, total_gpu_rows, 100.0*total_gpu_rows/(double) split_rows,
                total_launch/1000.0, total_wait/1000.0);
    }

    std::sort(all.begin(), all.end(),
              [](const expert_hits & a, const expert_hits & b) { return a.rows > b.rows; });

    // bytes that must be reachable to serve a given share of all routed rows
    const double targets[] = {0.5, 0.8, 0.9, 0.95};
    size_t  i     = 0;
    int64_t acc   = 0;
    size_t  bytes = 0;

    fprintf(stderr, "[moe-hybrid]   total %" PRId64 " rows over %8.2f MiB of expert weight\n",
            total_rows, total_bytes/1024.0/1024.0);

    for (double t : targets) {
        const int64_t want = (int64_t) (total_rows * t);
        while (i < all.size() && acc < want) {
            acc   += all[i].rows;
            bytes += all[i].bytes;
            i++;
        }
        fprintf(stderr, "[moe-hybrid]   %3.0f%% of rows served by %8.2f MiB (%5.1f%% of the weight)\n",
                t*100.0, bytes/1024.0/1024.0, 100.0*bytes/(double) total_bytes);
    }
}

// ---------------------------------------------------------------------------

bool ggml_mul_mat_id_hybrid_enabled(void) {
    return hybrid_requested() || hybrid_stats_enabled();
}

static void * hybrid_incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void hybrid_fill_node(
        struct ggml_moe_hybrid_node * node,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const int64_t * row_counts, const mmid_row_mapping * matrix_rows) {
    node->src0        = src0;
    node->src1        = src1;
    node->ids         = ids;
    node->dst         = dst;
    node->row_counts  = row_counts;
    node->matrix_rows = matrix_rows;
    node->rows_stride = ids->ne[0]*ids->ne[1];
}

void ggml_compute_forward_mul_mat_id_hybrid(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool    stats = hybrid_stats_enabled();
    const int64_t t_us  = (stats && params->ith == 0) ? ggml_time_us() : 0;

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    const enum ggml_type    vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const ggml_from_float_t from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        hybrid_incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        (int64_t *) hybrid_incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        (struct mmid_row_mapping *) hybrid_incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[HYBRID_CACHE_LINE_SIZE] = // [n_as]
        (char (*)[HYBRID_CACHE_LINE_SIZE]) hybrid_incr_ptr_aligned(&wdata_cur, HYBRID_CACHE_LINE_SIZE * n_as, HYBRID_CACHE_LINE_SIZE);

    // [n_as] the output row each expert's CPU share stops at. Written by thread 0, published by the
    // barrier below. ggml_graph_plan reserves this; the stock kernel never touches it.
    int64_t * cpu_row_end =
        (int64_t *) hybrid_incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = (char *) params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = mmid_row_mapping{id, (int32_t) iid1};
                matrix_row_counts[i02] += 1;
            }
        }

        // by default the CPU keeps every row
        for (int a = 0; a < n_as; ++a) {
            cpu_row_end[a] = ne01;
        }

        const ggml_moe_hybrid_api * api = hybrid_requested() ? ggml_moe_hybrid_get() : nullptr;
        if (api != nullptr) {
            ggml_moe_hybrid_node node;
            hybrid_fill_node(&node, src0, src1, ids, dst, matrix_row_counts, matrix_rows);
            if (api->supports(&node)) {
                api->dispatch(&node, cpu_row_end);
            }
        }

        // after dispatch, so the census can count only the rows the CPU keeps
        if (stats) {
            hybrid_stats_record(src0, ids, matrix_row_counts, n_as, cpu_row_end, ne01);
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        std::atomic<int> * current_chunk_ctr = (std::atomic<int> *)(atomic_current_chunk + cur_a);
        current_chunk_ctr->store(nth, std::memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

    // cpu_row_end is visible to every thread now, so both shares agree without further
    // synchronization. A thread that disagreed here would silently drop rows.
    int64_t gpu_rows = 0;
    int64_t cpu_rows = 0;

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 > 0) {
            gpu_rows += (ne01 - cpu_row_end[cur_a]) * cne1;
        }
    }

    // issue the backend's share now that the other threads are already busy on theirs
    int64_t us_launch = 0;
    int64_t us_wait   = 0;

    if (gpu_rows > 0 && ith == 0) {
        const ggml_moe_hybrid_api * api = ggml_moe_hybrid_get();
        if (api != nullptr) {
            const int64_t t0 = stats ? ggml_time_us() : 0;
            ggml_moe_hybrid_node node;
            hybrid_fill_node(&node, src0, src1, ids, dst, matrix_row_counts, matrix_rows);
            api->launch(&node);
            us_launch = stats ? ggml_time_us() - t0 : 0;
        }
    }

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const int64_t nr0 = cpu_row_end[cur_a];

        cpu_rows += nr0 * cne1;

        if (nr0 == 0) {
            continue; // taken entirely by the backend
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        std::atomic<int> * current_chunk_ctr = (std::atomic<int> *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = current_chunk_ctr->fetch_add(1, std::memory_order_relaxed);
        }
    }

    // the backend writes dst rows too, so nobody may leave this node before it is done. gpu_rows
    // comes from shared state, so this branch is uniform across threads.
    if (gpu_rows > 0) {
        if (ith == 0) {
            const ggml_moe_hybrid_api * api = ggml_moe_hybrid_get();
            if (api != nullptr) {
                // how long thread 0 blocks here says whether the backend is keeping up: near zero
                // means the CPU is still the limit, large means the GPU became the new one
                const int64_t t0 = stats ? ggml_time_us() : 0;
                ggml_moe_hybrid_node node;
                hybrid_fill_node(&node, src0, src1, ids, dst, matrix_row_counts, matrix_rows);
                api->join(&node);
                us_wait = stats ? ggml_time_us() - t0 : 0;
            }
        }
        ggml_barrier(params->threadpool);
    }

    if (t_us != 0) {
        hybrid_stats_record_split(src0, ids, cpu_rows, gpu_rows, ggml_time_us() - t_us, us_launch, us_wait);
    }

#undef MMID_MATRIX_ROW
}

// VRAM residency cache for MoE experts kept in host RAM.
//
// The CPU backend owns MUL_MAT_ID. This file keeps the hottest experts resident in VRAM and,
// when the CPU kernel asks, computes their output rows on the GPU instead. Rows are disjoint in
// dst, so the two sides never overlap and nothing has to be reduced.
//
// Rating and migration run on different clocks: demand is scored every node, the pool is rewritten
// only every few hundred tokens, so admission never churns inside a generation step.
//
// The dispatch reuses the stock MUL_MAT_ID mat-vec by describing the pool as a normal
// [ne00, ne01, n_slots] weight tensor and remapping ids to pool slots. That keeps this file free of
// quantisation kernels and picks up mmvk on RDNA4 K-quants for free.

#include "moe-hybrid.cuh"
#include "common.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-moe-hybrid.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

// (size_t)-1 means "take whatever VRAM is free"; anything else is an explicit cap, and 0 disables
// the residency pool entirely, which is useful for measuring the other mechanisms on their own.
static size_t moe_hybrid_budget_bytes() {
    static const size_t bytes = [] {
        const char * s = getenv("GGML_MOE_HYBRID_VRAM");
        if (s == nullptr || s[0] == '\0' || strcmp(s, "auto") == 0) {
            return (size_t) -1;
        }
        return (size_t) atoll(s) * 1024 * 1024;
    }();
    return bytes;
}

// Per-device ceiling for "auto", and it is deliberately tiny. Measured on two R9700s with
// speculation: a 512 MiB total pool ran Qwen3.5-122B decode at 41.9 t/s against 22.1 with no pool,
// while 2 GiB ran 22.9 and 8 GiB ran 21.9 - i.e. more VRAM for the pool is worse, because the rest
// of the graph wants it more than the extra coverage is worth. 15% of expert visits served from
// VRAM beat 38%. Raise it per model only with a measurement in hand.
// Number of experts per tensor to hold in VRAM. -1 means "as many as fit", which is the right
// default here: one model runs at a time, so VRAM the experts do not take is VRAM nobody uses.
static int64_t moe_hybrid_n_experts() {
    static const int64_t n = [] {
        const char * s = getenv("GGML_MOE_HYBRID_EXPERTS");
        if (s == nullptr || s[0] == '\0' || strcmp(s, "auto") == 0) {
            return (int64_t) -1;
        }
        return (int64_t) atoll(s);
    }();
    return n;
}

// Optional ceiling for the automatic size, for the case where something else needs the VRAM.
// Unset means no ceiling.
static size_t moe_hybrid_auto_cap() {
    static const size_t bytes = [] {
        const char * s = getenv("GGML_MOE_HYBRID_AUTO_CAP_MB");
        if (s == nullptr || s[0] == '\0') {
            return (size_t) -1;
        }
        return (size_t) atoll(s) * 1024 * 1024;
    }();
    return bytes;
}

// keep this much VRAM free for the KV cache and compute buffers when sizing automatically
static size_t moe_hybrid_reserve_bytes() {
    static const size_t bytes = [] {
        const char * s = getenv("GGML_MOE_HYBRID_RESERVE_MB");
        return (size_t) ((s != nullptr && s[0] != '\0') ? atoll(s) : 512) * 1024 * 1024;
    }();
    return bytes;
}

static int64_t moe_hybrid_migrate_tokens() {
    static const int64_t n = [] {
        const char * s = getenv("GGML_MOE_HYBRID_MIGRATE");
        return (int64_t) ((s != nullptr && s[0] != '\0') ? atoll(s) : 512);
    }();
    return n;
}

static bool moe_hybrid_verbose() {
    static const bool on = getenv("GGML_MOE_HYBRID_VERBOSE") != nullptr;
    return on;
}

// Share of a NON-resident expert's output rows the GPU takes, reading the weights straight out of
// host RAM over PCIe while the CPU works on the rest of the same expert. 0 disables class 2.
// The two readers share one memory controller (64 GB/s CPU alone, ~78 GB/s together), so the
// balanced point is near 0.18, not 0.5.
static bool moe_hybrid_stream_auto() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STREAM");
        return s != nullptr && strcmp(s, "auto") == 0;
    }();
    return on;
}

static double moe_hybrid_stream_frac() {
    static const double f = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STREAM");
        if (s == nullptr || s[0] == '\0') {
            return 0.0;
        }
        if (strcmp(s, "auto") == 0) {
            return 0.10; // starting point for the controller
        }
        const double v = atof(s);
        return v < 0.0 ? 0.0 : (v > 0.9 ? 0.9 : v);
    }();
    return f;
}

// Prefill and decode are different problems and want different answers. At decode the GPU is idle
// enough that reading host weights over PCIe adds bandwidth the CPU could not have reached alone.
// At prefill the GPU is already busy with attention and the dense layers, so the same work displaces
// work on the critical path: measured on Mistral-119B, class 2 at 0.30 took decode from 16.5 to 21.9
// while prefill fell from 373 to 300. So the share is a function of batch size, and defaults to
// nothing at prefill.
static double moe_hybrid_stream_prefill_frac() {
    static const double f = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STREAM_PREFILL");
        if (s == nullptr || s[0] == '\0') {
            return 0.0;
        }
        const double v = atof(s);
        return v < 0.0 ? 0.0 : (v > 0.9 ? 0.9 : v);
    }();
    return f;
}

// Whether the per-expert slots serve decode-sized nodes at all. With block promotion carrying
// decode (whole blocks native on the GPU), the remaining blocks are cheapest as pure CPU: the
// ~0.29 ms of per-node dispatch overhead is larger than what the slots save at batch 1. The slots
// then only serve batched (prefill) nodes, where MMQ amortizes everything.
static bool moe_hybrid_decode_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_DECODE");
        return s == nullptr || s[0] == '\0' || (strcmp(s, "off") != 0 && s[0] != '0');
    }();
    return on;
}

// Whether the facade graph path is enabled (mirrors the builder gate in src/models/deepseek2.cpp).
// The unified store depends on it: facade_run is what learns the shard geometry, so with the
// facade off the pool must keep double-storing nothing - it IS the store.
static bool moe_hybrid_facade_env_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_HYBRID_FACADE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

// Replay the per-node decode sequence as a HIP graph: one launch call instead of four API calls
// (copy, quantize, mat-vec, event). Captured lazily per (tensor, n_pairs) after two uncaptured
// runs have warmed the allocation pool - allocating inside a capture is this fork's known crash
// class. Pointer stability holds because the scratch and pool pointers are fixed per device and
// migration changes slot contents, not addresses.
// DEFAULT OFF: on ROCm the replayed sequence hit two distinct correctness hazards (a captured
// H2D that does not re-read host content, then "operation not permitted on an event last recorded
// in a capturing stream" from the live-recorded completion event). Opt-in for further work only.
static bool moe_hybrid_graphs_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_GRAPHS");
        return s != nullptr && s[0] != '\0' && strcmp(s, "off") != 0 && s[0] != '0';
    }();
    return on;
}

// Tokens in the node at or below which it counts as decode. A speculative verify batch is a dozen
// or so tokens and is still bandwidth bound, so it belongs on the decode side of this line.
static int64_t moe_hybrid_stream_batch() {
    static const int64_t n = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STREAM_BATCH");
        return (int64_t) ((s != nullptr && s[0] != '\0') ? atoll(s) : 32);
    }();
    return n;
}

static bool moe_hybrid_dma() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_DMA");
        return s != nullptr && s[0] != '\0' && strcmp(s, "off") != 0 && s[0] != '0';
    }();
    return on;
}

// Debug: synchronize and check after every class-1 sub-launch, printing its parameters, so an
// async illegal access names its exact launch instead of surfacing at the next event poll.
// Costs a sync per launch - diagnosis only, never benchmarking.
static bool moe_hybrid_sync_debug() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_SYNC");
        return s != nullptr && s[0] != '\0' && s[0] != '0';
    }();
    return on;
}

static void moe_debug_checkpoint(const char * what, const char * tname, int device,
                                 int64_t n_pairs, int64_t n_rows, bool batched) {
    if (!moe_hybrid_sync_debug()) {
        return;
    }
    ggml_cuda_set_device(device);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[moe-hybrid][SYNCDBG] FAULT at %s: tensor=%s dev=%d n_pairs=%" PRId64
                " n_rows=%" PRId64 " batched=%d: %s\n",
                what, tname, device, n_pairs, n_rows, (int) batched, cudaGetErrorString(err));
        fflush(stderr);
        GGML_ABORT("moe-hybrid sync-debug fault");
    }
}

// env GGML_MOE_HYBRID_NANCHECK: after every facade node and every hybrid join, scan the output
// (and its input, for attribution) and report the FIRST node whose input is sane but whose
// output is non-finite, exploded, or all-zero - the localizer for deterministic corruption.
static bool moe_hybrid_nancheck_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_NANCHECK");
        return s != nullptr && s[0] != '\0' && s[0] != '0';
    }();
    return on;
}

// returns 0 = sane, 1 = non-finite, 2 = exploded (>1e8), 3 = all-zero
static int moe_nan_scan(const float * p, int64_t n, int64_t * bad_i, float * bad_v) {
    bool any_nonzero = false;
    for (int64_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (v != 0.0f) {
            any_nonzero = true;
        }
        if (!std::isfinite(v)) { *bad_i = i; *bad_v = v; return 1; }
        if (fabsf(v) > 1e8f)   { *bad_i = i; *bad_v = v; return 2; }
    }
    if (!any_nonzero && n > 0) { *bad_i = 0; *bad_v = 0.0f; return 3; }
    return 0;
}

static const char * moe_nan_st(int s) {
    static const char * st[4] = { "ok", "nonfinite", "exploded", "allzero" };
    return st[s];
}

// ring of the last few scanned calls, healthy or not, dumped when the first corruption appears -
// it shows exactly which node preceded the first bad input
struct moe_nan_ring_ent { char where[8]; char tname[48]; int dev; int in_s; int out_s; int64_t call; };
static moe_nan_ring_ent g_nan_ring[8];
static int64_t g_nan_ring_n = 0;

static void moe_nan_ring_add(const char * where, const char * tname, int dev,
                             int in_s, int out_s, int64_t call_no) {
    moe_nan_ring_ent & e = g_nan_ring[g_nan_ring_n++ % 8];
    snprintf(e.where, sizeof(e.where), "%s", where);
    snprintf(e.tname, sizeof(e.tname), "%s", tname);
    e.dev = dev; e.in_s = in_s; e.out_s = out_s; e.call = call_no;
}

static void moe_nan_report(const char * where, const char * tname, int dev, int64_t call_no,
                           int in_state, int out_state, int64_t bad_i, float bad_v) {
    static int printed = 0;
    static bool ring_dumped = false;
    static std::set<std::string> seen;
    if (!ring_dumped) {
        ring_dumped = true;
        fprintf(stderr, "[moe-nan] ---- last %d calls before first corruption ----%c",
                (int) std::min<int64_t>(g_nan_ring_n, 8), 10);
        for (int64_t k = std::max<int64_t>(0, g_nan_ring_n - 8); k < g_nan_ring_n; ++k) {
            const moe_nan_ring_ent & e = g_nan_ring[k % 8];
            fprintf(stderr, "[moe-nan]   call %lld %s %s dev %d: in %s out %s%c",
                    (long long) e.call, e.where, e.tname, e.dev,
                    moe_nan_st(e.in_s), moe_nan_st(e.out_s), 10);
        }
    }
    if (printed >= 60 || !seen.insert(std::string(where) + tname + std::to_string(dev)).second) {
        return;
    }
    printed++;
    fprintf(stderr, "[moe-nan] %s %s dev %d call %lld: input %s -> output %s (i=%lld v=%g)%c",
            where, tname, dev, (long long) call_no, moe_nan_st(in_state), moe_nan_st(out_state),
            (long long) bad_i, (double) bad_v, 10);
    fflush(stderr);
}

// per-stage GPU event timing costs a few us per launch, so it rides the census switch
static bool moe_hybrid_stats_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_MOE_HYBRID_STATS");
        return s != nullptr && s[0] != '\0' && s[0] != '0';
    }();
    return on;
}

// Routing trace for the predictor analysis: every single-token node appends (tensor, expert ids)
// in dispatch order, written as TSV at shutdown. Feeds the offline miss-rate comparison of the
// reactive policy against LRU and the Belady bound. Off unless a path is given.
static const char * moe_hybrid_trace_path() {
    static const char * p = getenv("GGML_MOE_HYBRID_TRACE");
    return (p != nullptr && p[0] != '\0') ? p : nullptr;
}

// half-life of the demand score, in migration windows
// Weight kept from the previous window when folding in the new one. A 3000-token routing trace
// showed the demand is near-stationary: the long-run top-72 misses 9.0% of visits while the
// reactive EWMA missed 17.4%, and even the Belady bound (5.5%) pays more in migration traffic
// than the misses cost. So the score should converge toward long-run frequency: at 0.98 with
// 256-token passes the half-life is ~8.8k tokens - stable within a topic, still adapting across
// requests. (History: 0.5 originally, one-pass half-life, churned half the pool; then 0.875.)
static double moe_hybrid_decay() {
    static const double d = [] {
        const char * s = getenv("GGML_MOE_HYBRID_DECAY");
        return (s != nullptr && s[0] != '\0') ? atof(s) : 0.98;
    }();
    return d;
}

// How much better a challenger must score than a resident before it is worth paying for the swap.
// Most churn is near-ties trading places, which costs bandwidth and buys nothing.
static double moe_hybrid_hysteresis() {
    static const double h = [] {
        const char * s = getenv("GGML_MOE_HYBRID_HYSTERESIS");
        return (s != nullptr && s[0] != '\0') ? atof(s) : 1.5;
    }();
    return h;
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

struct moe_slot {
    int32_t expert = -1;    // expert currently held, -1 when free
    bool    ready  = false; // false while its copy is still in flight
};

struct moe_graph_entry {
    cudaGraphExec_t exec = nullptr;
    int    seen = 0;
    size_t gen  = 0;
};

// One contiguous run of a shard's rows in origin-row space. gate_up shards are segmented (the
// fused gate and up halves split independently), so a shard is a handful of runs, and the join
// can scatter a pair's output with one memcpy per run instead of one store per element.
struct moe_facade_seg {
    int64_t shard_r;  // first row inside the shard
    int64_t origin_r; // the origin row it maps to
    int64_t len;
};

struct moe_tensor_state {
    const ggml_tensor * src0 = nullptr;
    std::string name;
    std::unordered_map<int64_t, moe_graph_entry> dec_graphs; // key: n_pairs of the decode node

    // Pointer-table decode (the universal compute buffer): one device-side table of n_expert
    // base pointers - a resident expert points at its VRAM slot, a missing one at the mapped
    // host address of its rows. Refreshed only at migration boundaries, so a GPU-owned node can
    // dereference experts with no CPU involvement and no copies. Kept per home device today;
    // becomes per-device half-tables when the pool moves to half-slots for meta sharding.
    void * ptr_table_dev  = nullptr;  // [n_expert] void* on t->dev_idx's device
    bool   ptr_table_dirty = true;    // set by migration; uploaded lazily before use

    // facade path: per-device shard data (the meta-sharded facade allocation, once known) and the
    // per-device pointer tables the facade kernels dereference
    void * facade_data [GGML_CUDA_MAX_DEVICES] = { nullptr };
    int64_t facade_row_off[GGML_CUDA_MAX_DEVICES] = { -1, -1 }; // learned from the load-time marker
    void *  facade_rowmap [GGML_CUDA_MAX_DEVICES] = { nullptr }; // [shard_rows] int32 origin row per shard row
    int64_t facade_col_off[GGML_CUDA_MAX_DEVICES] = { 0 };       // byte offset into each origin row (contraction split)
    std::vector<int32_t> facade_slot_of;                          // [n_expert] -> facade slot, -1 when host-resident
    void * facade_table[GGML_CUDA_MAX_DEVICES] = { nullptr };
    bool   facade_dirty[GGML_CUDA_MAX_DEVICES] = { true, true };

    // Decode-side demand: the facade graph path never reaches the CPU dispatch that scores
    // routing, so its prep kernel counts expert visits here (device 0's shard only - every
    // device sees the same ids). Harvested into `window` when the re-fill asks for a ranking.
    void * facade_counts = nullptr; // [n_expert] int32 on device 0, atomically incremented

    // The unified store: facade twins exist for this tensor (noted at load), so its residents
    // live in the facade shards and the pool must not double-store them. The shard geometry below
    // is captured host-side when facade_run learns it from the markers, and is what the
    // prefill/verify launches use to read the shards and scatter their output rows.
    bool    has_facade = false;
    bool    facade_is_col = false; // contraction split (down): shards are column slices, partial sums
    int64_t facade_shard_rows[GGML_CUDA_MAX_DEVICES] = { 0 };
    int64_t facade_shard_cols[GGML_CUDA_MAX_DEVICES] = { 0 };
    int64_t facade_n_slots = 0;
    std::vector<moe_facade_seg> facade_segs_host[GGML_CUDA_MAX_DEVICES];
    ggml_type type  = GGML_TYPE_COUNT;
    int64_t n_expert = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t  expert_bytes = 0;   // nb02: one expert matrix

    std::vector<double>   rating;   // [n_expert] decayed demand
    std::vector<int64_t>  window;   // [n_expert] demand in the current window
    std::vector<int32_t>  slot_of;  // [n_expert] -> slot index, -1 when not resident
    std::vector<moe_slot> slots;

    int    dev_idx = 0;             // which device holds this tensor's slots
    char * pool = nullptr;          // device storage for slots, n_slots*expert_bytes

    // class 2: the host weights, made visible to the device so a kernel can read them over PCIe
    char * host_dev = nullptr;      // device-side view of src0->data, null when not registered
    int    host_dev_idx = -1;       // which device that pointer is for
    bool   host_tried = false;

    ggml_tensor host_desc {};       // src0->data as [ne00, ne01-row0, n_expert], offset to row0
    int64_t     host_row0 = -1;     // the row the descriptor starts at

    // describes the pool as a normal weight tensor so the stock mat-vec can consume it
    ggml_tensor pool_desc {};
};

// Per-stage GPU timing for one launch: ev[0] before the H2D copies, ev[1] after them, ev[2] after
// the compute, ev[3] after the D2H. Read back in join() once the stream is synced. One instance
// per (device, class) so class 1 and class 2 on the same stream cannot overwrite each other.
struct moe_timing {
    cudaEvent_t ev[4] = { nullptr, nullptr, nullptr, nullptr };
    bool valid = false;

    void ensure() {
        if (ev[0] == nullptr) {
            for (int i = 0; i < 4; ++i) {
                CUDA_CHECK(cudaEventCreate(&ev[i]));
            }
        }
    }
};

struct moe_scratch {
    void * dev      = nullptr;
    void * host     = nullptr;
    void * host_dev = nullptr; // device-side view of the pinned host block, for direct kernel writes
    size_t size = 0;
    size_t gen  = 0;           // bumped on reallocation; captured graphs bake pointers and must die

    void reserve(size_t n) {
        if (n <= size) {
            return;
        }
        gen++;
        if (dev) {
            CUDA_CHECK(cudaFree(dev));
        }
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
        }
        size = n + n/2;
        CUDA_CHECK(cudaMalloc(&dev, size));
        CUDA_CHECK(cudaMallocHost(&host, size));
        // decode nodes let the mat-vec write results straight here over PCIe (a few tens of KB),
        // which removes the device-to-host copy call from the per-node sequence entirely
        if (cudaHostGetDevicePointer(&host_dev, host, 0) != cudaSuccess) {
            host_dev = nullptr;
        }
    }
};

// Every expert tensor is pinned to one device, round robin. Each layer sees every token, so the
// load balances by itself and a node only ever touches a single device.
struct moe_device {
    int device = -1;
    // An integrated GPU has no memory of its own: its "VRAM" is carved out of the same system RAM
    // the CPU reads, so copying experts into it would only duplicate them. It can still compute,
    // reading the host weights in place, so it serves class 2 only.
    bool integrated = false;
    std::unique_ptr<ggml_backend_cuda_context> ctx; // our own streams, never the graph's

    ggml_backend_buffer_t pool_buf  = nullptr;
    // a few bytes of real device buffer, marked as weights. The mat-vec reads src0->buffer's usage
    // flag, so a descriptor over mapped host memory still needs one to point at.
    ggml_backend_buffer_t anchor_buf = nullptr;
    char *                pool_base = nullptr;
    size_t                pool_bytes = 0;
    size_t                budget = 0;
    size_t                used = 0;

    moe_scratch act;   // class 1: gathered activations, f32
    moe_scratch ids;   // class 1: pool slot per pair, i32
    moe_scratch out;   // class 1: dst rows, f32

    moe_scratch act2;  // class 2, in flight at the same time so it needs its own buffers
    moe_scratch ids2;
    moe_scratch out2;

    moe_timing tm1;    // class-1 launches
    moe_timing tm2;    // class-2 launches

    // completion markers, recorded after the D2H of each class's launch. join() spin-polls these
    // instead of cudaStreamSynchronize: the sync syscall costs 10-20 us per node, 72 nodes/token.
    cudaEvent_t done[2] = { nullptr, nullptr };

    // cross-device ordering for the contraction-shard reduction: recorded on this device's stream
    // after its partial has been peer-copied out, waited on by the receiving device's stream
    cudaEvent_t xfer = nullptr;

    cudaEvent_t migrate_done = nullptr;
    bool        migrating    = false;
};

// wait for a completion event by polling; returns immediately when the work already finished
static void moe_spin_wait(cudaEvent_t ev) {
    while (true) {
        const cudaError_t st = cudaEventQuery(ev);
        if (st == cudaSuccess) {
            return;
        }
        if (st != cudaErrorNotReady) {
            CUDA_CHECK(st);
        }
    }
}

// per-stage accumulators, one set per phase
struct moe_stage_acc {
    double  gather_us = 0;  // CPU wall: gathering activation rows into pinned staging
    double  h2d_us    = 0;  // GPU: activation + ids host-to-device
    double  kernel_us = 0;  // GPU: quantize + mat-vec or MMQ
    double  d2h_us    = 0;  // GPU: out device-to-host
    double  sync_us   = 0;  // CPU wall: time join() sat waiting on the stream
    int64_t nodes     = 0;
    int64_t pairs     = 0;
};

struct moe_hybrid_state {
    std::mutex mutex;

    std::vector<moe_device> devs;

    std::unordered_map<const ggml_tensor *, std::unique_ptr<moe_tensor_state>> tensors;
    std::vector<moe_tensor_state *> order;          // stable, for reporting

    size_t pool_bytes = 0;
    bool   pool_built = false;
    bool   p2p_ok    = false; // direct peer copies work between all pool devices (probed at init)

    int cur_dev = -1; // device serving class 1 for the node, between dispatch and join
    int c2_dev  = -1; // device serving class 2; the integrated GPU when there is one

    // Class 1 against the unified facade store runs once per shard device (each computes its own
    // disjoint dst rows), so the join must wait and scatter per device, not just cur_dev.
    bool c1_facade = false;
    int  c1_devs[GGML_CUDA_MAX_DEVICES] = { 0 };
    int  n_c1 = 0;

    // one plan() call per expert tensor per token, so this divided by the tensor count is tokens
    int64_t plan_calls        = 0;
    int64_t plan_calls_window = 0;

    // counters
    int64_t n_admit = 0;
    int64_t n_evict = 0;
    int64_t n_migrations = 0;
    int64_t gpu_pairs = 0;
    int64_t cpu_pairs = 0;

    // why nodes were turned away, so "it did nothing" is never a silent outcome
    int64_t n_reject_buffer = 0;
    int64_t n_reject_buft   = 0;
    int64_t n_reject_batch  = 0;

    int64_t migrate_bytes = 0; // H2D traffic spent refilling the pool

    // per-node work list, rebuilt by dispatch()
    std::vector<int32_t> pair_slot;   // class 1: pool slot per pair
    std::vector<int32_t> pair_i1;
    std::vector<int32_t> pair_i2;

    std::vector<int32_t> c2_expert;   // class 2: expert index per pair, weights read from host
    std::vector<int32_t> c2_i1;
    std::vector<int32_t> c2_i2;
    int64_t              c2_row0 = 0; // first output row the GPU takes for class 2

    int64_t stream_pairs = 0;         // counters
    int64_t host_registered_bytes = 0;

    // Under --split-mode tensor the CPU MUL_MAT_ID is MIRRORED and runs once per device, so the
    // same node can be launched more than once per token. Count launches per device and the time
    // the CPU sits in the join, which together say whether the GPU half is helping or blocking.
    int64_t n_launch = 0;
    int64_t join_wait_us = 0;
    int64_t cpu_work_us  = 0;
    int64_t dev_launch[GGML_CUDA_MAX_DEVICES] = { 0 };

    moe_stage_acc st_dec; // per-stage costs at decode-sized nodes
    moe_stage_acc st_bat; // and at batched (prefill) nodes

    // routing trace: (tensor ordinal, expert ids of one token) per single-token dispatch
    std::vector<std::pair<int32_t, std::array<int32_t, 16>>> trace;

    // adaptive class-2 share. launch() returning to join() being called is exactly the CPU's work
    // for the node; the sync inside join() is how much longer the GPU needed. Keeping the second
    // small means the GPU is not the straggler.
    double  frac_now   = -1.0; // < 0 until the first dispatch reads the operator's setting
    double  wait_share = 0.0;  // EWMA of gpu_extra / (cpu + gpu_extra)
    bool    last_was_decode = true; // frac_now describes decode, so only steer it on decode nodes
    int64_t t_launch_end = 0;
    int64_t adapt_calls  = 0;

    ~moe_hybrid_state();
};

static moe_hybrid_state & moe_state() {
    static moe_hybrid_state s;
    return s;
}

moe_hybrid_state::~moe_hybrid_state() {
    const int64_t rejects = n_reject_buffer + n_reject_buft + n_reject_batch;
    if (n_migrations == 0 && gpu_pairs == 0 && rejects == 0) {
        return;
    }
    // every expert visit falls in exactly one class, so these three shares sum to 100%
    const int64_t total = gpu_pairs + stream_pairs + cpu_pairs;
    fprintf(stderr, "[moe-hybrid] pool: %.2f GiB, %" PRId64 " migrations, %" PRId64 " admitted, "
            "%" PRId64 " evicted, %.2f GiB refilled\n",
            pool_bytes/1024.0/1024.0/1024.0, n_migrations, n_admit, n_evict,
            migrate_bytes/1024.0/1024.0/1024.0);
    if (total > 0) {
        fprintf(stderr, "[moe-hybrid] expert visits: %.1f%% from VRAM (class 1), %.1f%% shared with "
                "the CPU over PCIe (class 2), %.1f%% CPU only (class 3)\n",
                100.0*gpu_pairs/(double) total, 100.0*stream_pairs/(double) total,
                100.0*cpu_pairs/(double) total);
    }
    if (n_launch > 0) {
        fprintf(stderr, "[moe-hybrid] %" PRId64 " launches, CPU %.1f s in the node vs %.1f s waiting "
                "on the GPU (%.1f%% of node time is wait)\n", n_launch, cpu_work_us/1e6,
                join_wait_us/1e6, 100.0*join_wait_us/std::max<int64_t>(1, cpu_work_us + join_wait_us));
        auto stage_line = [](const char * name, const moe_stage_acc & a) {
            if (a.nodes == 0) {
                return;
            }
            const double n = (double) a.nodes;
            fprintf(stderr, "[moe-hybrid]   %s: %" PRId64 " nodes, %.1f pairs/node | per node: "
                    "gather %.1f us, h2d %.1f, kernel %.1f, d2h %.1f, join wait %.1f\n",
                    name, a.nodes, a.pairs/n, a.gather_us/n, a.h2d_us/n, a.kernel_us/n,
                    a.d2h_us/n, a.sync_us/n);
        };
        stage_line("decode ", st_dec);
        stage_line("batched", st_bat);
        fprintf(stderr, "[moe-hybrid] launches per device:");
        for (size_t d = 0; d < devs.size(); ++d) {
            fprintf(stderr, " dev%d=%" PRId64, devs[d].device, dev_launch[d]);
        }
        fprintf(stderr, "\n");
    }
    if (host_registered_bytes > 0) {
        fprintf(stderr, "[moe-hybrid] %.2f GiB of host weights registered for device reads\n",
                host_registered_bytes/1024.0/1024.0/1024.0);
    }
    if (!trace.empty() && moe_hybrid_trace_path() != nullptr) {
        FILE * f = fopen(moe_hybrid_trace_path(), "w");
        if (f != nullptr) {
            for (size_t i = 0; i < order.size(); ++i) {
                fprintf(f, "# %zu %s\n", i, order[i]->name.c_str());
            }
            for (const auto & e : trace) {
                fprintf(f, "%d", e.first);
                for (int32_t x : e.second) {
                    if (x >= 0) {
                        fprintf(f, "\t%d", x);
                    }
                }
                fputc('\n', f);
            }
            fclose(f);
            fprintf(stderr, "[moe-hybrid] wrote %zu trace records to %s\n", trace.size(), moe_hybrid_trace_path());
        }
    }
    if (rejects > 0) {
        fprintf(stderr, "[moe-hybrid] nodes declined: %" PRId64 " not host, %" PRId64 " wrong buffer type, "
                "%" PRId64 " batch too large\n", n_reject_buffer, n_reject_buft, n_reject_batch);
    }
}

// ---------------------------------------------------------------------------

// A pool slot is only usable once its copy has landed.
static void moe_check_migration(moe_hybrid_state & s) {
    for (size_t d = 0; d < s.devs.size(); ++d) {
        moe_device & dev = s.devs[d];
        if (!dev.migrating) {
            continue;
        }
        ggml_cuda_set_device(dev.device);
        if (cudaEventQuery(dev.migrate_done) != cudaSuccess) {
            continue;
        }
        for (moe_tensor_state * t : s.order) {
            if ((size_t) t->dev_idx != d) {
                continue;
            }
            for (moe_slot & sl : t->slots) {
                sl.ready = sl.expert >= 0;
            }
        }
        dev.migrating = false;
    }
}

// Which tensors the unified facade store serves as class 1. gate_up shards are row splits - each
// device computes its own disjoint dst rows, exactly the existing class-1 contract. down is a
// contraction split whose per-device results are partial sums that hop over P2P, so it only joins
// the store when direct peer copies exist; otherwise it keeps its pool.
static bool moe_facade_store(const moe_tensor_state * t) {
    if (!t->has_facade) {
        return false;
    }
    // Everything that is not `down` is a ROW split - fused gate_up and, in a split layout, the
    // separate gate and up. Each device computes disjoint dst rows, so there is nothing to
    // reduce and the store always serves them. Testing for the fused name alone denied the
    // store to split-layout gate/up (qwen35moe), which reach here and fall to the p2p test.
    if (t->name.find("ffn_down") == std::string::npos) {
        return true;
    }
    // `down` is contraction-split: per-device partial sums hop over P2P, so the store needs
    // direct peer copies. NOTE: extending this to a single pool device (where the tensor is not
    // split and no partial exists) was TRIED and produces finite garbage output - the class-1
    // store path at n_dev == 1 is not correct. Keep `down` on the pool until that is root-caused.
    return moe_state().p2p_ok;
}

// The facade store can serve class 1 only once the fill has run AND every shard device has
// learned its geometry, so the shards cover the whole computation. A partial device set would
// leave dst rows (row split) or input columns (contraction split) uncomputed - class 1 tells the
// CPU to skip the expert entirely, so coverage must be exact.
static bool moe_facade_ready(const moe_hybrid_state & s, const moe_tensor_state * t) {
    if (t->facade_slot_of.empty()) {
        return false;
    }
    int64_t rows = 0, cols = 0;
    for (size_t d = 0; d < s.devs.size(); ++d) {
        // anchor_buf carries the usage flag the stock kernels read through the weight descriptor
        if (t->facade_data[d] != nullptr && s.devs[d].anchor_buf != nullptr) {
            rows += t->facade_shard_rows[d];
            cols += t->facade_shard_cols[d];
            if (t->facade_is_col && t->facade_shard_rows[d] != t->ne01) {
                return false; // a contraction shard must carry every output row
            }
            if (!t->facade_is_col && t->facade_shard_cols[d] != t->ne00) {
                // a narrow shard whose slice happens to hold column 0 sees the markers and looks
                // row-split until its sibling learns; never launch off that half-knowledge
                return false;
            }
        }
    }
    return t->facade_is_col ? cols == t->ne00 : rows == t->ne01;
}

// Size the pool once, when every expert tensor has been seen at least once. Each tensor gets the
// same fraction of its experts: every layer sees every token, so demand per tensor is equal and
// only the spread over experts differs.
static void moe_build_pool(moe_hybrid_state & s) {
    if (s.pool_built || s.order.empty()) {
        return;
    }

    // Spread the tensors over the devices that actually have memory of their own. An integrated GPU
    // is skipped here: it gets no pool, and instead serves class 2 for every node.
    std::vector<int> pool_devs;
    for (size_t d = 0; d < s.devs.size(); ++d) {
        if (!s.devs[d].integrated) {
            pool_devs.push_back((int) d);
        }
    }
    if (pool_devs.empty()) {
        s.pool_built = true;
        return;
    }
    // Round-robin looks fair but is not: the tensors alternate gate_up, down, gate_up, ... and
    // gate_up is roughly twice the size, so every other slot handed one device all the big ones
    // (measured 1.85 GiB against 0.97 GiB for the same expert count). Assign each tensor to
    // whichever device is holding the least so far.
    std::vector<size_t> dev_bytes(s.devs.size(), 0);
    int n_fstore = 0;
    for (moe_tensor_state * t : s.order) {
        int best = pool_devs[0];
        for (int d : pool_devs) {
            if (dev_bytes[d] < dev_bytes[best]) {
                best = d;
            }
        }
        t->dev_idx = best; // facade-store tensors keep a home too: class 2 and the rating fold use it
        if (moe_facade_store(t)) {
            // no pool bytes, so the byte balance would park them all on one device; alternate the
            // home explicitly - it steers class-2 reads and the contraction reduction side
            t->dev_idx = pool_devs[n_fstore % (int) pool_devs.size()];
            n_fstore++;
            continue;
        }
        dev_bytes[best] += (size_t) t->n_expert * t->expert_bytes;
    }
    if (n_fstore > 0) {
        fprintf(stderr, "[moe-hybrid] %d tensor(s) served from the facade store, no pool slots for them\n", n_fstore);
    }

    // the integrated GPU, if any, is the one that streams host weights: it reads them in place
    for (size_t d = 0; d < s.devs.size(); ++d) {
        if (s.devs[d].integrated) {
            s.c2_dev = (int) d;
            break;
        }
    }

    const size_t forced  = moe_hybrid_budget_bytes();
    const size_t reserve = moe_hybrid_reserve_bytes();

    for (size_t d = 0; d < s.devs.size(); ++d) {
        moe_device & dev = s.devs[d];

        size_t total_expert_bytes = 0;
        for (const moe_tensor_state * t : s.order) {
            if ((size_t) t->dev_idx == d && !moe_facade_store(t)) {
                total_expert_bytes += (size_t) t->n_expert * t->expert_bytes;
            }
        }
        if (total_expert_bytes == 0) {
            continue;
        }

        size_t budget;
        if (forced == (size_t) -1) {
            size_t free_b = 0, total_b = 0;
            ggml_cuda_set_device(dev.device);
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
                free_b = 0;
            }
            budget = free_b > reserve ? free_b - reserve : 0;
            budget = std::min(budget, moe_hybrid_auto_cap());
        } else {
            budget = forced / s.devs.size();
        }
        budget = std::min(budget, total_expert_bytes);
        if (budget < 64u*1024*1024) {
            fprintf(stderr, "[moe-hybrid] device %d: only %.1f MiB free, not caching there\n",
                    dev.device, budget/1024.0/1024.0);
            continue;
        }

        // Slots per tensor. An explicit expert count applies to every tensor alike; otherwise the
        // budget is shared out proportionally, which comes to the same thing when the tensors are
        // the same size and keeps the share equal when they are not. Every layer sees every token,
        // so demand per tensor is equal and only the spread over experts differs.
        const int64_t want = moe_hybrid_n_experts();
        size_t need = 0;
        for (moe_tensor_state * t : s.order) {
            if ((size_t) t->dev_idx != d || moe_facade_store(t)) {
                continue;
            }
            int64_t n_slots;
            if (want >= 0) {
                n_slots = want;
            } else {
                const double share = (double) budget * ((double) t->n_expert * t->expert_bytes) / (double) total_expert_bytes;
                n_slots = (int64_t) (share / (double) t->expert_bytes);
            }
            n_slots = std::min<int64_t>(n_slots, t->n_expert);
            t->slots.assign(n_slots, moe_slot{});
            need += (size_t) n_slots * t->expert_bytes;
        }

        // An explicit count that does not fit is an operator error worth naming rather than an
        // allocation failure that silently leaves the device with no experts at all.
        if (want >= 0 && need > budget) {
            fprintf(stderr, "[moe-hybrid] device %d: %" PRId64 " experts per tensor needs %.2f GiB but only "
                    "%.2f GiB is free; reducing\n", dev.device, want, need/1024.0/1024.0/1024.0,
                    budget/1024.0/1024.0/1024.0);
            need = 0;
            for (moe_tensor_state * t : s.order) {
                if ((size_t) t->dev_idx != d || moe_facade_store(t)) {
                    continue;
                }
                const double share = (double) budget * ((double) t->n_expert * t->expert_bytes) / (double) total_expert_bytes;
                int64_t n_slots = std::min<int64_t>((int64_t) (share / (double) t->expert_bytes), t->n_expert);
                t->slots.assign(n_slots, moe_slot{});
                need += (size_t) n_slots * t->expert_bytes;
            }
        }
        if (need == 0) {
            continue;
        }

        ggml_cuda_set_device(dev.device);
        ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(dev.device);
        dev.pool_buf = ggml_backend_buft_alloc_buffer(buft, need);
        if (dev.pool_buf == nullptr) {
            fprintf(stderr, "[moe-hybrid] device %d: failed to allocate %.2f GiB for the expert pool\n",
                    dev.device, need/1024.0/1024.0/1024.0);
            for (moe_tensor_state * t : s.order) {
                if ((size_t) t->dev_idx == d) {
                    t->slots.clear();
                }
            }
            continue;
        }
        // weights, so the mat-vec skips its compute-buffer padding clear
        ggml_backend_buffer_set_usage(dev.pool_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        dev.pool_base  = (char *) ggml_backend_buffer_get_base(dev.pool_buf);
        dev.pool_bytes = need;
        s.pool_bytes  += need;

        size_t off = 0;
        for (moe_tensor_state * t : s.order) {
            if ((size_t) t->dev_idx != d || moe_facade_store(t)) {
                continue;
            }
            t->pool = dev.pool_base + off;
            off += t->slots.size() * t->expert_bytes;

            // the pool looks exactly like a weight tensor with one matrix per slot
            ggml_tensor & desc = t->pool_desc;
            desc = {};
            desc.type   = t->type;
            desc.buffer = dev.pool_buf;
            desc.data   = t->pool;
            desc.ne[0]  = t->ne00;
            desc.ne[1]  = t->ne01;
            desc.ne[2]  = (int64_t) t->slots.size();
            desc.ne[3]  = 1;
            desc.nb[0]  = ggml_type_size(t->type);
            desc.nb[1]  = ggml_row_size(t->type, t->ne00);
            desc.nb[2]  = t->expert_bytes;
            desc.nb[3]  = t->expert_bytes * t->slots.size();
        }

        int64_t slots_min = INT64_MAX, slots_max = 0, n_tot = 0;
        for (const moe_tensor_state * t : s.order) {
            if ((size_t) t->dev_idx == d && !moe_facade_store(t)) {
                slots_min = std::min<int64_t>(slots_min, t->slots.size());
                slots_max = std::max<int64_t>(slots_max, t->slots.size());
                n_tot     = t->n_expert;
            }
        }
        fprintf(stderr, "[moe-hybrid] device %d: %" PRId64 "-%" PRId64 " of %" PRId64 " experts per tensor "
                "resident, %.2f GiB\n", dev.device, slots_min, slots_max, n_tot, need/1024.0/1024.0/1024.0);
    }

    fprintf(stderr, "[moe-hybrid] expert pool %.2f GiB total over %zu tensors on %zu devices\n",
            s.pool_bytes/1024.0/1024.0/1024.0, s.order.size(), s.devs.size());
    s.pool_built = true;
}

// Rewrite the resident set from the current ratings. Nothing of ours is in flight here: the CPU
// kernel joins every node before it returns, so evicting a slot cannot race a running kernel.
static void moe_migrate(moe_hybrid_state & s) {
    if (!s.pool_built) {
        return;
    }

    const double keep = moe_hybrid_decay();

    int64_t admitted = 0;
    for (moe_tensor_state * t : s.order) {
        moe_device & dev = s.devs[t->dev_idx];
        if (dev.migrating) {
            continue; // its previous refill is still landing
        }
        ggml_cuda_set_device(dev.device);
        // its own stream: join() waits on the compute stream and must not be held up by a refill
        cudaStream_t stream = dev.ctx->stream(dev.device, 1);
        // fold this window into the decayed score
        for (int64_t a = 0; a < t->n_expert; ++a) {
            t->rating[a] = t->rating[a]*keep + (double) t->window[a];
            t->window[a] = 0;
        }
        if (t->slots.empty()) {
            continue;
        }

        // the experts we want resident: the highest rated, as many as there are slots
        std::vector<int32_t> want(t->n_expert);
        for (int64_t a = 0; a < t->n_expert; ++a) {
            want[a] = (int32_t) a;
        }
        const size_t n_slots = t->slots.size();
        // A resident expert defends its slot with a bonus, so it is only displaced by a challenger
        // that is clearly, not marginally, more in demand.
        const double hys = moe_hybrid_hysteresis();
        auto score = [&](int32_t a) {
            return t->slot_of[a] >= 0 ? t->rating[a]*hys : t->rating[a];
        };
        std::partial_sort(want.begin(), want.begin() + n_slots, want.end(),
                          [&](int32_t x, int32_t y) { return score(x) > score(y); });
        want.resize(n_slots);

        std::vector<bool> wanted(t->n_expert, false);
        for (int32_t a : want) {
            if (t->rating[a] > 0.0) {
                wanted[a] = true;
            }
        }

        // drop what fell out
        for (size_t sl = 0; sl < n_slots; ++sl) {
            const int32_t a = t->slots[sl].expert;
            if (a >= 0 && !wanted[a]) {
                t->slot_of[a] = -1;
                t->slots[sl]  = moe_slot{};
                s.n_evict++;
            }
        }

        // fill free slots with what came in
        size_t sl = 0;
        for (int32_t a : want) {
            if (!wanted[a] || t->slot_of[a] >= 0) {
                continue;
            }
            while (sl < n_slots && t->slots[sl].expert >= 0) {
                sl++;
            }
            if (sl >= n_slots) {
                break;
            }
            CUDA_CHECK(cudaMemcpyAsync(t->pool + sl*t->expert_bytes,
                                       (const char *) t->src0->data + (size_t) a*t->expert_bytes,
                                       t->expert_bytes, cudaMemcpyHostToDevice, stream));
            t->slots[sl].expert = a;
            t->slots[sl].ready  = false;
            t->slot_of[a]       = (int32_t) sl;
            s.n_admit++;
            s.migrate_bytes += t->expert_bytes;
            admitted++;
            sl++;
        }
    }

    if (admitted > 0) {
        for (moe_device & dev : s.devs) {
            if (dev.pool_buf == nullptr) {
                continue;
            }
            ggml_cuda_set_device(dev.device);
            CUDA_CHECK(cudaEventRecord(dev.migrate_done, dev.ctx->stream(dev.device, 1)));
            dev.migrating = true;
        }
    } else {
        // nothing moved, so everything already in a slot is usable
        for (moe_tensor_state * t : s.order) {
            for (moe_slot & slot : t->slots) {
                slot.ready = slot.expert >= 0;
            }
        }
    }

    s.n_migrations++;
    // residency changed: every tensor's device-side pointer table is stale until re-uploaded
    for (moe_tensor_state * t : s.order) {
        t->ptr_table_dirty = true;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            t->facade_dirty[d] = true;
        }
    }
    if (moe_hybrid_verbose()) {
        fprintf(stderr, "[moe-hybrid] migration %" PRId64 ", %" PRId64 " admitted\n",
                s.n_migrations, admitted);
    }
}

// Upload tensor t's expert pointer table to its home device: a resident expert points at its VRAM
// slot, a missing one at the mapped host address of its rows. This is the whole contract of the
// pointer-table decode path - a GPU-owned node dereferences the table per expert and never needs
// the CPU, and the table only changes at migration boundaries. Requires --moe-hybrid-dma (the
// host weights must be registered so host_dev pointers exist). Returns false when they do not.
static bool moe_ptr_table_upload(moe_hybrid_state & s, moe_tensor_state * t, void * host_dev_base, cudaStream_t stream) {
    if (!t->ptr_table_dirty && t->ptr_table_dev != nullptr) {
        return true;
    }
    if (host_dev_base == nullptr) {
        return false;
    }
    moe_device & dev = s.devs[t->dev_idx];
    ggml_cuda_set_device(dev.device);
    if (t->ptr_table_dev == nullptr) {
        CUDA_CHECK(cudaMalloc(&t->ptr_table_dev, t->n_expert * sizeof(void *)));
    }
    std::vector<void *> table(t->n_expert);
    for (int64_t a = 0; a < t->n_expert; ++a) {
        const int32_t sl = t->slot_of[a];
        if (sl >= 0 && t->slots[sl].ready) {
            table[a] = t->pool + (size_t) sl * t->expert_bytes;
        } else {
            table[a] = (char *) host_dev_base + (size_t) a * t->expert_bytes;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(t->ptr_table_dev, table.data(), t->n_expert * sizeof(void *),
                               cudaMemcpyHostToDevice, stream));
    t->ptr_table_dirty = false;
    return true;
}

// ---------------------------------------------------------------------------
// the API ggml-cpu calls
// ---------------------------------------------------------------------------

// The backend registry is built before main() parses arguments, so the operator's choice is only
// visible once compute starts. Everything device-side is therefore created on first use.
static bool moe_hybrid_ready() {
    static bool ok = [] {
        const char * s = getenv("GGML_MOE_HYBRID");
        if (s == nullptr || s[0] == '\0' || strcmp(s, "off") == 0 || strcmp(s, "0") == 0) {
            return false;
        }
        if (ggml_backend_cuda_get_device_count() == 0) {
            return false;
        }
        moe_hybrid_state & st = moe_state();
        const int n_dev = ggml_backend_cuda_get_device_count();
        st.devs.resize(n_dev);
        for (int i = 0; i < n_dev; ++i) {
            moe_device & dev = st.devs[i];
            dev.device = i;
            dev.ctx    = std::make_unique<ggml_backend_cuda_context>(i);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventCreateWithFlags(&dev.migrate_done, cudaEventDisableTiming));
            dev.anchor_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(i), 4096);
            if (dev.anchor_buf != nullptr) {
                ggml_backend_buffer_set_usage(dev.anchor_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                dev.integrated = prop.integrated != 0;
                fprintf(stderr, "[moe-hybrid] device %d: %s, %d CUs%s\n",
                        i, prop.name, prop.multiProcessorCount,
                        dev.integrated ? " (integrated: streams only, no pool)" : "");
            }
        }
        // The contraction-shard reduction (unified down store) hops partials between devices with
        // direct peer copies; probe and enable that once, here, so the pool-build decision later
        // can rely on the answer.
        st.p2p_ok = n_dev >= 2;
        for (int i = 0; i < n_dev && st.p2p_ok; ++i) {
            if (st.devs[i].integrated) {
                continue;
            }
            for (int j = 0; j < n_dev; ++j) {
                if (i == j || st.devs[j].integrated) {
                    continue;
                }
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || can == 0) {
                    st.p2p_ok = false;
                    break;
                }
                ggml_cuda_set_device(i);
                const cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    (void) cudaGetLastError();
                    st.p2p_ok = false;
                    break;
                }
                (void) cudaGetLastError();
            }
        }
        fprintf(stderr, "[moe-hybrid] residency cache enabled over %d device(s)%s\n", n_dev,
                st.p2p_ok ? ", peer copies direct" : "");
        return true;
    }();
    return ok;
}

static bool moe_hybrid_supports(const ggml_moe_hybrid_node * node) {
    if (!moe_hybrid_ready()) {
        return false;
    }

    const ggml_tensor * src0 = node->src0;

    if (!ggml_is_quantized(src0->type) || src0->ne[3] != 1) {
        return false;
    }
    if (node->src1->type != GGML_TYPE_F32 || node->dst->type != GGML_TYPE_F32) {
        return false;
    }
    // the weights must actually be in host RAM, otherwise this op would not be on the CPU at all
    if (src0->buffer == nullptr || !ggml_backend_buffer_is_host(src0->buffer)) {
        moe_state().n_reject_buffer++;
        return false;
    }
    // Allow only the two plain host layouts. "CPU_Mapped" is what mmap'd weights get; the extra
    // buffer types (CPU_REPACK, CPU_HBM, CPU_KLEIDIAI, ...) interleave the rows, so a copy of one
    // would be unreadable by the mat-vec.
    {
        const char * bt = ggml_backend_buft_name(src0->buffer->buft);
        if (strcmp(bt, "CPU") != 0 && strcmp(bt, "CPU_Mapped") != 0) {
            moe_state().n_reject_buft++;
            return false;
        }
    }
    // Bounds the activation/output scratch (pairs x ne00 x 4B twice, ~256 MiB each at the cap).
    // Large batches are welcome now that batched nodes run through MMQ rather than the mat-vec.
    if (node->ids->ne[1] * node->ids->ne[0] > 16384) {
        moe_state().n_reject_batch++;
        return false;
    }
    return true;
}

static moe_tensor_state * moe_tensor_for(moe_hybrid_state & s, const ggml_tensor * src0) {
    auto it = s.tensors.find(src0);
    if (it != s.tensors.end()) {
        return it->second.get();
    }
    if (s.pool_built) {
        return nullptr; // showed up after the pool was sized; leave it on the CPU
    }

    auto st = std::make_unique<moe_tensor_state>();
    st->src0         = src0;
    st->name         = src0->name; // snapshot: src0 may be freed before the census prints
    st->type         = src0->type;
    st->n_expert     = src0->ne[2];
    st->ne00         = src0->ne[0];
    st->ne01         = src0->ne[1];
    st->expert_bytes = src0->nb[2];
    st->rating.assign(st->n_expert, 0.0);
    st->window.assign(st->n_expert, 0);
    st->slot_of.assign(st->n_expert, -1);

    moe_tensor_state * raw = st.get();
    s.tensors.emplace(src0, std::move(st));
    s.order.push_back(raw);
    return raw;
}

// Class 2: let a kernel read the host weights directly over PCIe. Registering pins the pages, which
// was measured to cost ~11% of CPU-side decode when applied to a whole weight buffer, so it is
// opt-in and worth checking against the census before turning on.
static bool moe_register_host(moe_tensor_state * t, int dev_idx, int device) {
    if (t->host_tried) {
        return t->host_dev != nullptr && t->host_dev_idx == dev_idx;
    }
    t->host_tried = true;
    t->host_dev_idx = dev_idx;

    void * base = t->src0->data;
    const size_t bytes = ggml_nbytes(t->src0);

    ggml_cuda_set_device(device);
    cudaError_t err = cudaHostRegister(base, bytes,
            cudaHostRegisterPortable | cudaHostRegisterReadOnly | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        fprintf(stderr, "[moe-hybrid] cannot register %s for device reads: %s\n",
                t->src0->name, cudaGetErrorString(err));
        return false;
    }

    void * devp = nullptr;
    err = cudaHostGetDevicePointer(&devp, base, 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        cudaHostUnregister(base);
        fprintf(stderr, "[moe-hybrid] no device pointer for %s\n", t->src0->name);
        return false;
    }

    t->host_dev = (char *) devp;
    moe_state().host_registered_bytes += bytes;
    return true;
}

// Describe the host weights as [ne00, ne01-row0, n_expert] starting at row0. Per-expert stride is
// still nb02, so this is a plain strided view - no copy, no repack.
static void moe_setup_host_desc(moe_tensor_state * t, int64_t row0, ggml_backend_buffer_t borrow) {
    if (t->host_row0 == row0) {
        return;
    }
    ggml_tensor & d = t->host_desc;
    d = {};
    d.type   = t->type;
    d.buffer = borrow; // only read for its usage flag by the mat-vec
    d.data   = t->host_dev + row0 * ggml_row_size(t->type, t->ne00);
    d.ne[0]  = t->ne00;
    d.ne[1]  = t->ne01 - row0;
    d.ne[2]  = t->n_expert;
    d.ne[3]  = 1;
    d.nb[0]  = ggml_type_size(t->type);
    d.nb[1]  = ggml_row_size(t->type, t->ne00);
    d.nb[2]  = t->expert_bytes;
    d.nb[3]  = t->expert_bytes * t->n_expert;
    t->host_row0 = row0;
}

static bool moe_hybrid_dispatch(const ggml_moe_hybrid_node * node, int64_t * cpu_row_end) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);

    moe_tensor_state * t = moe_tensor_for(s, node->src0);
    if (t == nullptr) {
        return false;
    }

    const int64_t n_expert = t->n_expert;

    // score demand every node; it is one add per expert and never touches the device
    for (int64_t a = 0; a < n_expert; ++a) {
        t->window[a] += node->row_counts[a];
    }

    // routing trace for the predictor analysis: single-token nodes only, capped at ~4 MB
    if (moe_hybrid_trace_path() != nullptr && node->ids->ne[1] == 1 && s.trace.size() < (1u << 18)) {
        std::array<int32_t, 16> ex;
        ex.fill(-1);
        size_t k = 0;
        for (int64_t a = 0; a < n_expert && k < ex.size(); ++a) {
            if (node->row_counts[a] > 0) {
                ex[k++] = (int32_t) a;
            }
        }
        int32_t ord = 0;
        for (size_t i = 0; i < s.order.size(); ++i) {
            if (s.order[i] == t) {
                ord = (int32_t) i;
                break;
            }
        }
        s.trace.emplace_back(ord, ex);
    }

    s.plan_calls++;
    s.plan_calls_window++;

    moe_check_migration(s);

    // one plan call per tensor per token, so this is the token count for the window
    const int64_t tokens_in_window = s.order.empty() ? 0 : s.plan_calls_window / (int64_t) s.order.size();
    if (!s.pool_built && s.plan_calls >= (int64_t) s.order.size() * 8) {
        moe_build_pool(s);
        // fill it straight away from the demand seen so far, instead of idling for a whole window
        s.plan_calls_window = 0;
        moe_migrate(s);
    } else if (s.pool_built && tokens_in_window >= moe_hybrid_migrate_tokens()) {
        s.plan_calls_window = 0;
        moe_migrate(s);
    }

    // class 2 needs the host weights readable by a kernel, and a shared row boundary for the node
    if (s.frac_now < 0.0) {
        s.frac_now = moe_hybrid_stream_frac();
    }
    const bool   is_decode = node->ids->ne[1] <= moe_hybrid_stream_batch();
    const double frac      = is_decode ? s.frac_now : moe_hybrid_stream_prefill_frac();
    s.last_was_decode      = is_decode;
    int64_t row0 = t->ne01;

    // prefill-only mode: scoring above still ran (it feeds placement and promotion), but the CPU
    // keeps every row of a decode node - no launch, no join, stock-kernel cost
    if (is_decode && !moe_hybrid_decode_on()) {
        s.pair_slot.clear();
        s.c2_expert.clear();
        for (int64_t a = 0; a < n_expert; ++a) {
            s.cpu_pairs += node->row_counts[a];
        }
        return false;
    }

    // Registering is worth doing on its own: pinned pages transfer to the GPU at roughly PCIe speed
    // instead of the ~16-18 GB/s the driver gets from pageable memory, which is what the scheduler
    // pays for every ubatch of prefill. So do it whenever the operator asked, not only when class 2
    // is going to read from it.
    // Wait until the pool exists before registering: only then is the class-2 device settled, and
    // a range can only be registered once, so binding it to the wrong device is permanent.
    const int c2_idx = s.c2_dev >= 0 ? s.c2_dev : t->dev_idx;
    const bool registered = s.pool_built && moe_hybrid_dma() &&
                            moe_register_host(t, c2_idx, s.devs[c2_idx].device);

    // the host descriptor borrows the pool buffer for its usage flag, so it needs one to exist
    if (frac > 0.0 && registered && s.devs[c2_idx].anchor_buf != nullptr) {
        row0 = t->ne01 - (int64_t) (t->ne01 * frac);
        if (row0 >= t->ne01) {
            row0 = t->ne01;
        }
    }
    s.c2_row0 = row0;

    // Unified store: this tensor's residents live in the facade shards, so class 1 reads those
    // (one launch per shard device). Until the fill latch + geometry learn make it ready, its
    // pairs fall to class 2/3 - never to the pool, which was not built for it.
    const bool fstore = moe_facade_store(t) && moe_facade_ready(s, t);
    s.c1_facade = fstore;

    if (t->slots.empty() && !fstore && row0 >= t->ne01) {
        return false;
    }

    // class 1: experts held in VRAM, taken whole. class 2: the rest, shared by output row.
    s.pair_slot.clear();
    s.pair_i1.clear();
    s.pair_i2.clear();
    s.c2_expert.clear();
    s.c2_i1.clear();
    s.c2_i2.clear();

    for (int64_t a = 0; a < n_expert; ++a) {
        const int64_t cnt = node->row_counts[a];
        if (cnt == 0) {
            continue;
        }
        int32_t sl = -1;
        if (fstore) {
            sl = a < (int64_t) t->facade_slot_of.size() ? t->facade_slot_of[a] : -1;
        } else if (!t->slots.empty()) {
            const int32_t ps = t->slot_of[a];
            sl = (ps >= 0 && t->slots[ps].ready) ? ps : -1;
        }
        if (sl >= 0) {
            for (int64_t k = 0; k < cnt; ++k) {
                const mmid_row_mapping rm = node->matrix_rows[a*node->rows_stride + k];
                s.pair_slot.push_back(sl);
                s.pair_i1.push_back(rm.i1);
                s.pair_i2.push_back(rm.i2);
            }
            s.gpu_pairs += cnt;
            cpu_row_end[a] = 0; // the GPU takes this expert whole
            continue;
        }

        if (row0 < t->ne01) {
            for (int64_t k = 0; k < cnt; ++k) {
                const mmid_row_mapping rm = node->matrix_rows[a*node->rows_stride + k];
                s.c2_expert.push_back((int32_t) a);
                s.c2_i1.push_back(rm.i1);
                s.c2_i2.push_back(rm.i2);
            }
            s.stream_pairs += cnt;
            cpu_row_end[a] = row0; // shared: CPU keeps [0,row0), the GPU streams the rest
            continue;
        }

        s.cpu_pairs += cnt;
    }

    return !s.pair_slot.empty() || !s.c2_expert.empty();
}

// One batched mat-vec over a list of (expert, slot, token) triples. The weights come either from
// the VRAM pool or straight from mapped host memory; both are plain [ne00, n_rows, n_mat] strided
// views, so the stock MUL_MAT_ID mat-vec serves them unchanged.
// Gather one activation row per pair, on the device. At prefill the CPU-side gather was measured
// at 4.5 ms per node (1390 duplicated rows, single-threaded memcpy) while the whole MMQ kernel
// took 3.7 ms; uploading the layer's rows once and gathering here moves that copy from ~5 GB/s
// host memcpy to VRAM bandwidth. row_idx holds, per pair, the source row in the uploaded block.
static __global__ void moe_gather_rows(
        const float * __restrict__ src, const int32_t * __restrict__ row_idx,
        float * __restrict__ dst, const int64_t ne00, const int64_t src_stride, const int64_t col0) {
    const int64_t j   = blockIdx.x;
    const float * s   = src + (int64_t) row_idx[j]*src_stride + col0;
    float       * d   = dst + j*ne00;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) {
        d[i] = s[i];
    }
}

// the contraction-shard reduction: sum device B's partial (already peer-copied into staging) into
// device A's result before the single D2H
static __global__ void moe_add_partials(
        float * __restrict__ a, const float * __restrict__ b, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

static void moe_run_batch(
        moe_device & dev, moe_scratch & act, moe_scratch & ids, moe_scratch & out,
        const ggml_tensor * w_desc, const ggml_tensor * src1,
        const std::vector<int32_t> & mat, const std::vector<int32_t> & i1,
        const std::vector<int32_t> & i2, int64_t n_rows, bool batched, moe_timing * tm,
        int done_slot, moe_tensor_state * gt,
        // contraction-shard support: consume activation columns [col0, col0+ne00) of rows that are
        // src_row_elems wide (0 = the full row is the weight's width); leave_on_dev skips the D2H
        // and the completion event so the caller can chain a reduction on the results in out.dev
        int64_t src_row_elems = 0, int64_t col0 = 0, bool leave_on_dev = false) {
    const int64_t n_pairs = (int64_t) mat.size();
    if (n_pairs == 0) {
        return;
    }

    const int64_t ne00 = w_desc->ne[0];
    const int64_t src_row = src_row_elems > 0 ? src_row_elems : ne00;
    cudaStream_t stream = dev.ctx->stream();

    const int64_t ne11 = src1->ne[1];
    const int64_t rows_total = ne11 * src1->ne[2];
    // GPU-side gather wants the source rows as one dense block
    const bool src1_dense = src1->nb[0] == sizeof(float) &&
                            src1->nb[1] == (size_t) src_row*sizeof(float) &&
                            src1->nb[2] == (size_t) src_row*sizeof(float)*ne11;
    const bool gpu_gather = batched && src1_dense;

    act.reserve((n_pairs*ne00 + (gpu_gather ? rows_total*src_row : 0)) * sizeof(float));
    // decode packs the activation rows behind the indices for a single H2D; the rows start at a
    // 256-byte aligned offset so the mat-vec's vector loads stay happy
    const size_t pack_off = GGML_PAD(2 * n_pairs * sizeof(int32_t), 256);
    ids.reserve(pack_off + (gpu_gather ? 0 : n_pairs * ne00 * sizeof(float)));
    out.reserve(n_pairs * n_rows * sizeof(float));

    const bool timed = tm != nullptr && moe_hybrid_stats_on();
    const int64_t t_gather0 = timed ? ggml_time_us() : 0;

    int32_t * ids_host = (int32_t *) ids.host;
    memcpy(ids_host, mat.data(), n_pairs*sizeof(int32_t));

    if (gpu_gather) {
        // row indices only; the rows themselves go up in one dense copy and are gathered on device
        for (int64_t j = 0; j < n_pairs; ++j) {
            ids_host[n_pairs + j] = (int32_t) (i2[j]*ne11 + (i1[j] % ne11));
        }
    } else {
        // decode-sized: a handful of rows, the CPU copy is ~1 us. The rows go into the ids block
        // right after the indices so ONE H2D copy carries everything - each ROCm API call costs
        // 15-25 us and the per-node sequence is the decode bottleneck.
        float * act_host = (float *) ((char *) ids.host + pack_off);
        for (int64_t j = 0; j < n_pairs; ++j) {
            const int64_t i11 = i1[j] % ne11;
            const int64_t i12 = i2[j];
            const float * row = (const float *) ((const char *) src1->data + i11*src1->nb[1] + i12*src1->nb[2]);
            memcpy(act_host + j*ne00, row + col0, ne00*sizeof(float));
        }
    }

    moe_graph_entry * ge = nullptr;
    bool capturing = false;
    const bool graphable = gt != nullptr && !batched && !gpu_gather && !timed && !leave_on_dev && moe_hybrid_graphs_on();

    if (timed) {
        moe_stage_acc & a = batched ? moe_state().st_bat : moe_state().st_dec;
        a.gather_us += (double) (ggml_time_us() - t_gather0);
        a.nodes++;
        a.pairs += n_pairs;
        tm->ensure();
        CUDA_CHECK(cudaEventRecord(tm->ev[0], stream));
    }

    float * act_dev = (float *) act.dev;
    if (gpu_gather) {
        CUDA_CHECK(cudaMemcpyAsync(ids.dev, ids.host, 2*n_pairs*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        float * upload = (float *) act.dev + n_pairs*ne00;
        CUDA_CHECK(cudaMemcpyAsync(upload, src1->data, rows_total*src_row*sizeof(float), cudaMemcpyHostToDevice, stream));
        moe_gather_rows<<<(unsigned) n_pairs, 256, 0, stream>>>(
            upload, (const int32_t *) ids.dev + n_pairs, (float *) act.dev, ne00, src_row, col0);
        CUDA_CHECK(cudaGetLastError());
        moe_debug_checkpoint("run-gather", "-", dev.device, n_pairs, rows_total, batched);
    } else {
        // one call for indices and activation rows together
        CUDA_CHECK(cudaMemcpyAsync(ids.dev, ids.host,
            pack_off + n_pairs*ne00*sizeof(float), cudaMemcpyHostToDevice, stream));
        act_dev = (float *) ((char *) ids.dev + pack_off);
    }

    // The H2D above stays OUTSIDE the graph: a captured copy node must not be trusted to re-read
    // host content on replay (measured: replays diverged at the first graphed token). The graph
    // holds only the kernels and the completion event - still one launch instead of three.
    if (graphable) {
        ge = &gt->dec_graphs[n_pairs];
        const size_t gen = ids.gen + act.gen + out.gen;
        if (ge->exec != nullptr && ge->gen != gen) {
            cudaGraphExecDestroy(ge->exec);
            ge->exec = nullptr;
            ge->seen = 0;
        }
        if (ge->exec != nullptr) {
            // the graph holds only the kernels; the completion event is recorded live below,
            // because a replayed event-record node leaves the host-visible event state stale
            // until the GPU reaches it - the join would spin on the PREVIOUS node's signal
            CUDA_CHECK(cudaGraphLaunch(ge->exec, stream));
        }
        if (++ge->seen >= 3) {
            capturing = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            if (capturing) {
                ge->gen = gen;
            }
        }
    }

    if (timed) {
        CUDA_CHECK(cudaEventRecord(tm->ev[1], stream));
    }

    // synthetic shape: one column of dst, one channel per pair
    if (ge == nullptr || ge->exec == nullptr) {
    ggml_tensor src1_d {};
    src1_d.type  = GGML_TYPE_F32;
    src1_d.data  = act_dev;
    src1_d.ne[0] = ne00;   src1_d.ne[1] = n_pairs; src1_d.ne[2] = 1; src1_d.ne[3] = 1;
    src1_d.nb[0] = sizeof(float);
    src1_d.nb[1] = ne00*sizeof(float);
    src1_d.nb[2] = ne00*sizeof(float)*n_pairs;
    src1_d.nb[3] = src1_d.nb[2];

    ggml_tensor ids_d {};
    ids_d.type  = GGML_TYPE_I32;
    ids_d.data  = ids.dev;
    ids_d.ne[0] = n_pairs; ids_d.ne[1] = 1; ids_d.ne[2] = 1; ids_d.ne[3] = 1;
    ids_d.nb[0] = sizeof(int32_t);
    ids_d.nb[1] = n_pairs*sizeof(int32_t);
    ids_d.nb[2] = ids_d.nb[1];
    ids_d.nb[3] = ids_d.nb[1];

    // decode writes results directly into the mapped pinned block: tens of KB over PCIe, and the
    // separate device-to-host copy call disappears from the per-node sequence. A reduction caller
    // needs the results ON the device, so leave_on_dev forces the real out buffer.
    const bool direct_out = !batched && out.host_dev != nullptr && !leave_on_dev;

    ggml_tensor dst_d {};
    dst_d.type  = GGML_TYPE_F32;
    dst_d.data  = direct_out ? out.host_dev : out.dev;
    dst_d.ne[0] = n_rows;  dst_d.ne[1] = n_pairs; dst_d.ne[2] = 1; dst_d.ne[3] = 1;
    dst_d.nb[0] = sizeof(float);
    dst_d.nb[1] = n_rows*sizeof(float);
    dst_d.nb[2] = n_rows*sizeof(float)*n_pairs;
    dst_d.nb[3] = dst_d.nb[2];
    dst_d.src[0] = const_cast<ggml_tensor *>(w_desc);
    dst_d.src[1] = &src1_d;
    dst_d.src[2] = &ids_d;

    // Prefill and decode want different kernels, not just different shares. The mat-vec walks one
    // dst column per workgroup and re-reads the weights per token, which is right at decode and
    // disastrous at prefill (measured: fully dynamic prefill 167.8 vs 269.7 baseline, CPU waiting
    // on the GPU 45.4% of node time). MMQ buckets the pairs by matrix and runs a real tiled matmul
    // per expert, so batched nodes go there. Its ids contract is one expert per token row: ids
    // [1, n_pairs], src1 [ne00, 1, n_pairs], dst [n_rows, 1, n_pairs] - the same memory, reshaped.
    const int cc = ggml_cuda_info().devices[dev.device].cc;
    if (batched && ggml_cuda_should_use_mmq(w_desc->type, cc, n_pairs, w_desc->ne[2])) {
        src1_d.ne[1] = 1;       src1_d.ne[2] = n_pairs;
        src1_d.nb[2] = src1_d.nb[1];
        ids_d.ne[0]  = 1;       ids_d.ne[1]  = n_pairs;
        ids_d.nb[1]  = sizeof(int32_t);
        ids_d.nb[2]  = n_pairs*sizeof(int32_t);
        dst_d.ne[1]  = 1;       dst_d.ne[2]  = n_pairs;
        dst_d.nb[2]  = dst_d.nb[1];

        ggml_cuda_mul_mat_q(*dev.ctx, w_desc, &src1_d, &ids_d, &dst_d);
        moe_debug_checkpoint("run-mmq", "-", dev.device, n_pairs, n_rows, batched);
    } else {
        ggml_cuda_mul_mat_vec_q(*dev.ctx, w_desc, &src1_d, &ids_d, &dst_d);
        moe_debug_checkpoint("run-matvec", "-", dev.device, n_pairs, n_rows, batched);
    }

    if (timed) {
        CUDA_CHECK(cudaEventRecord(tm->ev[2], stream));
    }

    if (!direct_out && !leave_on_dev) {
        CUDA_CHECK(cudaMemcpyAsync(out.host, out.dev, n_pairs*n_rows*sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    if (timed) {
        CUDA_CHECK(cudaEventRecord(tm->ev[3], stream));
        tm->valid = true;
    }

    if (capturing) {
        cudaGraph_t g = nullptr;
        if (cudaStreamEndCapture(stream, &g) == cudaSuccess && g != nullptr) {
            if (cudaGraphInstantiate(&ge->exec, g, nullptr, nullptr, 0) != cudaSuccess) {
                ge->exec = nullptr; // capture failed; keep running uncaptured
            }
            cudaGraphDestroy(g);
        }
    }
    } // !replayed

    if (leave_on_dev) {
        return; // the caller chains the reduction, D2H and completion on this stream
    }
    if (dev.done[done_slot] == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&dev.done[done_slot], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(dev.done[done_slot], stream));
}

// Issue the work. Called after the barrier so the other threads are already computing.
static void moe_hybrid_launch(const ggml_moe_hybrid_node * node) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);

    moe_tensor_state * t = moe_tensor_for(s, node->src0);
    if (t == nullptr || (s.pair_slot.empty() && s.c2_expert.empty())) {
        return;
    }

    const bool batched = !s.last_was_decode;

    s.n_c1 = 0;
    if (s.c1_facade && t->facade_is_col && !s.pair_slot.empty()) {
        // Contraction shards (down): each device computes PARTIAL sums of every dst row over its
        // column slice of the weights and the matching activation columns. Device B's partial
        // hops to device A over P2P, a small add on A joins them, then one D2H serves the
        // scatter. Everything is stream-ordered: the peer copy follows B's kernel on B's stream,
        // A's add waits on B's transfer event, so the join only needs A's completion.
        int devA = -1, devB = -1;
        for (size_t d = 0; d < s.devs.size(); ++d) {
            if (t->facade_data[d] != nullptr && s.devs[d].anchor_buf != nullptr) {
                (devA < 0 ? devA : devB) = (int) d;
            }
        }
        GGML_ASSERT(devA >= 0 && devB >= 0); // readiness guaranteed a full column cover
        if (t->dev_idx == devB) {
            std::swap(devA, devB); // reduce on the tensor's home so the add+D2H alternates per block
        }
        moe_device & A = s.devs[devA];
        moe_device & B = s.devs[devB];

        const int64_t n_rows    = t->ne01;
        const size_t  out_bytes = s.pair_slot.size() * n_rows * sizeof(float);
        ggml_cuda_set_device(A.device);
        A.out.reserve(2*out_bytes); // second half stages B's incoming partial

        for (int pass = 0; pass < 2; ++pass) {
            const int d = pass == 0 ? devA : devB;
            moe_device & fdev = s.devs[d];
            ggml_cuda_set_device(fdev.device);

            const int64_t scols      = t->facade_shard_cols[d];
            const size_t  srow_bytes = ggml_row_size(t->type, scols);

            ggml_tensor w {};
            w.type   = t->type;
            w.buffer = fdev.anchor_buf; // read only for its usage flag
            w.data   = t->facade_data[d];
            w.ne[0]  = scols; w.ne[1] = n_rows; w.ne[2] = t->facade_n_slots; w.ne[3] = 1;
            w.nb[0]  = ggml_type_size(t->type);
            w.nb[1]  = srow_bytes;
            w.nb[2]  = srow_bytes * n_rows;
            w.nb[3]  = w.nb[2] * t->facade_n_slots;

            // the activation slice offset in elements, from the byte offset learned off the
            // markers: exact integer arithmetic, both are multiples of the quant block
            const int64_t col0 = t->ne00 * t->facade_col_off[d] / (int64_t) ggml_row_size(t->type, t->ne00);
            moe_run_batch(fdev, fdev.act, fdev.ids, fdev.out, &w, node->src1,
                          s.pair_slot, s.pair_i1, s.pair_i2, n_rows, batched,
                          nullptr, 0, nullptr, t->ne00, col0, /*leave_on_dev=*/true);
            moe_debug_checkpoint(pass == 0 ? "down-partial-A" : "down-partial-B",
                                 t->name.c_str(), fdev.device, (int64_t) s.pair_slot.size(), n_rows, batched);
        }

        char * staging = (char *) A.out.dev + out_bytes;
        ggml_cuda_set_device(B.device);
        if (B.xfer == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&B.xfer, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaMemcpyPeerAsync(staging, A.device, B.out.dev, B.device, out_bytes, B.ctx->stream()));
        CUDA_CHECK(cudaEventRecord(B.xfer, B.ctx->stream()));

        ggml_cuda_set_device(A.device);
        cudaStream_t sa = A.ctx->stream();
        CUDA_CHECK(cudaStreamWaitEvent(sa, B.xfer, 0));
        const int64_t n_elems = (int64_t) s.pair_slot.size() * n_rows;
        moe_add_partials<<<(unsigned) ((n_elems + 255)/256), 256, 0, sa>>>(
            (float *) A.out.dev, (const float *) staging, n_elems);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(A.out.host, A.out.dev, out_bytes, cudaMemcpyDeviceToHost, sa));
        if (A.done[0] == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&A.done[0], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(A.done[0], sa));
        moe_debug_checkpoint("down-reduce", t->name.c_str(), A.device,
                             (int64_t) s.pair_slot.size(), n_rows, batched);

        s.c1_devs[s.n_c1++] = devA; // the scatter reads A's summed rows only
        s.dev_launch[devA]++;
        s.dev_launch[devB]++;
        s.cur_dev = devA;
    } else if (s.c1_facade) {
        // class 1 out of the unified facade store: one launch per shard device, each computing
        // its own disjoint slice of every pair's dst rows. The weight descriptor is a stack view
        // of the shard - the stock kernels capture its fields at enqueue, nothing keeps it.
        for (size_t d = 0; !s.pair_slot.empty() && d < s.devs.size(); ++d) {
            if (t->facade_data[d] == nullptr || s.devs[d].anchor_buf == nullptr) {
                continue;
            }
            moe_device & fdev = s.devs[d];
            ggml_cuda_set_device(fdev.device);

            const int64_t srows      = t->facade_shard_rows[d];
            const int64_t scols      = t->facade_shard_cols[d];
            const size_t  srow_bytes = ggml_row_size(t->type, scols);

            ggml_tensor w {};
            w.type   = t->type;
            w.buffer = fdev.anchor_buf; // read only for its usage flag
            w.data   = t->facade_data[d];
            w.ne[0]  = scols; w.ne[1] = srows; w.ne[2] = t->facade_n_slots; w.ne[3] = 1;
            w.nb[0]  = ggml_type_size(t->type);
            w.nb[1]  = srow_bytes;
            w.nb[2]  = srow_bytes * srows;
            w.nb[3]  = w.nb[2] * t->facade_n_slots;

            // no graph capture here: the per-(tensor, n_pairs) cache key cannot tell the two
            // devices' captures apart
            moe_run_batch(fdev, fdev.act, fdev.ids, fdev.out, &w, node->src1,
                          s.pair_slot, s.pair_i1, s.pair_i2, srows, batched,
                          s.n_c1 == 0 ? &fdev.tm1 : nullptr, 0, nullptr);
            moe_debug_checkpoint("gateup-shard", t->name.c_str(), fdev.device,
                                 (int64_t) s.pair_slot.size(), srows, batched);
            s.c1_devs[s.n_c1++] = (int) d;
            s.dev_launch[d]++;
        }
        s.cur_dev = s.n_c1 > 0 ? s.c1_devs[0] : t->dev_idx;
    } else {
        // class 1: whole experts out of the VRAM pool
        moe_device & dev = s.devs[t->dev_idx];
        s.cur_dev = t->dev_idx;
        ggml_cuda_set_device(dev.device);
        moe_run_batch(dev, dev.act, dev.ids, dev.out, &t->pool_desc, node->src1,
                      s.pair_slot, s.pair_i1, s.pair_i2, t->ne01, batched, &dev.tm1, 0, t);
        moe_debug_checkpoint("pool", t->name.c_str(), dev.device,
                             (int64_t) s.pair_slot.size(), t->ne01, batched);
        s.c1_devs[s.n_c1++] = t->dev_idx;
        s.dev_launch[t->dev_idx]++;
    }

    // class 2: the top rows of the rest, read over PCIe while the CPU does rows [0, row0)
    if (!s.c2_expert.empty()) {
        moe_device & c2 = s.devs[s.c2_dev >= 0 ? s.c2_dev : t->dev_idx];
        ggml_cuda_set_device(c2.device);
        moe_setup_host_desc(t, s.c2_row0, c2.anchor_buf);
        moe_run_batch(c2, c2.act2, c2.ids2, c2.out2, &t->host_desc, node->src1,
                      s.c2_expert, s.c2_i1, s.c2_i2, t->ne01 - s.c2_row0, batched, &c2.tm2, 1, nullptr);
        moe_debug_checkpoint("class2", t->name.c_str(), c2.device,
                             (int64_t) s.c2_expert.size(), t->ne01 - s.c2_row0, batched);
    }

    s.n_launch++;
    s.t_launch_end = ggml_time_us();
}

static void moe_hybrid_join(const ggml_moe_hybrid_node * node) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);

    if (s.pair_slot.empty() && s.c2_expert.empty()) {
        return;
    }

    moe_device & dev = s.devs[s.cur_dev];
    ggml_cuda_set_device(dev.device);

    const int64_t t_join = ggml_time_us();
    if (!s.pair_slot.empty()) {
        for (int k = 0; k < s.n_c1; ++k) {
            moe_device & kdev = s.devs[s.c1_devs[k]];
            if (kdev.done[0] != nullptr) {
                ggml_cuda_set_device(kdev.device);
                moe_spin_wait(kdev.done[0]);
            }
        }
    }
    const int64_t t_done = ggml_time_us();

    if (s.t_launch_end > 0) {
        s.cpu_work_us  += t_join - s.t_launch_end;
        s.join_wait_us += t_done - t_join;
    }

    // read back the per-stage events now that the stream is drained; the join wait itself is
    // attributed once per node, on the class-1 harvest
    auto harvest = [&](moe_timing & tm, bool count_sync) {
        if (!tm.valid) {
            return;
        }
        float h2d = 0, krn = 0, d2h = 0;
        CUDA_CHECK(cudaEventElapsedTime(&h2d, tm.ev[0], tm.ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&krn, tm.ev[1], tm.ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, tm.ev[2], tm.ev[3]));
        moe_stage_acc & a = s.last_was_decode ? s.st_dec : s.st_bat;
        a.h2d_us    += 1e3*h2d;
        a.kernel_us += 1e3*krn;
        a.d2h_us    += 1e3*d2h;
        if (count_sync) {
            a.sync_us += (double) (t_done - t_join);
        }
        tm.valid = false;
    };
    if (moe_hybrid_stats_on()) {
        harvest(dev.tm1, true);
    }

    // t_cpu is how long the CPU threads worked on this node, gpu_extra how much longer the GPU
    // needed after them. Steer the class-2 share so the GPU stops just before the CPU does.
    if (moe_hybrid_stream_auto() && s.last_was_decode && s.t_launch_end > 0) {
        const double t_cpu     = (double) (t_join - s.t_launch_end);
        const double gpu_extra = (double) (t_done - t_join);
        const double share     = gpu_extra / std::max(1.0, t_cpu + gpu_extra);
        s.wait_share = 0.98*s.wait_share + 0.02*share;

        if (++s.adapt_calls % 512 == 0) {
            // 5% of the node spent waiting is close enough to balanced
            if (s.wait_share > 0.08 && s.frac_now > 0.01) {
                s.frac_now -= 0.01;
            } else if (s.wait_share < 0.03 && s.frac_now < 0.45) {
                s.frac_now += 0.01;
            }
            if (moe_hybrid_verbose()) {
                fprintf(stderr, "[moe-hybrid] stream share -> %.2f (gpu wait %.1f%% of node)\n",
                        s.frac_now, 100.0*s.wait_share);
            }
        }
    }

    // scatter each pair's rows into dst; a dst row is contiguous, so this is one memcpy per pair
    ggml_tensor * dst = node->dst;
    const int64_t ne01 = node->src0->ne[1];

    if (s.c1_facade) {
        // each shard device produced its pairs' rows in SHARD row order; translate to origin rows
        // through the learned segment runs (a couple of memcpys per pair)
        moe_tensor_state * t = moe_tensor_for(s, node->src0);
        for (int k = 0; t != nullptr && k < s.n_c1; ++k) {
            const int d = s.c1_devs[k];
            const int64_t srows = t->facade_shard_rows[d];
            const float * outk  = (const float *) s.devs[d].out.host;
            const auto  & segs  = t->facade_segs_host[d];
            for (size_t j = 0; j < s.pair_slot.size(); ++j) {
                float * dst_row = (float *) ((char *) dst->data + s.pair_i1[j]*dst->nb[1] + s.pair_i2[j]*dst->nb[2]);
                const float * src_row = outk + (int64_t) j*srows;
                for (const moe_facade_seg & g : segs) {
                    memcpy(dst_row + g.origin_r, src_row + g.shard_r, g.len*sizeof(float));
                }
            }
        }
    } else {
        const float * out1 = (const float *) dev.out.host;
        for (size_t j = 0; j < s.pair_slot.size(); ++j) {
            float * dst_row = (float *) ((char *) dst->data + s.pair_i1[j]*dst->nb[1] + s.pair_i2[j]*dst->nb[2]);
            memcpy(dst_row, out1 + j*ne01, ne01*sizeof(float));
        }
    }

    if (!s.c2_expert.empty()) {
        moe_device & c2 = s.devs[s.c2_dev >= 0 ? s.c2_dev : s.cur_dev];
        if (c2.done[1] != nullptr) {
            // unconditional: the class-1 wait loop may have left any device current
            ggml_cuda_set_device(c2.device);
            moe_spin_wait(c2.done[1]);
        }
        if (moe_hybrid_stats_on()) {
            harvest(c2.tm2, false);
        }
        const int64_t n_rows = ne01 - s.c2_row0;
        const float * out2 = (const float *) c2.out2.host;
        for (size_t j = 0; j < s.c2_expert.size(); ++j) {
            float * dst_row = (float *) ((char *) dst->data + s.c2_i1[j]*dst->nb[1] + s.c2_i2[j]*dst->nb[2]);
            memcpy(dst_row + s.c2_row0, out2 + j*n_rows, n_rows*sizeof(float));
        }
    }

    if (moe_hybrid_nancheck_on()) {
        static int64_t jcall = 0;
        jcall++;
        const int64_t n_in  = ggml_nelements(node->src1);
        const int64_t n_out = ggml_nelements(node->dst);
        if (n_in + n_out <= (int64_t) 8*1024*1024) {
            int64_t bi = 0, bo = 0; float vi = 0, vo = 0;
            const int si = moe_nan_scan((const float *) node->src1->data, n_in, &bi, &vi);
            const int so = moe_nan_scan((const float *) node->dst->data,  n_out, &bo, &vo);
            moe_nan_ring_add("join", node->src0->name, s.cur_dev, si, so, jcall);
            if (so != 0 || si != 0) {
                moe_nan_report("join", node->src0->name, s.cur_dev, jcall, si, so, so != 0 ? bo : bi, so != 0 ? vo : vi);
            }
        }
    }

    s.pair_slot.clear();
    s.c2_expert.clear();
}

// Rank blocks by decayed demand for the promotion policy. A block's score is the sum of its
// tensors' ratings; the name carries the block id ("blk.N.").
static void moe_hybrid_note_origin(const ggml_tensor * src0) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    moe_tensor_state * t = moe_tensor_for(s, src0); // creates the state entry; host registration stays lazy
    // note_origin is only ever called at facade creation, so it doubles as the load-time signal
    // that this tensor's residents live in the facade shards (the pool must not double-store
    // them). Gated on the facade graph switch: with it off, facade_run never learns the shard
    // geometry, and the pool must keep serving prefill.
    if (t != nullptr && moe_hybrid_facade_env_on()) {
        t->has_facade = true;
    }
}

static int moe_hybrid_top_experts(const char * tensor_name, int32_t * experts, int max_n) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (moe_tensor_state * t : s.order) {
        if (t->name != tensor_name) {
            continue;
        }
        // Fold the decode-side visits (counted on device by the facade prep kernel) into the
        // window first, so the ranking sees the demand of the tokens actually being generated,
        // not just prefills. Called between graphs, after the logits readback synced the GPU,
        // so the counter is quiescent and the fold is deterministic at temp 0.
        if (t->facade_counts != nullptr && !s.devs.empty()) {
            std::vector<int32_t> cnt(t->n_expert);
            ggml_cuda_set_device(s.devs[0].device);
            CUDA_CHECK(cudaMemcpy(cnt.data(), t->facade_counts, t->n_expert * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemset(t->facade_counts, 0, t->n_expert * sizeof(int32_t)));
            for (int64_t a = 0; a < t->n_expert; ++a) {
                t->window[a] += cnt[a];
            }
        }
        // A facade resident defends its slot with the hysteresis bonus, exactly like a pool
        // resident did: the periodic re-fill only swaps an expert whose challenger is clearly,
        // not marginally, hotter - near-ties must not cause 20 MB copies. Ties break by index so
        // the ranking (and therefore temp-0 output) stays deterministic.
        const double hys = moe_hybrid_hysteresis();
        auto score = [&](int32_t a) {
            const double v = t->rating[a] + (double) t->window[a];
            const bool resident = !t->facade_slot_of.empty() && t->facade_slot_of[a] >= 0;
            return resident ? v*hys : v;
        };
        std::vector<int32_t> idx(t->n_expert);
        for (int64_t a = 0; a < t->n_expert; ++a) {
            idx[a] = (int32_t) a;
        }
        std::sort(idx.begin(), idx.end(), [&](int32_t x, int32_t y) {
            const double sx = score(x), sy = score(y);
            return sx > sy || (sx == sy && x < y);
        });
        const int n = std::min<int>(max_n, (int) t->n_expert);
        for (int i = 0; i < n; ++i) {
            experts[i] = idx[i];
        }
        return n;
    }
    return 0;
}

static void moe_hybrid_note_facade_slot(const char * tensor_name, int32_t slot, int32_t expert) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (moe_tensor_state * t : s.order) {
        if (t->name != tensor_name) {
            continue;
        }
        if (t->facade_slot_of.empty()) {
            t->facade_slot_of.assign(t->n_expert, -1);
        }
        // Whoever held this slot before loses it - the re-fill unmaps (expert = -1), overwrites
        // the slot bytes, then maps the newcomer, so routing is correct at every instant: the
        // evicted expert falls back to its host weights the moment its slot is claimed.
        for (int64_t a = 0; a < t->n_expert; ++a) {
            if (t->facade_slot_of[a] == slot) {
                t->facade_slot_of[a] = -1;
            }
        }
        if (expert >= 0 && expert < (int32_t) t->n_expert) {
            t->facade_slot_of[expert] = slot;
        }
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            t->facade_dirty[d] = true;
        }
        return;
    }
}

static int moe_hybrid_top_blocks(int32_t * blocks, int max_n) {
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);

    std::unordered_map<int32_t, double> score;
    for (const moe_tensor_state * t : s.order) {
        int32_t b = -1;
        if (sscanf(t->name.c_str(), "blk.%d.", &b) != 1) {
            continue;
        }
        double sum = 0.0;
        for (int64_t a = 0; a < t->n_expert; ++a) {
            sum += t->rating[a] + (double) t->window[a];
        }
        score[b] += sum;
    }

    std::vector<std::pair<double, int32_t>> ranked;
    ranked.reserve(score.size());
    for (const auto & kv : score) {
        ranked.emplace_back(kv.second, kv.first);
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<>());

    int n = 0;
    for (const auto & kv : ranked) {
        if (n >= max_n) {
            break;
        }
        blocks[n++] = kv.second;
    }
    return n;
}

static const ggml_moe_hybrid_api g_api = {
    /* .supports = */ moe_hybrid_supports,
    /* .dispatch = */ moe_hybrid_dispatch,
    /* .launch      = */ moe_hybrid_launch,
    /* .join        = */ moe_hybrid_join,
    /* .note_origin      = */ moe_hybrid_note_origin,
};

// ---------------------------------------------------------------------------
// facade execution: a GPU-owned MUL_MAT_ID whose experts live wherever the pointer table says
// ---------------------------------------------------------------------------
//
// The facade tensor is meta-sharded, so this runs once per device on that device's row share.
// ids arrive on-device from the router; nothing here consults the host about routing:
//   1. prep: build the unique visited-expert list and remap ids to compact indices
//   2. gather: copy each unique expert's shard rows from table[expert] into compact staging
//      (a resident expert's entry points at the local facade shard = VRAM; a missing one at the
//      mapped host weights + this shard's row offset = one PCIe read, split across devices)
//   3. stock mat-vec on the staging with the remapped ids
// Worst-case unique count is bounded by the graph-builder threshold, so no host readback.

#define MOE_FACADE_MAGIC 0x4d6f4661 // 'MoFa'
// Bounds the staging and the graph-builder's facade cap: 4 tokens x top-8 (qwen35moe) = 32;
// top-4 models (mistral4) stay well inside it. Staging cost is MAX_UNIQ x shard_bytes per device.
#define MOE_FACADE_MAX_UNIQ 32

static __global__ void moe_facade_prep(const int32_t * ids, const int n_used, const int n_tokens,
                                       const int ids_stride,
                                       int32_t * remap, int32_t * uniq, const int n_expert,
                                       int32_t * counts) {
    // The graph's ids is the argsort_top_k VIEW: row stride is n_expert, NOT n_used. Reading it
    // flat is correct only at n_tokens == 1; at 2+ tokens the flat read serves token 0's ranks
    // n_used.. as "token 1's experts" - near-correct logits, token stutter. Read strided, write
    // the remap DENSE [n_used * n_tokens] so downstream descriptors can declare contiguous rows.
    const int n_ids = n_used * n_tokens;
    __shared__ int32_t compact_of[512];
    __shared__ int n_u;
    if (threadIdx.x == 0) {
        n_u = 0;
    }
    for (int a = threadIdx.x; a < n_expert; a += blockDim.x) {
        compact_of[a] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < n_ids; ++i) {
            const int32_t e = ids[(i % n_used) + (i / n_used) * ids_stride];
            if (e < 0 || e >= n_expert) {
                remap[i] = 0;
                continue;
            }
            if (counts != nullptr) {
                atomicAdd(&counts[e], 1); // decode demand, invisible to the CPU dispatch
            }
            if (compact_of[e] < 0 && n_u < MOE_FACADE_MAX_UNIQ) {
                compact_of[e] = n_u;
                uniq[n_u++] = e;
            }
            remap[i] = compact_of[e] < 0 ? 0 : compact_of[e];
        }
        for (int u = n_u; u < MOE_FACADE_MAX_UNIQ; ++u) {
            uniq[u] = -1;
        }
    }
}

// One block per (unique expert, shard row): copy this shard row's bytes from the origin expert,
// translated through the learned row map (segmented row splits) and column offset (contraction
// splits). table[e] points at the origin expert's base wherever it lives.
static __global__ void moe_facade_gather(void * const * table, const int32_t * uniq,
                                         const int32_t * rowmap, char * staging,
                                         const int64_t shard_rows, const int64_t shard_row_bytes,
                                         const int64_t origin_row_bytes, const int64_t col_off_bytes) {
    const int u = blockIdx.y;
    const int32_t e = uniq[u];
    if (e < 0) {
        return;
    }
    const int64_t r = blockIdx.x;
    const uintptr_t base = (uintptr_t) table[e];
    const char * src = (base & 1)
        ? (const char *) (base & ~(uintptr_t) 1) + r * shard_row_bytes            // facade slot, shard layout
        : (const char *) base + (int64_t) rowmap[r] * origin_row_bytes + col_off_bytes; // host, origin layout
    char * dst = staging + ((int64_t) u * shard_rows + r) * shard_row_bytes;
    // K-quant rows are only 2-byte aligned in general: a q6_K block is 210 bytes, so a row is a
    // multiple of 16 only when cols % 2048 == 0 (Qwen down shards are 420 B, Mistral down shards
    // 840 B). The vector path silently dropped the tail bytes of every such row - the final
    // super-block's scales and d - and read int4 through misaligned pointers. Only take it when
    // the row size and both endpoints are 16-byte clean; otherwise copy 2-byte units.
    if (((shard_row_bytes | (int64_t) (uintptr_t) src | (int64_t) (uintptr_t) dst) & 15) == 0) {
        for (int64_t i = threadIdx.x * 16; i + 16 <= shard_row_bytes; i += (int64_t) blockDim.x * 16) {
            *(int4 *) (dst + i) = *(const int4 *) (src + i);
        }
    } else {
        for (int64_t i = threadIdx.x * 2; i + 2 <= shard_row_bytes; i += (int64_t) blockDim.x * 2) {
            *(int16_t *) (dst + i) = *(const int16_t *) (src + i);
        }
        if (threadIdx.x == 0 && (shard_row_bytes & 1)) {
            dst[shard_row_bytes - 1] = src[shard_row_bytes - 1];
        }
    }
}

// Scratch shared by all facade nodes of one device; nodes are stream-serialized.
struct moe_facade_scratch {
    moe_scratch stage;   // gathered expert rows
    moe_scratch idsbuf;  // remap + uniq
};
static moe_facade_scratch g_facade_scratch[GGML_CUDA_MAX_DEVICES];

bool ggml_moe_hybrid_facade_run(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (src0->op_params[0] != (int32_t) MOE_FACADE_MAGIC) {
        return false;
    }
    moe_hybrid_state & s = moe_state();
    std::lock_guard<std::mutex> lock(s.mutex);

    // the facade shares its origin tensor's name; the origin carries the host mapping
    moe_tensor_state * t = nullptr;
    for (moe_tensor_state * c : s.order) {
        if (c->name == src0->name) {
            t = c;
            break;
        }
    }
    if (t == nullptr) {
        return false; // origin unknown; caller must not fall back silently
    }

    const int dev = ctx.device;
    if (t->host_dev == nullptr && !moe_register_host(t, dev, dev)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();

    // Learn this shard's exact geometry once from the load-time markers. Row-split shards (also
    // segmented ones, like the fused gate_up whose gate and up halves split independently) yield a
    // full row map; a shard whose markers are absent is a contraction split (down), whose column
    // offset follows from marker presence at column 0 and the shard/origin width ratio.
    const int64_t shard_rows = src0->ne[1];
    const int64_t shard_cols = src0->ne[0];
    const int64_t shard_row_bytes  = ggml_row_size(t->type, shard_cols);
    const int64_t origin_row_bytes = ggml_row_size(t->type, t->ne00);

    if (t->facade_rowmap[dev] == nullptr) {
        std::vector<int32_t> heads(shard_rows);
        CUDA_CHECK(cudaMemcpy2DAsync(heads.data(), sizeof(int32_t), src0->data, shard_row_bytes,
                                     sizeof(int32_t), shard_rows, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<int32_t> rowmap(shard_rows);
        const bool has_markers = (heads[0] & 0x7FFF0000) == 0x0FAC0000;
        if (has_markers) {
            for (int64_t r = 0; r < shard_rows; ++r) {
                rowmap[r] = ((heads[r] & 0x7FFF0000) == 0x0FAC0000) ? (heads[r] & 0xFFFF) : (int32_t) r;
            }
            t->facade_col_off[dev] = 0;
        } else {
            // contraction split: rows map to themselves; this shard holds the upper columns
            for (int64_t r = 0; r < shard_rows; ++r) {
                rowmap[r] = (int32_t) r;
            }
            t->facade_col_off[dev] = origin_row_bytes - shard_row_bytes;
        }
        ggml_cuda_set_device(dev);
        CUDA_CHECK(cudaMalloc(&t->facade_rowmap[dev], shard_rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(t->facade_rowmap[dev], rowmap.data(), shard_rows * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // host-side copy of the geometry for the unified-store launches (prefill/verify read the
        // facade shards directly): the rowmap compressed to contiguous runs for the join scatter
        t->facade_shard_rows[dev] = shard_rows;
        t->facade_shard_cols[dev] = shard_cols;
        t->facade_n_slots         = src0->ne[2];
        if (!has_markers) {
            t->facade_is_col = true; // contraction split: shards are column slices, partial sums
        }
        t->facade_segs_host[dev].clear();
        for (int64_t r = 0; r < shard_rows; ) {
            int64_t r2 = r + 1;
            while (r2 < shard_rows && rowmap[r2] == rowmap[r2-1] + 1) {
                r2++;
            }
            t->facade_segs_host[dev].push_back({ r, rowmap[r], r2 - r });
            r = r2;
        }
        t->facade_data[dev] = src0->data;
        t->facade_row_off[dev] = has_markers ? 0 : -2; // diagnostics only
    }
    const int64_t shard_bytes = shard_row_bytes * shard_rows;

    // per-device facade table: v1 serves every expert from mapped host (correctness baseline);
    // the perf step will route residents at filled facade slots
    if (t->facade_table[dev] == nullptr) {
        ggml_cuda_set_device(dev);
        CUDA_CHECK(cudaMalloc(&t->facade_table[dev], t->n_expert * sizeof(void *)));
        t->facade_dirty[dev] = true;
    }
    bool table_rebuilt = false;
    if (t->facade_dirty[dev]) {
        std::vector<void *> table(t->n_expert);
        const int64_t slot_shard_bytes = shard_row_bytes * shard_rows;
        for (int64_t a = 0; a < t->n_expert; ++a) {
            const int32_t fsl = a < (int64_t) t->facade_slot_of.size() ? t->facade_slot_of[a] : -1;
            if (fsl >= 0 && t->facade_data[dev] != nullptr) {
                // resident: the facade shard already holds this expert in SHARD layout; the tag
                // bit tells the gather to address it identically instead of translating
                table[a] = (void *) ((uintptr_t) ((char *) t->facade_data[dev] + (size_t) fsl * slot_shard_bytes) | 1);
            } else {
                table[a] = (char *) t->host_dev + (size_t) a * t->expert_bytes;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(t->facade_table[dev], table.data(), t->n_expert * sizeof(void *),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t->facade_dirty[dev] = false;
        table_rebuilt = true;
    }

    // env GGML_MOE_HYBRID_STORE_CHECK: after each table (re)build, read every mapped slot back
    // from the device and byte-compare it against the origin host weights through the learned
    // geometry - the oracle for "does the store hold what the gather's resident path assumes".
    static const bool store_check = [] {
        const char * sc = getenv("GGML_MOE_HYBRID_STORE_CHECK");
        return sc != nullptr && sc[0] != '\0' && sc[0] != '0';
    }();
    if (store_check && table_rebuilt && t->facade_data[dev] != nullptr) {
        const int64_t slot_shard_bytes = shard_row_bytes * shard_rows;
        std::vector<char> slot_bytes(slot_shard_bytes);
        std::vector<char> origin_row(shard_row_bytes);
        static int printed = 0;
        for (int64_t a = 0; a < (int64_t) t->facade_slot_of.size(); ++a) {
            const int32_t fsl = t->facade_slot_of[a];
            if (fsl < 0) {
                continue;
            }
            CUDA_CHECK(cudaMemcpy(slot_bytes.data(),
                (const char *) t->facade_data[dev] + (size_t) fsl * slot_shard_bytes,
                slot_shard_bytes, cudaMemcpyDeviceToHost));
            int64_t bad_rows = 0, first_row = -1, first_off = -1;
            for (const moe_facade_seg & g : t->facade_segs_host[dev]) {
                for (int64_t i = 0; i < g.len; ++i) {
                    const char * dev_row = slot_bytes.data() + (g.shard_r + i) * shard_row_bytes;
                    CUDA_CHECK(cudaMemcpy(origin_row.data(),
                        (const char *) t->host_dev + (size_t) a * t->expert_bytes +
                            (size_t) (g.origin_r + i) * origin_row_bytes + t->facade_col_off[dev],
                        shard_row_bytes, cudaMemcpyDeviceToHost));
                    if (memcmp(dev_row, origin_row.data(), shard_row_bytes) != 0) {
                        bad_rows++;
                        if (first_row < 0) {
                            first_row = g.shard_r + i;
                            for (int64_t b = 0; b < shard_row_bytes; ++b) {
                                if (dev_row[b] != origin_row[b]) { first_off = b; break; }
                            }
                        }
                    }
                }
            }
            if (bad_rows > 0 && printed < 24) {
                printed++;
                fprintf(stderr, "[moe-facade] STORE MISMATCH %s dev %d expert %lld slot %d: "
                        "%lld/%lld shard rows differ, first row %lld byte %lld%c",
                        t->name.c_str(), dev, (long long) a, fsl, (long long) bad_rows,
                        (long long) shard_rows, (long long) first_row, (long long) first_off, 10);
            }
        }
        if (printed == 0) {
            static std::set<std::string> ok_logged;
            if (ok_logged.insert(t->name + std::to_string(dev)).second) {
                fprintf(stderr, "[moe-facade] store check OK: %s dev %d%c", t->name.c_str(), dev, 10);
            }
        }
    }

    const int64_t n_ids = ids->ne[0] * ids->ne[1];
    moe_facade_scratch & fs = g_facade_scratch[dev];
    ggml_cuda_set_device(dev);
    fs.idsbuf.reserve((n_ids + MOE_FACADE_MAX_UNIQ) * sizeof(int32_t));
    fs.stage.reserve((size_t) MOE_FACADE_MAX_UNIQ * shard_bytes);

    int32_t * remap_dev = (int32_t *) fs.idsbuf.dev;
    int32_t * uniq_dev  = remap_dev + n_ids;

    // decode demand is counted on device 0's shard only - both devices see the same ids, so
    // counting on both would double every visit
    if (dev == 0 && t->facade_counts == nullptr) {
        CUDA_CHECK(cudaMalloc(&t->facade_counts, t->n_expert * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(t->facade_counts, 0, t->n_expert * sizeof(int32_t), stream));
    }
    moe_facade_prep<<<1, 256, 0, stream>>>((const int32_t *) ids->data, (int) ids->ne[0],
                                           (int) ids->ne[1], (int) (ids->nb[1] / sizeof(int32_t)),
                                           remap_dev, uniq_dev, (int) t->n_expert,
                                           dev == 0 ? (int32_t *) t->facade_counts : nullptr);
    {
        dim3 grid((unsigned) shard_rows, MOE_FACADE_MAX_UNIQ);
        moe_facade_gather<<<grid, 128, 0, stream>>>((void * const *) t->facade_table[dev],
            uniq_dev, (const int32_t *) t->facade_rowmap[dev], (char *) fs.stage.dev,
            shard_rows, shard_row_bytes, origin_row_bytes, t->facade_col_off[dev]);
    }
    CUDA_CHECK(cudaGetLastError());

    // stock mat-vec over the compact staging, ids remapped to compact indices
    ggml_tensor w {};
    w.type  = t->type;
    w.data  = fs.stage.dev;
    w.ne[0] = shard_cols; w.ne[1] = shard_rows; w.ne[2] = MOE_FACADE_MAX_UNIQ; w.ne[3] = 1;
    w.nb[0] = ggml_type_size(t->type);
    w.nb[1] = shard_row_bytes;
    w.nb[2] = shard_bytes;
    w.nb[3] = shard_bytes * MOE_FACADE_MAX_UNIQ;
    w.buffer = src0->buffer;

    ggml_tensor ids_r = *ids;
    ids_r.data = remap_dev;
    // the remap buffer is DENSE [ne0, ne1] - never inherit the argsort view's row stride, or
    // the stock entry's ids_stride walks past the buffer for every token after the first
    ids_r.nb[0] = sizeof(int32_t);
    ids_r.nb[1] = ids->ne[0] * sizeof(int32_t);
    ids_r.nb[2] = ids->ne[0] * ids->ne[1] * sizeof(int32_t);
    ids_r.nb[3] = ids_r.nb[2];

    // the stock kernels reach the operands through dst->src[], so the clone must point at the
    // synthetic staging descriptors (same trap the hybrid launch path documented)
    ggml_tensor dst_r = *dst;
    dst_r.src[0] = &w;
    dst_r.src[1] = const_cast<ggml_tensor *>(src1);
    dst_r.src[2] = &ids_r;

    if (ids->ne[1] <= 1) {
        ggml_cuda_mul_mat_vec_q(ctx, &w, src1, &ids_r, &dst_r);
    } else {
        // Multi-token verify batches: decompose into per-token single-column calls. The
        // combined multi-token ids launch produced token stutter (near-correct logits) that
        // resisted a full contract audit of the moe kernel and its entry strides; the
        // single-token path is content-validated, and at 2-4 tokens the extra launches cost
        // microseconds. Bonus: ncols_dst == 1 keeps q6_K on the fast RDNA4 mmvk route.
        for (int64_t tok = 0; tok < ids->ne[1]; ++tok) {
            ggml_tensor ids_t = ids_r;
            ids_t.data  = (char *) ids_r.data + tok*ids->ne[0]*sizeof(int32_t);
            ids_t.ne[1] = 1;

            ggml_tensor src1_t = *src1;
            src1_t.data  = (char *) src1->data + tok*src1->nb[2];
            src1_t.ne[2] = 1;

            ggml_tensor dst_t = *dst;
            dst_t.data  = (char *) dst->data + tok*dst->nb[2];
            dst_t.ne[2] = 1;
            dst_t.src[0] = &w;
            dst_t.src[1] = &src1_t;
            dst_t.src[2] = &ids_t;

            ggml_cuda_mul_mat_vec_q(ctx, &w, &src1_t, &ids_t, &dst_t);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    moe_debug_checkpoint("facade-decode", src0->name, dev, n_ids, shard_rows, false);
    if (moe_hybrid_nancheck_on()) {
        static int64_t fcall = 0;
        fcall++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int64_t n_in  = ggml_nelements(src1);
        const int64_t n_out = ggml_nelements(dst);
        if (n_in + n_out <= (int64_t) 8*1024*1024) {
            std::vector<float> hin(n_in), hout(n_out);
            CUDA_CHECK(cudaMemcpy(hin.data(),  src1->data, n_in*sizeof(float),  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hout.data(), dst->data,  n_out*sizeof(float), cudaMemcpyDeviceToHost));
            int64_t bi = 0, bo = 0; float vi = 0, vo = 0;
            const int si = moe_nan_scan(hin.data(),  n_in,  &bi, &vi);
            const int so = moe_nan_scan(hout.data(), n_out, &bo, &vo);
            moe_nan_ring_add("facade", src0->name, dev, si, so, fcall);
            if (fcall >= 1400 && fcall <= 1600) {
                fprintf(stderr, "[moe-win] %lld facade %s dev %d in %s out %s%c", (long long) fcall,
                        src0->name, dev, moe_nan_st(si), moe_nan_st(so), 10);
            }
            if (so != 0 || si != 0) {
                moe_nan_report("facade", src0->name, dev, fcall, si, so, so != 0 ? bo : bi, so != 0 ? vo : vi);
            }
        }
    }
    static int logged[GGML_CUDA_MAX_DEVICES] = { 0 };
    if (logged[dev] < 2) {
        logged[dev]++;
        fprintf(stderr, "[moe-facade] serving %s on device %d (shard %lldx%lld of %lldx%lld, col off %lld)%c",
                src0->name, dev, (long long) shard_cols, (long long) shard_rows,
                (long long) t->ne00, (long long) t->ne01, (long long) t->facade_col_off[dev], 10);
    }
    return true;
}

void ggml_moe_hybrid_cuda_register(void) {
    // unconditional: the operator's choice is not readable yet, so supports() gates instead
    ggml_moe_hybrid_register(&g_api);
}

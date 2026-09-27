#include "moe-tiered.cuh"

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// See moe-tiered.cuh. VRAM layout of a tiered expert tensor (its allocation starts at tensor->data):
//   [hot experts, n_hot*nb[2]] [zeroed row padding] [pad to 16] [table: n_expert device addresses]
// and its cold experts in one pinned mapped host block: [cold experts, n_cold*nb[2]] [zeroed row padding].

namespace {

struct hot_lists {
    bool disabled = false;                         // GGML_CUDA_MOE_TIERED=0: every expert cold
    std::string path;
    std::map<int, std::vector<int32_t>> layers;    // layer -> sorted unique hot expert ids
};

const hot_lists & get_hot_lists() {
    static std::once_flag once;
    static hot_lists h;
    std::call_once(once, [] {
        const char * off = getenv("GGML_CUDA_MOE_TIERED");
        h.disabled = off != nullptr && atoi(off) == 0;
        const char * p = getenv("GGML_CUDA_MOE_HOT_FILE");
        if (p == nullptr || h.disabled) {
            GGML_LOG_INFO("%s: tiered experts: %s, every expert cold\n", __func__,
                h.disabled ? "GGML_CUDA_MOE_TIERED=0" : "no GGML_CUDA_MOE_HOT_FILE");
            return;
        }
        h.path = p;
        std::ifstream f(p);
        if (!f) {
            GGML_ABORT("tiered experts: cannot read GGML_CUDA_MOE_HOT_FILE=%s", p);
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream ss(line);
            std::string key;
            ss >> key;
            int layer = -1;
            if (sscanf(key.c_str(), "blk.%d", &layer) != 1 || layer < 0) {
                GGML_ABORT("tiered experts: bad line in %s: %s", p, line.c_str());
            }
            std::vector<int32_t> ids;
            for (int32_t e; ss >> e;) {
                if (e < 0) {
                    GGML_ABORT("tiered experts: negative expert id in %s, layer %d", p, layer);
                }
                ids.push_back(e);
            }
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
                GGML_ABORT("tiered experts: duplicate expert id in %s, layer %d", p, layer);
            }
            h.layers[layer] = std::move(ids);
        }
        GGML_LOG_INFO("%s: tiered experts: hot lists for %zu layers from %s\n", __func__, h.layers.size(), p);
    });
    return h;
}

// a tensor that gets the tiered layout: contiguous quantized [ne0, ne1, n_expert]; anything else placed on a
// _TIERED buffer is an ordinary VRAM tensor
bool is_expert_tensor(const ggml_tensor * t) {
    return t->view_src == nullptr && ggml_is_quantized(t->type) && t->ne[2] > 1 && t->ne[3] == 1 &&
        ggml_is_contiguous(t);
}

struct tiered_layout {
    std::vector<int32_t> hot;
    int64_t n_expert     = 0;
    size_t  expert_bytes = 0;
    size_t  pad          = 0;   // row padding after the last expert of a block (MATRIX_ROW_PADDING)
    size_t  table_off    = 0;
    size_t  vram_size    = 0;
    size_t  cold_size    = 0;
};

tiered_layout get_layout(const ggml_tensor * t) {
    tiered_layout l;
    l.n_expert     = t->ne[2];
    l.expert_bytes = t->nb[2];
    if (t->ne[0] % MATRIX_ROW_PADDING != 0) {
        l.pad = ggml_row_size(t->type, MATRIX_ROW_PADDING - t->ne[0] % MATRIX_ROW_PADDING);
    }
    const hot_lists & h = get_hot_lists();
    int layer = -1;
    if (!h.disabled && sscanf(t->name, "blk.%d.", &layer) == 1) {
        const auto it = h.layers.find(layer);
        if (it != h.layers.end()) {
            if (!it->second.empty() && it->second.back() >= l.n_expert) {
                GGML_ABORT("tiered experts: %s has %" PRId64 " experts but the hot list names expert %d",
                    t->name, l.n_expert, it->second.back());
            }
            l.hot = it->second;
        }
    }
    const size_t n_hot  = l.hot.size();
    const size_t n_cold = (size_t) l.n_expert - n_hot;
    l.table_off = GGML_PAD(n_hot*l.expert_bytes + (n_hot ? l.pad : 0), 16);
    l.vram_size = l.table_off + (size_t) l.n_expert*sizeof(void *);
    l.cold_size = n_cold ? n_cold*l.expert_bytes + l.pad : 0;
    return l;
}

struct tiered_tensor {
    const void ** d_table = nullptr;          // in the tensor's VRAM allocation
    std::vector<const void *> h_table;        // host copy of the device addresses
    std::vector<int32_t> slot;                // expert -> hot slot (>= 0) or -1 - cold slot
    char *  hot_dev      = nullptr;
    char *  cold_host    = nullptr;
    char *  cold_dev     = nullptr;
    size_t  expert_bytes = 0;
    int64_t n_expert     = 0;
    size_t  n_hot        = 0;
    size_t  cold_size    = 0;                 // bytes of the cold block (with its row padding)
    int     device       = -1;
    int     layer        = -1;                // blk.<layer>, -1 if the name has none
    bool    is_down      = false;             // ffn_down_exps: the last expert matmul of its layer
    const void ** d_table_staged[4] = {};     // per staging area: the table with the cold experts there (prefill)

    bool  hot(int64_t e)            const { return slot[e] >= 0; }
    char * host(int64_t e)          const { return cold_host + (size_t) (-1 - slot[e])*expert_bytes; }
    char * dev(int64_t e)           const { return (char *) h_table[e]; }
};

// Prefill staging (see ggml_cuda_tiered_table_prefill). The tiered tensors of every device, and per device the
// staging area, its copy stream and the layer it holds.
struct stage_member {
    tiered_tensor * t;
    size_t          off;   // in the staging area
};

struct stage_state {
    static constexpr int max_areas = 4;
    std::mutex mutex;
    std::vector<tiered_tensor *> tensors;      // registered at init
    bool         ready   = false;              // set up (or failed: n_areas == 0)
    int          n_areas = 0;                  // staging areas, each one layer's cold experts (GGML_CUDA_MOE_STAGE_AREAS)
    std::vector<int> layers;                   // the layers with cold experts, ascending
    std::map<int, std::vector<stage_member>> members;
    size_t       size   = 0;                   // of one area
    char *       buf    = nullptr;             // n_areas areas
    void **      tables = nullptr;             // the staged tables of all tensors for every area, one allocation
    cudaStream_t stream = nullptr;
    cudaEvent_t  ev_staged  [max_areas] = {};  // after the copies into the area
    cudaEvent_t  ev_released[max_areas] = {};  // after the last expert matmul that read the area
    bool         released   [max_areas] = {};  // ev_released holds a matmul the next copy has to wait for
    int          area_layer [max_areas];       // the layer an area holds or is being filled with, -1 free

    int area_of(int layer) const {
        for (int a = 0; a < n_areas; ++a) {
            if (area_layer[a] == layer) {
                return a;
            }
        }
        return -1;
    }
};

stage_state g_stage[GGML_CUDA_MAX_DEVICES];

struct stage_state_init {
    stage_state_init() {
        for (auto & st : g_stage) {
            for (int a = 0; a < stage_state::max_areas; ++a) {
                st.area_layer[a] = -1;
            }
        }
    }
} g_stage_init;

// back to "not set up": the next prefill sets the staging up again over the tensors registered then. Waits for the
// copies in flight, which read the cold blocks of the tensors (a buffer being freed, or new tensors being placed).
void stage_reset(stage_state & st, int device) {
    if (st.stream) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamSynchronize(st.stream));
    }
    if (st.buf) {
        CUDA_CHECK(cudaFree(st.buf));
        CUDA_CHECK(cudaFree(st.tables));
    }
    for (tiered_tensor * t : st.tensors) {
        for (auto & tab : t->d_table_staged) {
            tab = nullptr;
        }
    }
    st.buf     = nullptr;
    st.tables  = nullptr;
    st.size    = 0;
    st.n_areas = 0;
    st.layers.clear();
    st.members.clear();
    for (int a = 0; a < stage_state::max_areas; ++a) {
        st.area_layer[a] = -1;
        st.released[a]   = false;
    }
    st.ready   = false;
}

struct tiered_buffer_context {
    int    device;
    void * dev_ptr;
    std::vector<std::unique_ptr<tiered_tensor>> tensors;

    ~tiered_buffer_context() {
        {
            // the prefill staging may still copy from these cold blocks: stop it and forget these tensors first
            stage_state & st = g_stage[device];
            std::lock_guard<std::mutex> lock(st.mutex);
            stage_reset(st, device);
            for (auto & t : tensors) {
                st.tensors.erase(std::remove(st.tensors.begin(), st.tensors.end(), t.get()), st.tensors.end());
            }
        }
        for (auto & t : tensors) {
            if (t->cold_host) {
                CUDA_CHECK(cudaFreeHost(t->cold_host));
            }
        }
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};


// placement totals per device, printed once at the first MUL_MAT_ID that reads a table
std::atomic<size_t> g_hot_bytes[GGML_CUDA_MAX_DEVICES];
std::atomic<size_t> g_cold_bytes[GGML_CUDA_MAX_DEVICES];
std::atomic<int>    g_tensors[GGML_CUDA_MAX_DEVICES];
std::atomic<bool>   g_summary_done{false};

void tiered_free_buffer(ggml_backend_buffer_t buffer) {
    delete (tiered_buffer_context *) buffer->context;
}

void * tiered_get_base(ggml_backend_buffer_t buffer) {
    return ((tiered_buffer_context *) buffer->context)->dev_ptr;
}

const tiered_tensor * tiered_info(const ggml_tensor * t) {
    if (t == nullptr || t->extra == nullptr || t->buffer == nullptr || t->buffer->iface.free_buffer != tiered_free_buffer) {
        return nullptr;
    }
    return (const tiered_tensor *) t->extra;
}

enum ggml_status tiered_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    if (tensor->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    ggml_cuda_set_device(ctx->device);

    if (!is_expert_tensor(tensor)) {
        // an ordinary VRAM tensor: zero the row padding like the regular CUDA buffer
        const size_t original = ggml_nbytes(tensor);
        const size_t padded   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
        if (padded > original && ggml_is_quantized(tensor->type)) {
            CUDA_CHECK(cudaMemset((char *) tensor->data + original, 0, padded - original));
        }
        return GGML_STATUS_SUCCESS;
    }

    const tiered_layout l = get_layout(tensor);
    auto info = std::make_unique<tiered_tensor>();
    info->expert_bytes = l.expert_bytes;
    info->n_expert     = l.n_expert;
    info->n_hot        = l.hot.size();
    info->cold_size    = l.cold_size;
    info->device       = ctx->device;
    if (sscanf(tensor->name, "blk.%d.", &info->layer) != 1) {
        info->layer = -1;
    }
    info->is_down      = strstr(tensor->name, "ffn_down_exps") != nullptr;
    info->hot_dev      = (char *) tensor->data;
    info->d_table      = (const void **) ((char *) tensor->data + l.table_off);

    if (l.cold_size > 0) {
        unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
#ifdef GGML_USE_HIP
        if (getenv("GGML_CUDA_UVA_NONCOHERENT") != nullptr) {
            flags |= hipHostMallocNonCoherent;
        }
#endif
        void * host = nullptr;
        cudaError_t err = cudaHostAlloc(&host, l.cold_size, flags);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: %s: allocating %.2f MiB of mapped host memory failed: %s\n", __func__, tensor->name,
                l.cold_size/1024.0/1024.0, cudaGetErrorString(err));
            return GGML_STATUS_ALLOC_FAILED;
        }
        void * dev = nullptr;
        err = cudaHostGetDevicePointer(&dev, host, 0);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            CUDA_CHECK(cudaFreeHost(host));
            GGML_LOG_ERROR("%s: %s: no device pointer for mapped host memory: %s\n", __func__, tensor->name, cudaGetErrorString(err));
            return GGML_STATUS_ALLOC_FAILED;
        }
        info->cold_host = (char *) host;
        info->cold_dev  = (char *) dev;
        if (l.pad) {
            memset(info->cold_host + (l.n_expert - info->n_hot)*l.expert_bytes, 0, l.pad);
        }
    }
    if (info->n_hot && l.pad) {
        CUDA_CHECK(cudaMemset(info->hot_dev + info->n_hot*l.expert_bytes, 0, l.pad));
    }

    info->slot.assign(l.n_expert, 0);
    std::vector<char> is_hot(l.n_expert, 0);
    for (size_t s = 0; s < l.hot.size(); ++s) {
        is_hot[l.hot[s]]      = 1;
        info->slot[l.hot[s]]  = (int32_t) s;
    }
    int32_t c = 0;
    info->h_table.resize(l.n_expert);
    for (int64_t e = 0; e < l.n_expert; ++e) {
        if (!is_hot[e]) {
            info->slot[e] = -1 - c++;
        }
        info->h_table[e] = info->slot[e] >= 0 ? info->hot_dev  + (size_t) info->slot[e]*l.expert_bytes
                                              : info->cold_dev + (size_t) (-1 - info->slot[e])*l.expert_bytes;
    }
    CUDA_CHECK(cudaMemcpy(info->d_table, info->h_table.data(), l.n_expert*sizeof(void *), cudaMemcpyHostToDevice));

    g_hot_bytes [ctx->device] += info->n_hot*l.expert_bytes;
    g_cold_bytes[ctx->device] += (l.n_expert - info->n_hot)*l.expert_bytes;
    g_tensors   [ctx->device] += 1;

    tensor->extra = info.get();
    {
        stage_state & st = g_stage[ctx->device];
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.ready) {
            stage_reset(st, ctx->device); // a model loaded after a prefill: set up again with its tensors
        }
        st.tensors.push_back(info.get());
    }
    ctx->tensors.push_back(std::move(info));
    return GGML_STATUS_SUCCESS;
}

// Byte-range copies into / out of a tiered tensor, split per expert. Writes to a hot expert are gathered in a
// host staging buffer while they stay contiguous and go to the device in one transfer: the -sm tensor loader
// writes an axis-0 shard row by row (about a million small copies per tensor).
struct tiered_writer {
    const tiered_tensor & t;
    std::vector<char> staging;
    int64_t run_e  = -1;
    size_t  run_lo = 0;
    size_t  run_hi = 0;

    explicit tiered_writer(const tiered_tensor & t) : t(t) {}

    void flush() {
        if (run_e >= 0 && run_hi > run_lo) {
            CUDA_CHECK(cudaMemcpy(t.dev(run_e) + run_lo, staging.data() + run_lo, run_hi - run_lo, cudaMemcpyHostToDevice));
        }
        run_e = -1;
    }

    void write(size_t off, const char * src, size_t size) {
        while (size > 0) {
            const int64_t e      = off / t.expert_bytes;
            const size_t  within = off % t.expert_bytes;
            GGML_ASSERT(e < t.n_expert);
            const size_t  len    = std::min(size, t.expert_bytes - within);
            if (!t.hot(e)) {
                memcpy(t.host(e) + within, src, len);
            } else {
                if (e != run_e || within != run_hi) {
                    flush();
                    run_e  = e;
                    run_lo = run_hi = within;
                    if (staging.size() < t.expert_bytes) {
                        staging.resize(t.expert_bytes);
                    }
                }
                memcpy(staging.data() + within, src, len);
                run_hi = within + len;
            }
            off += len; src += len; size -= len;
        }
    }
};

void tiered_read(const tiered_tensor & t, size_t off, char * dst, size_t size) {
    while (size > 0) {
        const int64_t e      = off / t.expert_bytes;
        const size_t  within = off % t.expert_bytes;
        GGML_ASSERT(e < t.n_expert);
        const size_t  len    = std::min(size, t.expert_bytes - within);
        if (!t.hot(e)) {
            memcpy(dst, t.host(e) + within, len);
        } else {
            CUDA_CHECK(cudaMemcpy(dst, t.dev(e) + within, len, cudaMemcpyDeviceToHost));
        }
        off += len; dst += len; size -= len;
    }
}

void tiered_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    const tiered_tensor * t = tiered_info(tensor);
    if (t == nullptr) {
        CUDA_CHECK(cudaMemset((char *) tensor->data + offset, value, size));
        return;
    }
    while (size > 0) {
        const int64_t e      = offset / t->expert_bytes;
        const size_t  within = offset % t->expert_bytes;
        GGML_ASSERT(e < t->n_expert);
        const size_t  len    = std::min(size, t->expert_bytes - within);
        if (!t->hot(e)) {
            memset(t->host(e) + within, value, len);
        } else {
            CUDA_CHECK(cudaMemset(t->dev(e) + within, value, len));
        }
        offset += len; size -= len;
    }
}

void tiered_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    const tiered_tensor * t = tiered_info(tensor);
    if (t == nullptr) {
        CUDA_CHECK(cudaMemcpy((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice));
        return;
    }
    tiered_writer w(*t);
    w.write(offset, (const char *) data, size);
    w.flush();
}

void tiered_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    const tiered_tensor * t = tiered_info(tensor);
    if (t == nullptr) {
        CUDA_CHECK(cudaMemcpy(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost));
        return;
    }
    tiered_read(*t, offset, (char *) data, size);
}

void tiered_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    const tiered_tensor * t = tiered_info(tensor);
    if (t == nullptr) {
        for (size_t i = 0; i < n_copies; i++) {
            CUDA_CHECK(cudaMemcpy((char *) tensor->data + offset + i*stride_tensor, (const char *) data + i*stride_data,
                size, cudaMemcpyHostToDevice));
        }
        return;
    }
    tiered_writer w(*t);
    for (size_t i = 0; i < n_copies; i++) {
        w.write(offset + i*stride_tensor, (const char *) data + i*stride_data, size);
    }
    w.flush();
}

void tiered_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    const tiered_tensor * t = tiered_info(tensor);
    for (size_t i = 0; i < n_copies; i++) {
        if (t == nullptr) {
            CUDA_CHECK(cudaMemcpy((char *) data + i*stride_data, (const char *) tensor->data + offset + i*stride_tensor,
                size, cudaMemcpyDeviceToHost));
        } else {
            tiered_read(*t, offset + i*stride_tensor, (char *) data + i*stride_data, size);
        }
    }
}

bool tiered_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) {
    return false;
}

void tiered_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    tiered_buffer_context * ctx = (tiered_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    for (auto & t : ctx->tensors) {
        if (t->cold_host) {
            memset(t->cold_host, value, (t->n_expert - t->n_hot)*t->expert_bytes);
        }
        // the clear wiped the tables along with the data
        CUDA_CHECK(cudaMemcpy(t->d_table, t->h_table.data(), t->n_expert*sizeof(void *), cudaMemcpyHostToDevice));
    }
}

const ggml_backend_buffer_i tiered_buffer_interface = {
    /* .free_buffer     = */ tiered_free_buffer,
    /* .get_base        = */ tiered_get_base,
    /* .init_tensor     = */ tiered_init_tensor,
    /* .memset_tensor   = */ tiered_memset_tensor,
    /* .set_tensor      = */ tiered_set_tensor,
    /* .get_tensor      = */ tiered_get_tensor,
    /* .set_tensor_2d   = */ tiered_set_tensor_2d,
    /* .get_tensor_2d   = */ tiered_get_tensor_2d,
    /* .cpy_tensor      = */ tiered_cpy_tensor,
    /* .clear           = */ tiered_clear,
    /* .reset           = */ NULL,
};

struct tiered_buffer_type_context {
    int         device;
    std::string name;
};

const char * tiered_buft_get_name(ggml_backend_buffer_type_t buft) {
    return ((tiered_buffer_type_context *) buft->context)->name.c_str();
}

ggml_backend_buffer_t tiered_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    tiered_buffer_type_context * buft_ctx = (tiered_buffer_type_context *) buft->context;
    ggml_cuda_set_device(buft_ctx->device);
    void * dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, std::max<size_t>(size, 1));
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d failed: %s\n", __func__, size/1024.0/1024.0,
            buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }
    tiered_buffer_context * ctx = new tiered_buffer_context{buft_ctx->device, dev_ptr, {}};
    return ggml_backend_buffer_init(buft, tiered_buffer_interface, ctx, size);
}

size_t tiered_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 128;
}

size_t tiered_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    if (is_expert_tensor(tensor)) {
        return get_layout(tensor).vram_size;
    }
    size_t size = ggml_nbytes(tensor);
    if (ggml_is_quantized(tensor->type) && tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING);
    }
    return size;
}

const ggml_backend_buffer_type_i tiered_buffer_type_interface = {
    /* .get_name         = */ tiered_buft_get_name,
    /* .alloc_buffer     = */ tiered_buft_alloc_buffer,
    /* .get_alignment    = */ tiered_buft_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ tiered_buft_get_alloc_size,
    /* .is_host          = */ NULL, // must stay NULL: the loader and the scheduler treat host buffers as CPU memory
};

} // namespace

ggml_backend_buffer_type_t ggml_backend_cuda_tiered_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    static ggml_backend_buffer_type types[GGML_CUDA_MAX_DEVICES];
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            types[i] = {
                /* .iface    = */ tiered_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new tiered_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i) + "_TIERED"},
            };
        }
        initialized = true;
    }
    return &types[device];
}

bool ggml_backend_buft_is_cuda_tiered(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == tiered_buft_get_name;
}

const void * const * ggml_cuda_tiered_table(const ggml_tensor * t) {
    const tiered_tensor * info = tiered_info(t);
    if (info == nullptr) {
        return nullptr;
    }
    if (!g_summary_done.exchange(true)) {
        for (int d = 0; d < ggml_backend_cuda_get_device_count(); ++d) {
            if (g_tensors[d] > 0) {
                GGML_LOG_INFO("%s: tiered experts on %s%d: %d tensors, %.2f GiB hot (VRAM), %.2f GiB cold (mapped host)\n",
                    __func__, GGML_CUDA_NAME, d, g_tensors[d].load(), g_hot_bytes[d]/1073741824.0, g_cold_bytes[d]/1073741824.0);
            }
        }
    }
    return (const void * const *) info->d_table;
}

const void * ggml_cuda_tiered_expert(const ggml_tensor * t, int64_t expert) {
    const tiered_tensor * info = tiered_info(t);
    GGML_ASSERT(info != nullptr && expert >= 0 && expert < info->n_expert);
    return info->h_table[expert];
}

namespace {

bool stage_enabled(int64_t n_tokens) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_STAGE");
        return env == nullptr || atoi(env) != 0;
    }();
    static const int64_t min_tokens = [] {
        const char * env = getenv("GGML_CUDA_MOE_STAGE_MIN_TOKENS");
        return env != nullptr ? (int64_t) atoll(env) : (int64_t) 256;
    }();
    return enabled && n_tokens >= min_tokens;
}

// the staging area of a device: one layer's cold experts, the largest layer's size; a staged table per tensor
void stage_setup(stage_state & st, int device) {
    st.ready = true;
    size_t n_ptrs = 0;
    for (tiered_tensor * t : st.tensors) {
        if (t->cold_size == 0 || t->layer < 0) {
            continue;
        }
        auto & m = st.members[t->layer];
        size_t off = 0;
        for (const stage_member & x : m) {
            off = std::max(off, GGML_PAD(x.off + x.t->cold_size, 256));
        }
        m.push_back({t, off});
        st.size = std::max(st.size, GGML_PAD(off + t->cold_size, 256));
        n_ptrs += t->n_expert;
    }
    for (const auto & it : st.members) {
        st.layers.push_back(it.first);
    }
    if (st.size == 0) {
        return;
    }
    // two areas by default: the next layer but one is copied while the next one is read, so a copy has about two
    // layers of compute to hide behind (with one area it had only the part of a layer before its expert matmuls)
    static const int n_areas = [] {
        const char * env = getenv("GGML_CUDA_MOE_STAGE_AREAS");
        return std::max(1, std::min(env ? atoi(env) : 2, stage_state::max_areas));
    }();
    ggml_cuda_set_device(device);
    if (cudaMalloc((void **) &st.buf, n_areas*st.size) != cudaSuccess ||
            cudaMalloc((void **) &st.tables, n_areas*n_ptrs*sizeof(void *)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (st.buf) {
            CUDA_CHECK(cudaFree(st.buf));
        }
        st.buf = nullptr;
        GGML_LOG_WARN("%s: no VRAM for the prefill staging of the cold experts on %s%d (%d x %.1f MiB), they are read in place\n",
            __func__, GGML_CUDA_NAME, device, n_areas, st.size/1048576.0);
        return;
    }
    st.n_areas = n_areas;
    if (st.stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking));
        for (int a = 0; a < stage_state::max_areas; ++a) {
            CUDA_CHECK(cudaEventCreateWithFlags(&st.ev_staged[a],   cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&st.ev_released[a], cudaEventDisableTiming));
        }
    }
    for (int a = 0; a < stage_state::max_areas; ++a) {
        st.area_layer[a] = -1;
        st.released[a]   = false;
    }
    void ** next = st.tables;
    std::vector<const void *> h;
    for (int a = 0; a < n_areas; ++a) {
        char * area = st.buf + (size_t) a*st.size;
        for (auto & it : st.members) {
            for (const stage_member & m : it.second) {
                tiered_tensor * t = m.t;
                h.assign(t->h_table.begin(), t->h_table.end());
                for (int64_t e = 0; e < t->n_expert; ++e) {
                    if (!t->hot(e)) {
                        h[e] = area + m.off + (size_t) (-1 - t->slot[e])*t->expert_bytes;
                    }
                }
                CUDA_CHECK(cudaMemcpy(next, h.data(), t->n_expert*sizeof(void *), cudaMemcpyHostToDevice));
                t->d_table_staged[a] = (const void **) next;
                next += t->n_expert;
            }
        }
    }
    GGML_LOG_INFO("%s: prefill staging of the cold experts on %s%d: %zu layers, %d x %.1f MiB\n", __func__, GGML_CUDA_NAME, device,
        st.layers.size(), n_areas, st.size/1048576.0);
}

} // namespace

const void * const * ggml_cuda_tiered_table_prefill(const ggml_tensor * t, int64_t n_tokens, cudaStream_t stream) {
    const void * const * table = ggml_cuda_tiered_table(t);
    tiered_tensor * info = (tiered_tensor *) tiered_info(t);
    if (info == nullptr || info->cold_size == 0 || info->layer < 0 || !stage_enabled(n_tokens)) {
        return table;
    }
    stage_state & st = g_stage[info->device];
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.ready) {
        stage_setup(st, info->device);
    }
    const int a = st.n_areas > 0 ? st.area_of(info->layer) : -1;
    if (a < 0) {
        return table; // not staged: read in place
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, st.ev_staged[a], 0));
    return (const void * const *) info->d_table_staged[a];
}

void ggml_cuda_tiered_prefill_launched(const ggml_tensor * t, int64_t n_tokens, cudaStream_t stream) {
    tiered_tensor * info = (tiered_tensor *) tiered_info(t);
    if (info == nullptr || !info->is_down || info->layer < 0 || !stage_enabled(n_tokens)) {
        return;
    }
    stage_state & st = g_stage[info->device];
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.ready) {
        stage_setup(st, info->device);
    }
    if (st.n_areas == 0 || st.layers.empty()) {
        return;
    }
    // this layer's area is free once its last expert matmul is done
    const int own = st.area_of(info->layer);
    if (own >= 0) {
        CUDA_CHECK(cudaEventRecord(st.ev_released[own], stream));
        st.released  [own] = true;
        st.area_layer[own] = -1;
    }
    // fill the free areas with the next layers in order (after the last layer: the first ones, the next ubatch)
    int layer = info->layer;
    for (int k = 0; k < st.n_areas; ++k) {
        auto it = std::upper_bound(st.layers.begin(), st.layers.end(), layer);
        layer = it == st.layers.end() ? st.layers.front() : *it;
        if (layer == info->layer) {
            break; // fewer layers than areas
        }
        if (st.area_of(layer) >= 0) {
            continue;
        }
        const int a = st.area_of(-1);
        if (a < 0) {
            break;
        }
        if (st.released[a]) {
            CUDA_CHECK(cudaStreamWaitEvent(st.stream, st.ev_released[a], 0));
        }
        char * area = st.buf + (size_t) a*st.size;
        for (const stage_member & m : st.members[layer]) {
            CUDA_CHECK(cudaMemcpyAsync(area + m.off, m.t->cold_host, m.t->cold_size, cudaMemcpyHostToDevice, st.stream));
        }
        CUDA_CHECK(cudaEventRecord(st.ev_staged[a], st.stream));
        st.area_layer[a] = layer;
    }
}

int ggml_cuda_tiered_is_hot(const ggml_tensor * t, int64_t expert) {
    const tiered_tensor * info = tiered_info(t);
    if (info == nullptr || expert < 0 || expert >= info->n_expert) {
        return -1;
    }
    return info->hot(expert) ? 1 : 0;
}

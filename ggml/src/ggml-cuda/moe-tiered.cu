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

    bool  hot(int64_t e)            const { return slot[e] >= 0; }
    char * host(int64_t e)          const { return cold_host + (size_t) (-1 - slot[e])*expert_bytes; }
    char * dev(int64_t e)           const { return (char *) h_table[e]; }
};

struct tiered_buffer_context {
    int    device;
    void * dev_ptr;
    std::vector<std::unique_ptr<tiered_tensor>> tensors;

    ~tiered_buffer_context() {
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

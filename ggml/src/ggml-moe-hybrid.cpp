#include "ggml-moe-hybrid.h"

#include <atomic>

// Set once while a backend registers, read from the CPU compute threads.
static std::atomic<const ggml_moe_hybrid_api *> g_api{nullptr};

void ggml_moe_hybrid_register(const struct ggml_moe_hybrid_api * api) {
    g_api.store(api, std::memory_order_release);
}

const struct ggml_moe_hybrid_api * ggml_moe_hybrid_get(void) {
    return g_api.load(std::memory_order_acquire);
}

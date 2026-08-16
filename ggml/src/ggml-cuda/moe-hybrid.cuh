#pragma once

// Serves the top output rows of hot MoE experts from VRAM while the CPU keeps the rest.
// Registered into ggml-base from ggml_backend_cuda_reg(); see ggml-moe-hybrid.h.

void ggml_moe_hybrid_cuda_register(void);

// Facade MUL_MAT_ID: a GPU-owned expert node served through the per-device pointer tables.
// Returns false when src0 is not a facade or the origin is not ready; the caller must then treat
// the node as unsupported rather than fall back to reading the facade's raw allocation.
struct ggml_backend_cuda_context;
struct ggml_tensor;
bool ggml_moe_hybrid_facade_run(ggml_backend_cuda_context & ctx,
    const struct ggml_tensor * src0, const struct ggml_tensor * src1,
    const struct ggml_tensor * ids, struct ggml_tensor * dst);

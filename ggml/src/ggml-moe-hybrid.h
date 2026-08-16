#pragma once

// Backend-agnostic hook for sharing one MUL_MAT_ID between the CPU and an accelerator.
//
// The CPU backend owns the op. A backend that registers here may take the top output rows of any
// expert; the CPU computes the rest. Rows are disjoint in dst, so there is no reduction.
//
// This lives in ggml-base because ggml-cpu links only ggml-base and cannot reach the backend
// registry.

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One (slot, token) pair routed to an expert. Built by the CPU kernel while it groups rows.
struct mmid_row_mapping {
    int32_t i1; // slot: which of the n_expert_used picks, indexes dst->ne[1]
    int32_t i2; // token: indexes src1 and dst->ne[2]
};

struct ggml_moe_hybrid_node {
    const struct ggml_tensor * src0; // [ne00, ne01, n_expert] expert weights, host resident
    const struct ggml_tensor * src1; // activations, f32
    const struct ggml_tensor * ids;
    struct ggml_tensor       * dst;

    // routing table, valid until the node ends
    const int64_t                 * row_counts;  // [n_expert] pairs routed to each expert
    const struct mmid_row_mapping * matrix_rows; // [n_expert][rows_stride]
    int64_t                         rows_stride;
};

struct ggml_moe_hybrid_api {
    // True when this backend can serve nodes shaped like this one at all. Cheap; called per node.
    bool (*supports)(const struct ggml_moe_hybrid_node * node);

    // Decide the split. Writes, for every expert a, the output row the CPU must stop at:
    // cpu_row_end[a] == src0->ne[1] leaves the whole expert on the CPU, 0 gives it entirely to the
    // backend. Called by one thread before the barrier that publishes the routing table, so the
    // answer is visible to every thread afterwards. Must be cheap - the other threads are waiting.
    // Returns true if the backend took anything.
    bool (*dispatch)(const struct ggml_moe_hybrid_node * node, int64_t * cpu_row_end);

    // Issue the work decided by dispatch. Called after the barrier, so the CPU threads are already
    // computing their own rows while this runs.
    void (*launch)(const struct ggml_moe_hybrid_node * node);

    // Wait for the launched work and make its dst rows visible to the CPU.
    void (*join)(const struct ggml_moe_hybrid_node * node);

    // Pre-register a host-resident expert tensor as a facade origin, so facade nodes can be
    // served before the CPU path ever dispatched it (the load-time warmup runs decode first).
    void (*note_origin)(const struct ggml_tensor * src0);



};

// Called by a backend at registration time. Passing NULL clears it.
GGML_API void ggml_moe_hybrid_register(const struct ggml_moe_hybrid_api * api);

// NULL when no backend registered, which is the normal case.
GGML_API const struct ggml_moe_hybrid_api * ggml_moe_hybrid_get(void);

#ifdef __cplusplus
}
#endif

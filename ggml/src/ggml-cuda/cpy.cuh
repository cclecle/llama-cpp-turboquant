#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// A run of GGML_OP_CPY nodes (cpys[c]->src[0] -> cpys[c]->src[1]) that share one type (2 or 4 bytes per element, no
// conversion), one shape and one set of strides, and none of which reads what another writes, in one launch: the
// rollback tails of a convolution state are one copy per speculative slot (Qwen3.8-Flash-Next: 5 per gated delta net
// layer). The caller checks the conditions.
#define GGML_CUDA_CPY_MULTI_MAX 8
void ggml_cuda_cpy_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * cpys, int n);

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_qsa_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_qsa_pool_supported(const ggml_tensor * op);

#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
// the one-kernel top-k of each row of a contiguous [ncols, nrows] F32 matrix (ncols <= 65536, else false): the
// selected columns in ascending order, ties taking the lowest columns
bool ggml_cuda_top_k_rows(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream);
#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

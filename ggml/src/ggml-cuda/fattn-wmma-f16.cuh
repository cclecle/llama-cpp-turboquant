#pragma once

#include "common.cuh"

#include <cstdlib>

#if defined(GGML_USE_MUSA)
#define GGML_USE_WMMA_FATTN
#endif // defined(GGML_USE_MUSA)

#if defined(GGML_HIP_ROCWMMA_FATTN)
#if defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#define GGML_USE_WMMA_FATTN
#elif defined(CDNA)
#warning "rocwmma fattn on CDNA is broken on rocwmma v2.0.0, expect degraded performance"
#endif // defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#if defined(RDNA3)
#define GGML_USE_WMMA_FATTN
#endif // defined(RDNA3)
#if defined(RDNA4) && ROCWMMA_VERSION_MAJOR > 1
#define GGML_USE_WMMA_FATTN
#elif defined(RDNA4)
#warning "rocwmma fattn is not supported on RDNA4 on rocwmma < v2.0.0, expect degraded performance"
#endif // defined(RDNA4) && ROCWMMA_VERSION_MAJOR > 1
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

// WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
// Runtime override on top of the compile-time flag. The rocWMMA path preempts the native MMA path
// below it in ggml_cuda_get_best_fattn_kernel, and which one wins is model dependent: measured on
// gfx1201 at pp512 @ d32768, rocWMMA is 42% SLOWER on Mistral-family dense models (Devstral,
// Magistral) but 36% FASTER on Qwen3.6-27B, while MLA models are unaffected. Until the dispatch
// becomes shape-aware this lets it be selected per run/per model instead of per build.
// 1/on/true forces it on (still requires the compile-time support), 0/off/false forces it off.
static bool ggml_cuda_rocwmma_fattn_override(bool & out) {
    static const int val = []() {
        const char * s = getenv("GGML_HIP_ROCWMMA_FATTN");
        if (!s || !*s) {
            return -1;
        }
        if (s[0] == '0' || s[0] == 'f' || s[0] == 'F' || s[0] == 'n' || s[0] == 'N') {
            return 0;
        }
        return 1;
    }();
    if (val < 0) {
        return false;
    }
    out = val != 0;
    return true;
}

static bool ggml_cuda_should_use_wmma_fattn(const int cc) {
    bool forced = false;
    if (ggml_cuda_rocwmma_fattn_override(forced) && !forced) {
        return false; // forcing on can never enable a path that was not compiled in, so only
                      // the disable direction is honoured unconditionally here
    }
#if defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
    return false;
#else
    if ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
        return true;
    } else if (GGML_CUDA_CC_IS_CDNA(cc)){
#if defined(GGML_HIP_ROCWMMA_FATTN) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
        return true;
#else
        return false;
#endif // defined(GGML_HIP_ROCWMMA_FATTN) (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
    } else if (GGML_CUDA_CC_IS_RDNA4(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
        return true;
#else
        return false;
#endif // defined(GGML_HIP_ROCWMMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
    } else {
        return false;
    }
#endif // defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// D=512 decode-only VEC: q8_0 K + f16 V

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_F16);

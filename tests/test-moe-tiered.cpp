// test-moe-tiered: MUL_MAT_ID with the expert weights on the CUDA/HIP <dev>_TIERED buffer type (hot experts in
// VRAM, cold ones in mapped host memory, read through a per-expert address table) must give bit-identical
// results to the same weights on the plain device buffer. Covers the fused gate+up+GLU pattern (mmvq/mmvk at
// few tokens, the fused MMQ at many), the down projection, several quant types and token counts, and three
// placements per case: half the experts hot, none hot (layer not listed), all hot.
//
// usage: test-moe-tiered [device name, default: the first GPU]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int K = 512, N = 256, E = 64, N_USED = 8;

static ggml_backend_buffer_type_t find_tiered(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return nullptr;
    }
    const std::string want = std::string(ggml_backend_dev_name(dev)) + "_TIERED";
    for (ggml_backend_buffer_type_t * p = get_extra(dev); p && *p; ++p) {
        if (want == ggml_backend_buft_name(*p)) {
            return *p;
        }
    }
    return nullptr;
}

// random rows quantized to `type`, [ne0, ne1, ne2]
static std::vector<uint8_t> random_quant(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, std::mt19937 & rng) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> f(ne0*ne1*ne2);
    for (auto & v : f) {
        v = d(rng);
    }
    std::vector<uint8_t> q(ggml_row_size(type, ne0)*ne1*ne2);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, ne1*ne2, ne0, nullptr);
    return q;
}

struct weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

static weights make_weights(ggml_backend_buffer_type_t buft, ggml_type type, int layer,
        const std::vector<uint8_t> & qg, const std::vector<uint8_t> & qu, const std::vector<uint8_t> & qd) {
    weights w;
    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
    w.ctx  = ggml_init(ip);
    w.gate = ggml_new_tensor_3d(w.ctx, type, K, N, E);
    w.up   = ggml_new_tensor_3d(w.ctx, type, K, N, E);
    w.down = ggml_new_tensor_3d(w.ctx, type, N, K, E);
    ggml_format_name(w.gate, "blk.%d.ffn_gate_exps.weight", layer);
    ggml_format_name(w.up,   "blk.%d.ffn_up_exps.weight",   layer);
    ggml_format_name(w.down, "blk.%d.ffn_down_exps.weight", layer);
    w.buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, buft);
    if (!w.buf) {
        fprintf(stderr, "allocation failed on %s\n", ggml_backend_buft_name(buft));
        exit(1);
    }
    ggml_backend_tensor_set(w.gate, qg.data(), 0, qg.size());
    ggml_backend_tensor_set(w.up,   qu.data(), 0, qu.size());
    ggml_backend_tensor_set(w.down, qd.data(), 0, qd.size());
    return w;
}

// the llama.cpp MoE FFN: down(swiglu(gate(x), up(x))) with the same expert ids for all three
static std::vector<float> run(ggml_backend_t backend, const weights & w, const std::vector<float> & x,
        const std::vector<int32_t> & ids, int n_tokens) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * xt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tokens);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n_tokens);
    ggml_set_input(xt);
    ggml_set_input(idt);
    ggml_tensor * g   = ggml_mul_mat_id(ctx, w.gate, xt, idt);
    ggml_tensor * u   = ggml_mul_mat_id(ctx, w.up,   xt, idt);
    ggml_tensor * h   = ggml_swiglu_split(ctx, g, u);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w.down, h, idt);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_tensor_set(xt,  x.data(),   0, x.size()*sizeof(float));
    ggml_backend_tensor_set(idt, ids.data(), 0, ids.size()*sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    std::vector<float> r(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, r.size()*sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    // placements: blk.0 even experts hot, blk.1 not listed (all cold), blk.2 all hot
    const char * hot_path = "test-moe-tiered.hot.txt";
    {
        FILE * f = fopen(hot_path, "w");
        fprintf(f, "# test-moe-tiered\nblk.0");
        for (int e = 0; e < E; e += 2) {
            fprintf(f, " %d", e);
        }
        fprintf(f, "\nblk.2");
        for (int e = 0; e < E; ++e) {
            fprintf(f, " %d", e);
        }
        fprintf(f, "\n");
        fclose(f);
    }
#ifdef _WIN32
    _putenv_s("GGML_CUDA_MOE_HOT_FILE", hot_path);
#else
    setenv("GGML_CUDA_MOE_HOT_FILE", hot_path, 1);
#endif

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    if (argc > 1) {
        dev = ggml_backend_dev_by_name(argv[1]);
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count() && !dev; ++i) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                dev = ggml_backend_dev_get(i);
            }
        }
    }
    if (!dev) {
        printf("no GPU device, skipping\n");
        return 0;
    }
    ggml_backend_buffer_type_t tiered = find_tiered(dev);
    if (!tiered) {
        printf("%s has no _TIERED buffer type, skipping\n", ggml_backend_dev_name(dev));
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t plain = ggml_backend_dev_buffer_type(dev);

    const ggml_type types[] = { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 };
    const int n_tokens_list[] = { 1, 2, 3, 5, 8, 64, 512 };
    std::mt19937 rng(42);
    int n_fail = 0, n_ok = 0;

    for (ggml_type type : types) {
        const auto qg = random_quant(type, K, N, E, rng);
        const auto qu = random_quant(type, K, N, E, rng);
        const auto qd = random_quant(type, N, K, E, rng);
        weights wp = make_weights(plain, type, 0, qg, qu, qd);
        for (int layer = 0; layer < 3; ++layer) {
            weights wt = make_weights(tiered, type, layer, qg, qu, qd);
            for (int n_tokens : n_tokens_list) {
                std::normal_distribution<float> d(0.0f, 1.0f);
                std::vector<float> x((size_t) K*n_tokens);
                for (auto & v : x) {
                    v = d(rng);
                }
                std::vector<int32_t> ids((size_t) N_USED*n_tokens);
                for (int t = 0; t < n_tokens; ++t) {
                    std::vector<int32_t> perm(E);
                    for (int e = 0; e < E; ++e) {
                        perm[e] = e;
                    }
                    std::shuffle(perm.begin(), perm.end(), rng);
                    for (int k = 0; k < N_USED; ++k) {
                        ids[(size_t) t*N_USED + k] = perm[k];
                    }
                }
                const auto rp = run(backend, wp, x, ids, n_tokens);
                const auto rt = run(backend, wt, x, ids, n_tokens);
                const bool same = rp.size() == rt.size() && memcmp(rp.data(), rt.data(), rp.size()*sizeof(float)) == 0;
                bool finite = true;
                for (float v : rt) {
                    finite = finite && std::isfinite(v);
                }
                printf("  %-7s blk.%d (%s) tokens %3d: %s\n", ggml_type_name(type), layer,
                    layer == 0 ? "half hot" : layer == 1 ? "all cold" : "all hot ", n_tokens,
                    same && finite ? "OK" : same ? "FAIL (non-finite)" : "FAIL (differs)");
                (same && finite ? n_ok : n_fail)++;
            }
            ggml_backend_buffer_free(wt.buf);
            ggml_free(wt.ctx);
        }
        ggml_backend_buffer_free(wp.buf);
        ggml_free(wp.ctx);
    }
    printf("%d/%d OK\n", n_ok, n_ok + n_fail);
    ggml_backend_free(backend);
    remove(hot_path);
    return n_fail ? 1 : 0;
}

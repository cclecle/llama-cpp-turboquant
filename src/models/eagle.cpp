#include "models.h"

// EAGLE-1/2 style draft head (e.g. mistralai/Mistral-Medium-3.5-128B-EAGLE).
//
// Reference implementation: vLLM `EagleMistralModel` (vllm/model_executor/models/mistral_eagle.py)
//
//     inputs_embeds = embed_tokens(input_ids)                     # shared with the target
//     hidden_states = fc(cat((inputs_embeds, hidden_states), -1)) # `eagle_linear`, 2*n_embd -> n_embd
//     for layer in layers:                                        # plain decoder layers
//         hidden_states, residual = layer(positions, hidden_states, residual)
//     hidden_states, _ = norm(hidden_states, residual)
//     return hidden_states, hidden_states                         # logits input AND next-step h
//
// Two attention flavours are supported, chosen by hparams:
//   - plain MHA/GQA (wq/wk/wv/wo) - Mistral-Medium-3.5-128B-EAGLE.
//   - MLA latent attention (wq_a/wq_b, wkv_a_mqa/wkv_b, q_a_norm/kv_a_norm) - Mistral-Small-4-119B
//     EAGLE, whose target uses DeepSeek-style latent attention. We run MLA via the *decompressed*
//     MHA path (wkv_b -> full per-head K/V, regular KV cache), which is simplest and, since the
//     draft is only 2 layers, has negligible KV cost. Adapted from models/deepseek2.cpp.
//
// The target model supplies its post-final-norm hidden state for every token via `h` (ubatch->embd).

void llama_model_eagle::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MLA (optional): present when the target uses latent attention (Mistral-Small-4).
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,  hparams.n_lora_q,  false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false);

    // llama-4 attention-temperature tuning (optional), and the YaRN mscale used by MLA.
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE,  hparams.f_attn_temp_scale,       false);
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.n_attn_temp_floor_scale, false);
    hparams.f_attn_temp_offset = 0.0f;
    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        hparams.rope_yarn_log_mul /= 0.1f; // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] cancel the convert-script factor
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const bool is_mla = hparams.n_lora_kv > 0;

    // feature fusion layer: concat(token embedding, target hidden state) -> n_embd
    fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), {2 * n_embd, n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // token_embd and output are optional: EAGLE normally shares both with the target model
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    if (tok_embd == nullptr) {
        LLAMA_LOG_INFO("%s: EAGLE without token_embd - sharing the target model's embeddings\n", __func__);
    }
    if (output == nullptr) {
        LLAMA_LOG_INFO("%s: EAGLE without output - sharing the target model's lm_head\n", __func__);
    }

    // MLA geometry (only used when is_mla); n_embd_head_k/v and n_rot come from LLAMA_LOAD_LOCALS.
    // We run the decompressed-MHA path with full head dims; n_rot is the rope-only slice (qk_rope).
    const int64_t n_head_dim_qk_nope = n_embd_head_k - n_rot;
    const int64_t q_lora_rank        = hparams.n_lora_q;
    const int64_t kv_lora_rank       = hparams.n_lora_kv;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (is_mla) {
            GGML_ASSERT(n_head_dim_qk_nope >= 1);
            if (q_lora_rank > 0) {
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
                layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
                layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
            } else {
                layer.wq            = create_tensor(tn(LLM_TENSOR_ATTN_Q,        "weight", i), {n_embd, n_head * n_embd_head_k}, 0);
            }
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_rot}, 0);
            layer.wkv_b          = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,      "weight", i), {kv_lora_rank, n_head * (n_head_dim_qk_nope + n_embd_head_v)}, 0);
            layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head * n_embd_head_v, n_embd}, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_eagle::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_eagle::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const bool is_mla = hparams.n_lora_kv > 0;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // token embeddings: usually the target model's (the draft ships none of its own)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "EAGLE requires token embeddings (own or from the target model)");
        tok_embd = model_other->tok_embd;
    }

    // inputs: the draft token ids, plus one hidden-state row per token.
    // At memory pos P the input pair is (t_{P+1}, h_P) - same convention as EAGLE3.
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inp_embd, "inp_embd", -1);

    ggml_tensor * inp_h = inp->embd;
    cb(inp_h, "inp_h", -1);

    res->add_input(std::move(inp));

    // feature fusion: fc @ concat(embd, h) - embedding first, matching torch.cat((inputs_embeds, hidden_states), -1)
    cur = ggml_concat(ctx0, inp_embd, inp_h, 0);
    cb(cur, "fc_inp", -1);

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    inpL = cur;

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    // (optional) llama-4 attention-temperature scaling
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_inp_attn_scale();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA geometry (matches load_arch_tensors); n_embd_head_k/v and n_rot are graph-context members.
    const int64_t qk_nope           = n_embd_head_k - n_rot;
    const int64_t kv_lora_rank      = hparams.n_lora_kv;

    // kq scale. For plain attention this is 1/sqrt(head). For MLA we replicate deepseek2's
    // YaRN-aware mscale so the draft's attention matches how the target scales it.
    float kq_scale = 1.0f/sqrtf(float(n_embd_head_k));
    if (is_mla) {
        const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k));
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (!is_mla) {
            // -------- plain MHA/GQA --------
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        } else {
            // -------- MLA via decompressed MHA (adapted from models/deepseek2.cpp) --------
            ggml_tensor * q;
            if (model.layers[il].wq_a) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            }
            cb(q, "q", il);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_rot, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head,
                    ggml_row_size(q->type, qk_nope));

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);

            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_rot, 1, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);

            ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, qk_nope + n_embd_head_v),
                    ggml_row_size(kv->type, qk_nope + n_embd_head_v) * n_head, 0);
            ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                    ggml_row_size(kv->type, qk_nope + n_embd_head_v),
                    ggml_row_size(kv->type, qk_nope + n_embd_head_v) * n_head,
                    ggml_row_size(kv->type, qk_nope));
            Vcur = ggml_cont(ctx0, Vcur);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
            ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);

            if (inp_attn_scale) {
                Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // the post-norm hidden state is BOTH the lm_head input and the h fed to the next draft step
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    // lm_head - usually the target model's (the draft ships none of its own)
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->output != nullptr && "EAGLE requires an output projection (own or from the target model)");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

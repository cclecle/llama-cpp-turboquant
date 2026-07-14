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
// Notes vs EAGLE3 (see eagle3.cpp), which is a *different* architecture:
//   - `fc` fuses the token embedding with ONE target hidden state (2*n_embd in), not three
//     target layers (3*n_embd_tgt in). Because its input depends on the draft token, `fc` lives
//     in the decoder graph and there is no separate encoder pass.
//   - the decoder layers are plain (attention input is n_embd, not 2*n_embd) and layer 0 keeps
//     its input_layernorm.
//   - the recurrent hidden state handed to the next draft step is the POST-final-norm tensor
//     (EAGLE3 feeds back the pre-norm one). Both `t_logits` and `t_h_nextn` read the same tensor.
//
// The target model must supply its post-final-norm hidden state for every token; that is what
// `h` (ubatch->embd) carries here.

void llama_model_eagle::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

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

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_eagle::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_eagle::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

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

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
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

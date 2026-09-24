#include "models.h"
#include "ggml-alloc.h"
#include <fstream>
#include <vector>

void llama_model_gemma4::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer_all - (int32_t)n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // when non_causal is set, the model will use bidirectional attention on SWA layers only, while dense layers will remain causal
    // ref: use_bidirectional_attention == "vision" in HF config
    // note: E2B/E4B are always causal, bypassing this logic
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_ONLY;

    switch (hparams.n_layer()) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp();

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    if (n_embd_per_layer > 0) {
        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, TENSOR_READ_LAZY);
        per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
        per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i),
            {n_embd, n_embd_head * n_head + n_embd_k + n_embd_v}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        if (!layer.wqkv) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, kv_flags);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr;

        // norm
        if (has_expert) {
            layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_gate_up_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);

            if (layer.ffn_gate_up_exps == nullptr) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            }

            layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // per-expert scale will be loaded as down_exps_s at the end of the current switch case
        }

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

void llama_model_gemma4::init_adapter_weights() {
    if (cascade_stages[0].w_merge != nullptr || hparams.n_layer() < 34) {
        return;
    }

    auto * target_layer_tensor = layers[25].ffn_down;
    if (!target_layer_tensor || !target_layer_tensor->buffer) {
        return;
    }

    auto * buft = ggml_backend_buffer_get_type(target_layer_tensor->buffer);
    const size_t ctx_size = ggml_tensor_overhead() * (llama_cascade_config::NUM_STAGES * 2 + 8);
    struct ggml_init_params p = { ctx_size, nullptr, true };
    struct ggml_context * actx = ggml_init(p);

    const int64_t n_embd = hparams.n_embd;

    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        cascade_stages[s].fork_layer  = llama_cascade_config::get_fork_layer(s);
        cascade_stages[s].merge_layer = llama_cascade_config::get_merge_layer(s);
        cascade_stages[s].is_active   = false;

        cascade_stages[s].w_merge   = ggml_new_tensor_2d(actx, GGML_TYPE_F16, n_embd, n_embd);
        ggml_set_name(cascade_stages[s].w_merge, ("w_merge_" + std::to_string(s + 1)).c_str());

        cascade_stages[s].gate_proj = ggml_new_tensor_2d(actx, GGML_TYPE_F16, n_embd, 1);
        ggml_set_name(cascade_stages[s].gate_proj, ("gate_proj_" + std::to_string(s + 1)).c_str());
    }

    cascade_adapter_buf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
    if (!cascade_adapter_buf) {
        fprintf(stderr, "[Cascade Adapter] Failed to allocate backend buffer for adapter tensors\n");
        return;
    }

    // Zero-initialize all adapter tensors
    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        std::vector<uint8_t> zeros_w(ggml_nbytes(cascade_stages[s].w_merge), 0);
        std::vector<uint8_t> zeros_g(ggml_nbytes(cascade_stages[s].gate_proj), 0);
        ggml_backend_tensor_set(cascade_stages[s].w_merge, zeros_w.data(), 0, zeros_w.size());
        ggml_backend_tensor_set(cascade_stages[s].gate_proj, zeros_g.data(), 0, zeros_g.size());
    }

    auto find_weight_file = [](const std::string & fname) -> std::string {
        std::vector<std::string> search_dirs;
        const char * env_dir = std::getenv("LLAMA_CASCADE_DIR");
        if (env_dir && env_dir[0] != '\0') {
            search_dirs.emplace_back(env_dir);
        }
        search_dirs.emplace_back("F:/AI/CascadeLearning/weights");
        search_dirs.emplace_back("F:\\AI\\CascadeLearning\\weights");
        search_dirs.emplace_back(".");

        for (const auto & dir : search_dirs) {
            std::string full_path = dir.empty() ? fname : (dir + "/" + fname);
            std::ifstream f(full_path, std::ios::binary);
            if (f.good()) {
                return full_path;
            }
        }
        return "";
    };

    size_t loaded_count = 0;
    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        const std::string w_fname = "w_merge_" + std::to_string(s + 1) + ".bin";
        const std::string g_fname = "gate_" + std::to_string(s + 1) + ".bin";

        const std::string w_path = find_weight_file(w_fname);
        const std::string g_path = find_weight_file(g_fname);

        if (!w_path.empty()) {
            std::ifstream fw(w_path, std::ios::binary);
            if (fw.is_open()) {
                fw.seekg(0, std::ios::end);
                const size_t file_size = (size_t) fw.tellg();
                fw.seekg(0, std::ios::beg);

                const size_t exp_size = ggml_nbytes(cascade_stages[s].w_merge);
                if (file_size != exp_size) {
                    fprintf(stderr, "\033[1;33m[Cascade Adapter]\033[0m File %s size mismatch (expected %zu bytes, got %zu). Skipping.\n",
                        w_path.c_str(), exp_size, file_size);
                } else {
                    std::vector<char> data(exp_size);
                    fw.read(data.data(), exp_size);
                    if (fw.gcount() == static_cast<std::streamsize>(exp_size)) {
                        ggml_backend_tensor_set(cascade_stages[s].w_merge, data.data(), 0, exp_size);
                        cascade_stages[s].is_active = true;
                        loaded_count++;
                        fprintf(stderr, "\033[1;32m[Cascade Latent Adapter]\033[0m Stage %zu (L%d->L%d) loaded %s (%.2f MB, FP16)\n",
                            s + 1, cascade_stages[s].fork_layer + 1, cascade_stages[s].merge_layer + 1,
                            w_fname.c_str(), exp_size / (1024.0 * 1024.0));
                        fflush(stderr);
                    }
                }
            }
        }

        if (!g_path.empty()) {
            std::ifstream fg(g_path, std::ios::binary);
            if (fg.is_open()) {
                fg.seekg(0, std::ios::end);
                const size_t file_size = (size_t) fg.tellg();
                fg.seekg(0, std::ios::beg);

                const size_t exp_size = ggml_nbytes(cascade_stages[s].gate_proj);
                if (file_size != exp_size) {
                    fprintf(stderr, "\033[1;33m[Cascade Adapter]\033[0m File %s size mismatch (expected %zu bytes, got %zu). Skipping.\n",
                        g_path.c_str(), exp_size, file_size);
                } else {
                    std::vector<char> data(exp_size);
                    fg.read(data.data(), exp_size);
                    if (fg.gcount() == static_cast<std::streamsize>(exp_size)) {
                        ggml_backend_tensor_set(cascade_stages[s].gate_proj, data.data(), 0, exp_size);
                        fprintf(stderr, "\033[1;32m[Cascade Latent Adapter]\033[0m Stage %zu gate active (%.2f KB, FP16)\n",
                            s + 1, exp_size / 1024.0);
                        fflush(stderr);
                    }
                }
            }
        }
    }

    if (loaded_count > 0) {
        fprintf(stderr, "\033[1;36m[Cascade Latent Adapter]\033[0m Successfully activated %zu/%zu lookahead stages\n\n",
            loaded_count, llama_cascade_config::NUM_STAGES);
        fflush(stderr);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(const llm_graph_params & params) const {
    const_cast<llama_model_gemma4 *>(this)->init_adapter_weights();
    return std::make_unique<graph>(*this, params);
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * gemma4_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

llama_model_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    ggml_tensor * alt_streams[llama_cascade_config::NUM_STAGES] = { nullptr };
    const bool is_extracting = (cparams.cb_eval != nullptr);

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        res->t_layer_inp[il] = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * qkv_fused = nullptr;
        ggml_tensor * Qcur;
        if (model.layers[il].wqkv) {
            qkv_fused = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
            cb(qkv_fused, "wqkv", il);
            const int64_t q_dim = n_embd_head * n_head;
            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, n_tokens, qkv_fused->nb[1], 0));
        } else {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
        }
        {
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (qkv_fused) {
                const int64_t q_dim = n_embd_head * n_head;
                const int64_t k_dim = n_embd_head * n_head_kv;
                const int64_t v_dim = n_embd_head * n_head_kv;
                const size_t  esize = ggml_element_size(qkv_fused);
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, k_dim, n_tokens, qkv_fused->nb[1], q_dim * esize));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, v_dim, n_tokens, qkv_fused->nb[1], (q_dim + k_dim) * esize));
            } else {
                Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
                Vcur = model.layers[il].wv
                       ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                       : Kcur; // if v_proj is not present, use Kcur as Vcur
            }
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = gemma4_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // --- Pipelined Lookahead Thought Cascade (7 Stages: L24..L30 -> L26..L32) ---
        // 1. Incubation: pass active streams through layer il's FFN
        for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
            const int32_t fork_l  = llama_cascade_config::get_fork_layer(s);
            const int32_t merge_l = llama_cascade_config::get_merge_layer(s);

            if (alt_streams[s] && il > fork_l && il <= merge_l) {
                ggml_tensor * norm = build_norm(alt_streams[s], model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
                ggml_tensor * ffn  = build_ffn(norm,
                        model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                        nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
                ffn = build_norm(ffn, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
                alt_streams[s] = ggml_add(ctx0, alt_streams[s], ffn);
                cb(alt_streams[s], ("alt_incubated_" + std::to_string(s + 1)).c_str(), il);
            }
        }

        // 2. Fusion: merge thoughts that have reached their target layer
        for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
            const int32_t merge_l = llama_cascade_config::get_merge_layer(s);
            const auto & stage = model.cascade_stages[s];

            // Fusion ONLY happens during normal generation with trained weights, NEVER during extraction
            if (!is_extracting && il == merge_l && alt_streams[s] && stage.is_active && stage.w_merge) {
                // w_merge is FP16, alt_streams[s] is FP32 -> accelerated on Tensor Cores!
                ggml_tensor * alt_proj = ggml_mul_mat(ctx0, stage.w_merge, alt_streams[s]);
                if (stage.gate_proj) {
                    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, stage.gate_proj, alt_streams[s]));
                    alt_proj = ggml_mul(ctx0, alt_proj, gate);
                }
                cur = ggml_add(ctx0, cur, alt_proj);
                cb(cur, ("cascade_fused_" + std::to_string(s + 1)).c_str(), il);
            }
        }

        // 3. Fork: spawn new thought stream from current (already fused) representation
        for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
            const int32_t fork_l = llama_cascade_config::get_fork_layer(s);
            const auto & stage = model.cascade_stages[s];
            if (il == fork_l && (stage.is_active || is_extracting)) {
                alt_streams[s] = ggml_add(ctx0, cur, ggml_scale(ctx0, ggml_sin(ctx0, cur), llama_cascade_config::SIN_SCALE));
                cb(alt_streams[s], ("cascade_fork_" + std::to_string(s + 1)).c_str(), il);
            }
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    // Expose the post-output-norm hidden state (the LM-head input feature) so that
    // MTP draft contexts can read it via llama_get_embeddings_nextn_ith() as the
    // recurrent h input. This matches the reference (transformers/vLLM/SGLang),
    // which feeds the drafter the target's post-final-norm hidden state.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (is_extracting) {
        for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
            if (alt_streams[s]) {
                ggml_build_forward_expand(gf, alt_streams[s]);
            }
        }
    }
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llama_model_gemma4::graph::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows  (ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llama_model_gemma4::graph::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    per_layer_proj = ggml_mul_mat   (ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}

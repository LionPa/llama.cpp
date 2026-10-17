#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct llama_model;
struct llama_hparams;
struct llama_model_params;
struct llm_graph_context;

struct llama_cpt_config {
    static constexpr size_t NUM_HUBS = 5;
    static constexpr size_t NUM_FORKS = 14;
    static constexpr int32_t LOOKAHEAD_SPAN = 3;
    static constexpr float DEFAULT_SIN_SCALE = 0.05f;

    static constexpr int32_t get_hub_merge_layer(size_t hub_idx) {
        return 21 + static_cast<int32_t>(hub_idx) * 3;
    }

    static constexpr int32_t get_hub_target_layer(size_t hub_idx) {
        return get_hub_merge_layer(hub_idx) + LOOKAHEAD_SPAN;
    }

    static constexpr int get_hub_fork_count(size_t hub_idx) {
        return (hub_idx == 0) ? 2 : 3;
    }

    static constexpr int get_hub_fork_layer(size_t hub_idx, size_t branch_idx) {
        if (hub_idx == 0) {
            return (branch_idx == 0) ? 18 : 19;
        }
        return 20 + static_cast<int32_t>(hub_idx - 1) * 3 + static_cast<int32_t>(branch_idx);
    }

    static constexpr int get_fork_layer_by_index(size_t fork_idx) {
        return 18 + static_cast<int32_t>(fork_idx);
    }

    static constexpr int get_fork_hub_index(size_t fork_idx) {
        if (fork_idx < 2) {
            return 0;
        }
        return 1 + static_cast<int>((fork_idx - 2) / 3);
    }

    static constexpr int get_fork_merge_layer(size_t fork_idx) {
        return get_hub_merge_layer(get_fork_hub_index(fork_idx));
    }
};

struct llama_cpt_hub {
    int32_t merge_layer = 0;
    int32_t target_layer = 0;
    int num_forks = 0;
    bool is_active = false;
    struct ggml_tensor * w_merge = nullptr;
    struct ggml_tensor * gate_proj = nullptr;
};

struct llama_cpt_model {
    llama_cpt_hub hubs[llama_cpt_config::NUM_HUBS] = {};
    ggml_backend_buffer_t adapter_buf = nullptr;
    std::string weights_dir;
    float sin_scale = llama_cpt_config::DEFAULT_SIN_SCALE;
    float sin_scales[llama_cpt_config::NUM_FORKS] = {};
    float sin_mult = 1.0f;
    bool has_sin_scales = false;
    int train_port = 0;
    bool is_training = false;
};

struct llama_cpt_context {
    struct ggml_tensor * forks[llama_cpt_config::NUM_FORKS] = {};
    struct ggml_tensor * x_hubs[llama_cpt_config::NUM_HUBS] = {};
    struct ggml_tensor * y_hubs[llama_cpt_config::NUM_HUBS] = {};
    struct ggml_tensor * merge_inps[llama_cpt_config::NUM_HUBS] = {};
};

void llama_cpt_init_weights(
    llama_cpt_model & cpt,
    const llama_hparams & hparams,
    const llama_model_params & mparams,
    struct ggml_tensor * ref_tensor);

void llama_cpt_free_weights(llama_cpt_model & cpt);

struct ggml_tensor * llama_cpt_before_layer(
    llm_graph_context * gctx,
    struct ggml_tensor * inpL,
    int il,
    const llama_model & model,
    llama_cpt_context & cpt_ctx);

struct ggml_tensor * llama_cpt_after_layer(
    llm_graph_context * gctx,
    struct ggml_tensor * cur,
    int il,
    const llama_model & model,
    llama_cpt_context & cpt_ctx,
    struct ggml_tensor * inp_sin);

void llama_cpt_finalize_graph(
    llm_graph_context * gctx,
    llama_cpt_context & cpt_ctx);

bool llama_cpt_stream_batch(
    int train_port,
    int hub_idx,
    const float * x_data,
    const float * y_data,
    int64_t n_tokens,
    int64_t n_embd,
    int num_forks = 1);

#include "llama.h"
#include "llama-cpt.h"
#include "llama-model.h"
#include "llama-graph.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
static void init_winsock_once() {
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
        initialized = true;
    }
}
#endif

static int64_t total_streamed_batches = 0;

static int get_stream_socket(int port) {
    static int cached_port = -1;
    static uintptr_t cached_sock = static_cast<uintptr_t>(-1);

#if defined(_WIN32)
    init_winsock_once();
#endif

    if (cached_sock != static_cast<uintptr_t>(-1) && cached_port == port) {
        return static_cast<int>(cached_sock);
    }

    if (cached_sock != static_cast<uintptr_t>(-1)) {
#if defined(_WIN32)
        closesocket(static_cast<SOCKET>(cached_sock));
#else
        close(static_cast<int>(cached_sock));
#endif
        cached_sock = static_cast<uintptr_t>(-1);
    }

    uintptr_t sock = static_cast<uintptr_t>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (sock == static_cast<uintptr_t>(-1)) {
        return -1;
    }

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(static_cast<SOCKET>(sock), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
#if defined(_WIN32)
        closesocket(static_cast<SOCKET>(sock));
#else
        close(static_cast<int>(sock));
#endif
        return -1;
    }

    cached_sock = sock;
    cached_port = port;
    return static_cast<int>(cached_sock);
}

static void close_stream_socket() {
    int dummy = -1;
    GGML_UNUSED(dummy);
}

static bool send_all(uintptr_t sock, const void * data, size_t total_bytes) {
    const char * ptr = reinterpret_cast<const char *>(data);
    size_t remaining = total_bytes;
    while (remaining > 0) {
        int sent = send(static_cast<SOCKET>(sock), ptr, static_cast<int>(remaining), 0);
        if (sent <= 0) {
            return false;
        }
        remaining -= static_cast<size_t>(sent);
        ptr += sent;
    }
    return true;
}

bool llama_cpt_stream_batch(
    int train_port,
    int hub_idx,
    const float * x_data,
    const float * y_data,
    int64_t n_tokens,
    int64_t n_embd,
    int num_forks) {
    if (train_port <= 0) {
        return false;
    }

    int sock = get_stream_socket(train_port);
    if (sock < 0) {
        return false;
    }

    int32_t header[5] = {
        static_cast<int32_t>(0x43505433),
        static_cast<int32_t>(hub_idx),
        static_cast<int32_t>(n_tokens),
        static_cast<int32_t>(n_embd),
        static_cast<int32_t>(num_forks)
    };

    if (!send_all(static_cast<uintptr_t>(sock), header, sizeof(header))) {
        return false;
    }

    if (hub_idx < 0) {
        return true;
    }

    size_t x_bytes = static_cast<size_t>(num_forks * n_tokens * n_embd) * sizeof(float);
    size_t y_bytes = static_cast<size_t>(n_tokens * n_embd) * sizeof(float);
    if (!send_all(static_cast<uintptr_t>(sock), x_data, x_bytes)) {
        return false;
    }
    if (!send_all(static_cast<uintptr_t>(sock), y_data, y_bytes)) {
        return false;
    }

    total_streamed_batches++;
    return true;
}

void llama_cpt_init_weights(
    llama_cpt_model & cpt,
    const llama_hparams & hparams,
    const llama_model_params & mparams,
    struct ggml_tensor * ref_tensor) {
    if (cpt.hubs[0].w_merge != nullptr || hparams.n_layer() < 34) {
        return;
    }

    std::string path_dir;
    if (mparams.cascade_dir && mparams.cascade_dir[0] != '\0') {
        path_dir = mparams.cascade_dir;
    } else {
        const char * env_dir = std::getenv("LLAMA_CPT_DIR");
        if (!env_dir || env_dir[0] == '\0') {
            env_dir = std::getenv("LLAMA_CASCADE_DIR");
        }
        if (env_dir && env_dir[0] != '\0') {
            path_dir = env_dir;
        }
    }

    cpt.weights_dir = path_dir;
    cpt.sin_scale = mparams.cascade_sin > 0.0f ? mparams.cascade_sin : llama_cpt_config::DEFAULT_SIN_SCALE;
    cpt.sin_mult  = mparams.cascade_sin > 0.0f ? mparams.cascade_sin : 1.0f;

    const char * env_port = std::getenv("LLAMA_CPT_TRAIN_PORT");
    if (env_port && env_port[0] != '\0') {
        cpt.train_port = std::atoi(env_port);
        cpt.is_training = (cpt.train_port > 0);
    }
    if (mparams.cpt_train_port > 0) {
        cpt.train_port = mparams.cpt_train_port;
        cpt.is_training = true;
    }

    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        cpt.hubs[h].merge_layer  = llama_cpt_config::get_hub_merge_layer(h);
        cpt.hubs[h].target_layer = llama_cpt_config::get_hub_target_layer(h);
        cpt.hubs[h].num_forks    = llama_cpt_config::get_hub_fork_count(h);
        cpt.hubs[h].is_active    = false;
    }

    if (path_dir.empty() && cpt.train_port <= 0) {
        return;
    }

    if (!ref_tensor || !ref_tensor->buffer) {
        return;
    }

    auto * buft = ggml_backend_buffer_get_type(ref_tensor->buffer);
    const size_t ctx_size = ggml_tensor_overhead() * (llama_cpt_config::NUM_HUBS * 2 + 8);
    struct ggml_init_params p = { ctx_size, nullptr, true };
    struct ggml_context * actx = ggml_init(p);

    const int64_t n_embd = hparams.n_embd;

    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        cpt.hubs[h].w_merge = ggml_new_tensor_2d(actx, GGML_TYPE_F16, n_embd, n_embd);
        ggml_set_name(cpt.hubs[h].w_merge, ("w_merge_" + std::to_string(h + 1)).c_str());

        cpt.hubs[h].gate_proj = ggml_new_tensor_2d(actx, GGML_TYPE_F16, n_embd, 1);
        ggml_set_name(cpt.hubs[h].gate_proj, ("gate_" + std::to_string(h + 1)).c_str());
    }

    cpt.adapter_buf = ggml_backend_alloc_ctx_tensors_from_buft(actx, buft);
    if (!cpt.adapter_buf) {
        return;
    }

    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        std::vector<uint8_t> zeros_w(ggml_nbytes(cpt.hubs[h].w_merge), 0);
        std::vector<uint8_t> zeros_g(ggml_nbytes(cpt.hubs[h].gate_proj), 0);
        ggml_backend_tensor_set(cpt.hubs[h].w_merge, zeros_w.data(), 0, zeros_w.size());
        ggml_backend_tensor_set(cpt.hubs[h].gate_proj, zeros_g.data(), 0, zeros_g.size());
    }

    if (path_dir.empty()) {
        return;
    }

    size_t loaded_count = 0;
    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        const std::string w_fname = "w_merge_" + std::to_string(h + 1) + ".bin";
        std::string w_path = path_dir + "/" + w_fname;
        std::string g_path = path_dir + "/gate_" + std::to_string(h + 1) + ".bin";
        {
            std::ifstream fg_test(g_path, std::ios::binary);
            if (!fg_test.good()) {
                g_path = path_dir + "/gate_proj_" + std::to_string(h + 1) + ".bin";
            }
        }

        std::ifstream fw(w_path, std::ios::binary);
        if (fw.is_open()) {
            fw.seekg(0, std::ios::end);
            const size_t file_size = static_cast<size_t>(fw.tellg());
            fw.seekg(0, std::ios::beg);

            const size_t exp_size = ggml_nbytes(cpt.hubs[h].w_merge);
            if (file_size == exp_size) {
                std::vector<char> data(exp_size);
                fw.read(data.data(), exp_size);
                if (fw.gcount() == static_cast<std::streamsize>(exp_size)) {
                    ggml_backend_tensor_set(cpt.hubs[h].w_merge, data.data(), 0, exp_size);
                    cpt.hubs[h].is_active = true;
                    loaded_count++;
                    fprintf(stderr, "\033[1;32m[CPT Pyramidal Adapter]\033[0m Hub %zu (L%d, K=%d) loaded %s (%.2f MB, FP16)\n",
                        h + 1, cpt.hubs[h].merge_layer, cpt.hubs[h].num_forks,
                        w_fname.c_str(), exp_size / (1024.0 * 1024.0));
                    fflush(stderr);
                }
            }
        }

        std::ifstream fg(g_path, std::ios::binary);
        if (fg.is_open()) {
            fg.seekg(0, std::ios::end);
            const size_t file_size = static_cast<size_t>(fg.tellg());
            fg.seekg(0, std::ios::beg);

            const size_t exp_size = ggml_nbytes(cpt.hubs[h].gate_proj);
            if (file_size == exp_size) {
                std::vector<char> data(exp_size);
                fg.read(data.data(), exp_size);
                if (fg.gcount() == static_cast<std::streamsize>(exp_size)) {
                    ggml_backend_tensor_set(cpt.hubs[h].gate_proj, data.data(), 0, exp_size);
                }
            }
        }
    }

    std::string s_path = path_dir + "/sin_scale.bin";
    std::ifstream fs(s_path, std::ios::binary);
    if (fs.is_open()) {
        fs.seekg(0, std::ios::end);
        const size_t file_size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        const size_t exp_size = llama_cpt_config::NUM_FORKS * sizeof(float);
        if (file_size == exp_size) {
            fs.read(reinterpret_cast<char *>(cpt.sin_scales), exp_size);
            if (fs.gcount() == static_cast<std::streamsize>(exp_size)) {
                cpt.has_sin_scales = true;
                fprintf(stderr, "\033[1;32m[CPT Pyramidal Adapter]\033[0m Loaded trained sin_scale.bin (14 fork scales, mult = %.3f)\n", cpt.sin_mult);
                fflush(stderr);
            }
        }
    }
    if (!cpt.has_sin_scales) {
        for (size_t f = 0; f < llama_cpt_config::NUM_FORKS; ++f) {
            cpt.sin_scales[f] = llama_cpt_config::DEFAULT_SIN_SCALE;
        }
    }

    if (loaded_count > 0) {
        fprintf(stderr, "\033[1;36m[CPT Pyramidal Adapter]\033[0m Successfully activated %zu/%zu merge hubs (sin = %.3f)\n\n",
            loaded_count, llama_cpt_config::NUM_HUBS, cpt.sin_scale);
        fflush(stderr);
    }
}

void llama_cpt_free_weights(llama_cpt_model & cpt) {
    if (cpt.train_port > 0 && total_streamed_batches > 0) {
        llama_cpt_stream_batch(cpt.train_port, -1, nullptr, nullptr, 0, 0, 0);
        total_streamed_batches = 0;
    }
    if (cpt.adapter_buf) {
        ggml_backend_buffer_free(cpt.adapter_buf);
        cpt.adapter_buf = nullptr;
    }
}

struct ggml_tensor * llama_cpt_before_layer(
    llm_graph_context * gctx,
    struct ggml_tensor * inpL,
    int il,
    const llama_model & model,
    llama_cpt_context & cpt_ctx) {
    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        if (il == llama_cpt_config::get_hub_target_layer(h)) {
            if (cpt_ctx.merge_inps[h] != nullptr && (model.cpt.is_training || model.cpt.train_port > 0 || (gctx->cparams.cb_eval != nullptr))) {
                cpt_ctx.y_hubs[h] = ggml_sub(gctx->ctx0, inpL, cpt_ctx.merge_inps[h]);
                ggml_set_name(cpt_ctx.y_hubs[h], ("cpt_y_hub_" + std::to_string(h + 1)).c_str());
                ggml_build_forward_expand(gctx->gf, cpt_ctx.y_hubs[h]);
                gctx->cb(cpt_ctx.y_hubs[h], ("cpt_y_hub_" + std::to_string(h + 1)).c_str(), il);
            }
        }
    }

    int target_hub = -1;
    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        if (il == llama_cpt_config::get_hub_merge_layer(h)) {
            target_hub = static_cast<int>(h);
            break;
        }
    }

    if (target_hub < 0) {
        return inpL;
    }

    const size_t h = static_cast<size_t>(target_hub);
    cpt_ctx.merge_inps[h] = inpL;

    int fork_count = llama_cpt_config::get_hub_fork_count(h);
    std::vector<ggml_tensor *> normed_forks;
    normed_forks.reserve(fork_count);

    for (int b = 0; b < fork_count; ++b) {
        int fork_layer = llama_cpt_config::get_hub_fork_layer(h, b);
        size_t fork_idx = static_cast<size_t>(fork_layer - 18);
        if (fork_idx < llama_cpt_config::NUM_FORKS && cpt_ctx.forks[fork_idx] != nullptr) {
            ggml_tensor * fnorm = ggml_rms_norm(gctx->ctx0, cpt_ctx.forks[fork_idx], 1e-5f);
            normed_forks.push_back(fnorm);
        }
    }

    if (normed_forks.empty()) {
        return inpL;
    }

    if (model.cpt.is_training || model.cpt.train_port > 0 || (gctx->cparams.cb_eval != nullptr)) {
        ggml_tensor * fork_sum = normed_forks[0];
        for (size_t i = 1; i < normed_forks.size(); ++i) {
            fork_sum = ggml_add(gctx->ctx0, fork_sum, normed_forks[i]);
        }
        float k_scale = 1.0f / sqrtf(static_cast<float>(normed_forks.size()));
        ggml_tensor * x_hub_pre = ggml_scale(gctx->ctx0, fork_sum, k_scale);
        ggml_set_name(x_hub_pre, ("cpt_x_hub_" + std::to_string(h + 1)).c_str());
        ggml_build_forward_expand(gctx->gf, x_hub_pre);
        cpt_ctx.x_hubs[h] = x_hub_pre;
        gctx->cb(x_hub_pre, ("cpt_x_hub_" + std::to_string(h + 1)).c_str(), il);
    }

    const auto & hub = model.cpt.hubs[h];
    if (hub.is_active && hub.w_merge != nullptr && !model.cpt.is_training) {
        ggml_tensor * fork_sum = normed_forks[0];
        for (size_t i = 1; i < normed_forks.size(); ++i) {
            fork_sum = ggml_add(gctx->ctx0, fork_sum, normed_forks[i]);
        }
        float k_scale = 1.0f / sqrtf(static_cast<float>(normed_forks.size()));
        ggml_tensor * x_hub = ggml_scale(gctx->ctx0, fork_sum, k_scale);

        ggml_tensor * delta = ggml_mul_mat(gctx->ctx0, hub.w_merge, x_hub);
        if (hub.gate_proj != nullptr) {
            ggml_tensor * gate_logits = ggml_mul_mat(gctx->ctx0, hub.gate_proj, x_hub);
            ggml_tensor * gate = ggml_sigmoid(gctx->ctx0, gate_logits);
            delta = ggml_mul(gctx->ctx0, delta, gate);
        }
        inpL = ggml_add(gctx->ctx0, inpL, delta);
        gctx->cb(inpL, ("cpt_fused_hub_" + std::to_string(h + 1)).c_str(), il);
    }

    return inpL;
}

struct ggml_tensor * llama_cpt_after_layer(
    llm_graph_context * gctx,
    struct ggml_tensor * cur,
    int il,
    const llama_model & model,
    llama_cpt_context & cpt_ctx,
    struct ggml_tensor * inp_sin) {

    for (size_t f = 0; f < llama_cpt_config::NUM_FORKS; ++f) {
        int fork_layer = llama_cpt_config::get_fork_layer_by_index(f);
        int merge_layer = llama_cpt_config::get_fork_merge_layer(f);

        if (cpt_ctx.forks[f] != nullptr && il > fork_layer && il < merge_layer) {
            ggml_tensor * norm = gctx->build_norm(cpt_ctx.forks[f], model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * ffn  = gctx->build_ffn(norm,
                model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
            ffn = gctx->build_norm(ffn, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
            cpt_ctx.forks[f] = ggml_add(gctx->ctx0, cpt_ctx.forks[f], ffn);
            gctx->cb(cpt_ctx.forks[f], ("cpt_incubated_f" + std::to_string(f + 1)).c_str(), il);
        }
    }

    if (il >= 18 && il <= 31) {
        size_t f = static_cast<size_t>(il - 18);
        size_t h = static_cast<size_t>(llama_cpt_config::get_fork_hub_index(f));
        const auto & hub = model.cpt.hubs[h];

        if (hub.is_active || model.cpt.is_training || (model.cpt.train_port > 0) || (gctx->cparams.cb_eval != nullptr)) {
            const float base_s = model.cpt.has_sin_scales ? model.cpt.sin_scales[f] : model.cpt.sin_scale;
            ggml_tensor * scaled_sin = inp_sin
                ? ggml_scale(gctx->ctx0, ggml_mul(gctx->ctx0, ggml_sin(gctx->ctx0, cur), inp_sin), fabsf(base_s))
                : ggml_scale(gctx->ctx0, ggml_sin(gctx->ctx0, cur), fabsf(base_s) * model.cpt.sin_mult);
            cpt_ctx.forks[f] = ggml_add(gctx->ctx0, cur, scaled_sin);
            gctx->cb(cpt_ctx.forks[f], ("cpt_fork_f" + std::to_string(f + 1)).c_str(), il);
        }
    }

    return cur;
}

void llama_cpt_finalize_graph(
    llm_graph_context * gctx,
    llama_cpt_context & cpt_ctx) {
    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        if (cpt_ctx.x_hubs[h] != nullptr) {
            ggml_build_forward_expand(gctx->gf, cpt_ctx.x_hubs[h]);
        }
        if (cpt_ctx.y_hubs[h] != nullptr) {
            ggml_build_forward_expand(gctx->gf, cpt_ctx.y_hubs[h]);
        }
    }
}

LLAMA_API bool llama_cpt_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!t || !t->name) {
        return false;
    }

    bool is_x = (strncmp(t->name, "cpt_x_hub_", 10) == 0);
    bool is_y = (strncmp(t->name, "cpt_y_hub_", 10) == 0);

    if (!is_x && !is_y) {
        return false;
    }

    if (ask) {
        return true;
    }

    int hub_num = t->name[10] - '1';
    if (hub_num < 0 || hub_num >= static_cast<int>(llama_cpt_config::NUM_HUBS)) {
        return true;
    }

    static std::vector<float> x_buf[llama_cpt_config::NUM_HUBS];
    static std::vector<float> y_buf[llama_cpt_config::NUM_HUBS];
    static bool has_x[llama_cpt_config::NUM_HUBS] = {};
    static bool has_y[llama_cpt_config::NUM_HUBS] = {};
    static int64_t last_n_tokens[llama_cpt_config::NUM_HUBS] = {};
    static int64_t last_n_embd[llama_cpt_config::NUM_HUBS] = {};

    size_t h = static_cast<size_t>(hub_num);
    int fork_count = llama_cpt_config::get_hub_fork_count(h);

    if (is_x) {
        int64_t n_embd = t->ne[0];
        int64_t total_tok = t->ne[1];
        int64_t n_tokens = total_tok / fork_count;
        size_t n_elems = static_cast<size_t>(n_embd * total_tok);

        x_buf[h].resize(n_elems);
        ggml_backend_tensor_get(t, x_buf[h].data(), 0, n_elems * sizeof(float));
        has_x[h] = true;
        last_n_tokens[h] = n_tokens;
        last_n_embd[h] = n_embd;
    } else if (is_y) {
        int64_t n_embd = t->ne[0];
        int64_t n_tokens = t->ne[1];
        size_t n_elems = static_cast<size_t>(n_embd * n_tokens);

        y_buf[h].resize(n_elems);
        ggml_backend_tensor_get(t, y_buf[h].data(), 0, n_elems * sizeof(float));
        has_y[h] = true;
    }

    if (has_x[h] && has_y[h]) {
        int port = static_cast<int>(reinterpret_cast<intptr_t>(user_data));
        if (port > 0) {
            llama_cpt_stream_batch(port, hub_num, x_buf[h].data(), y_buf[h].data(), last_n_tokens[h], last_n_embd[h], fork_count);
        }
        has_x[h] = false;
        has_y[h] = false;
    }

    return true;
}

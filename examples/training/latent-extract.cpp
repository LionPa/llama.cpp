#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

struct stage_collector {
    std::ofstream out_file;
    std::vector<float> buf_main;
    std::vector<float> buf_alt;
    std::vector<float> buf_target;
    int64_t n_tokens_curr = 0;
    bool has_main = false;
    bool has_alt = false;
    bool has_target = false;
    int64_t total_tokens = 0;

    void check_and_flush(uint32_t dim) {
        if (has_main && has_alt && has_target) {
            const uint32_t n_tok = (uint32_t) n_tokens_curr;
            out_file.write(reinterpret_cast<const char *>(&n_tok), sizeof(n_tok));
            out_file.write(reinterpret_cast<const char *>(&dim),   sizeof(dim));
            out_file.write(reinterpret_cast<const char *>(buf_main.data()),   n_tok * dim * sizeof(float));
            out_file.write(reinterpret_cast<const char *>(buf_alt.data()),    n_tok * dim * sizeof(float));
            out_file.write(reinterpret_cast<const char *>(buf_target.data()), n_tok * dim * sizeof(float));
            total_tokens += n_tok;
            has_main   = false;
            has_alt    = false;
            has_target = false;
        }
    }
};

struct cascade_collector {
    stage_collector s1;
    stage_collector s2;
};

static bool latent_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }
    if (!t || !t->name || !user_data) {
        return true;
    }

    auto * col = (cascade_collector *) user_data;

    // --- STAGE 1 (Layers 24-26) ---
    if (std::strcmp(t->name, "l_out-25") == 0) {
        const size_t n_bytes = ggml_nbytes(t);
        col->s1.buf_main.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s1.buf_main.data(), 0, n_bytes);
        col->s1.has_main = true;
        col->s1.n_tokens_curr = t->ne[1];
    } else if (std::strcmp(t->name, "alt1_incubated-25") == 0) {
        const size_t n_bytes = ggml_nbytes(t);
        col->s1.buf_alt.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s1.buf_alt.data(), 0, n_bytes);
        col->s1.has_alt = true;
    } else if (std::strcmp(t->name, "l_out-28") == 0) {
        const size_t n_bytes = ggml_nbytes(t);
        col->s1.buf_target.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s1.buf_target.data(), 0, n_bytes);
        col->s1.has_target = true;

        // Stage 2 main input is also at layer 29 (index 28)
        col->s2.buf_main.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s2.buf_main.data(), 0, n_bytes);
        col->s2.has_main = true;
        col->s2.n_tokens_curr = t->ne[1];
    }

    // --- STAGE 2 (Layers 27-29) ---
    if (std::strcmp(t->name, "alt2_incubated-28") == 0) {
        const size_t n_bytes = ggml_nbytes(t);
        col->s2.buf_alt.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s2.buf_alt.data(), 0, n_bytes);
        col->s2.has_alt = true;
    } else if (std::strcmp(t->name, "l_out-31") == 0) {
        const size_t n_bytes = ggml_nbytes(t);
        col->s2.buf_target.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->s2.buf_target.data(), 0, n_bytes);
        col->s2.has_target = true;
    }

    col->s1.check_and_flush(3840);
    col->s2.check_and_flush(3840);

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    cascade_collector collector;
    collector.s1.out_file.open("latents_stage1.bin", std::ios::binary);
    collector.s2.out_file.open("latents_stage2.bin", std::ios::binary);

    if (!collector.s1.out_file.is_open() || !collector.s2.out_file.is_open()) {
        LOG_ERR("Failed to open output files\n");
        return 1;
    }

    params.cb_eval           = latent_cb_eval;
    params.cb_eval_user_data = &collector;
    params.warmup            = false;
    params.n_ubatch          = 2048;
    params.n_batch           = 2048;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to load model or context\n");
        return 1;
    }

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    std::string dataset_file = "rich_train_dataset.txt";
    std::ifstream file(dataset_file);
    if (!file.is_open()) {
        dataset_file = "F:/AI/PatchedLlama/rich_train_dataset.txt";
        file.open(dataset_file);
    }
    if (!file.is_open()) {
        LOG_ERR("Failed to open dataset file: %s\n", dataset_file.c_str());
        return 1;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<std::string> problems;
    std::string delimiter = "\n---\n";
    size_t pos = 0;
    while ((pos = content.find(delimiter)) != std::string::npos) {
        std::string token = content.substr(0, pos);
        if (!token.empty()) {
            problems.push_back(token);
        }
        content.erase(0, pos + delimiter.length());
    }
    if (!content.empty()) {
        problems.push_back(content);
    }

    LOG_INF("Loaded %zu items from %s. Extracting Cascade latents on GPU...\n", problems.size(), dataset_file.c_str());

    auto * mem = llama_get_memory(ctx);

    for (size_t i = 0; i < problems.size(); ++i) {
        llama_memory_clear(mem, true);

        std::vector<llama_token> tokens = common_tokenize(ctx, problems[i], true, true);
        if (tokens.empty()) {
            continue;
        }
        size_t max_tok = std::min((size_t) llama_n_ctx(ctx), (size_t) llama_n_batch(ctx));
        if (tokens.size() > max_tok) {
            tokens.resize(max_tok);
        }

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            LOG_ERR("Failed to decode item %zu\n", i + 1);
            break;
        }

        LOG_INF("Extracted item %3zu/%3zu | tokens: %zu | Stage 1: %lld | Stage 2: %lld\n",
                i + 1, problems.size(), tokens.size(),
                (long long) collector.s1.total_tokens, (long long) collector.s2.total_tokens);
    }

    collector.s1.out_file.close();
    collector.s2.out_file.close();

    LOG_INF("Saved Stage 1 latents to latents_stage1.bin (%lld tokens)\n", (long long) collector.s1.total_tokens);
    LOG_INF("Saved Stage 2 latents to latents_stage2.bin (%lld tokens)\n", (long long) collector.s2.total_tokens);

    llama_backend_free();
    return 0;
}

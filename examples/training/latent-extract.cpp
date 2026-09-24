#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"

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
    stage_collector stages[llama_cascade_config::NUM_STAGES];
};

static bool latent_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }
    if (!t || !t->name || !user_data) {
        return true;
    }

    auto * col = (cascade_collector *) user_data;
    const std::string name(t->name);

    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        const int32_t merge_l  = llama_cascade_config::get_merge_layer(s);
        const int32_t target_l = merge_l + 3;

        const std::string expected_alt    = "alt_incubated_" + std::to_string(s + 1) + "-" + std::to_string(merge_l);
        const std::string expected_main   = "l_out-" + std::to_string(merge_l);
        const std::string expected_target = "l_out-" + std::to_string(target_l);

        if (name == expected_alt) {
            const size_t n_bytes = ggml_nbytes(t);
            col->stages[s].buf_alt.resize(n_bytes / sizeof(float));
            ggml_backend_tensor_get(t, col->stages[s].buf_alt.data(), 0, n_bytes);
            col->stages[s].has_alt = true;
        } else if (name == expected_main) {
            const size_t n_bytes = ggml_nbytes(t);
            col->stages[s].buf_main.resize(n_bytes / sizeof(float));
            ggml_backend_tensor_get(t, col->stages[s].buf_main.data(), 0, n_bytes);
            col->stages[s].has_main = true;
            col->stages[s].n_tokens_curr = t->ne[1];
        } else if (name == expected_target) {
            const size_t n_bytes = ggml_nbytes(t);
            col->stages[s].buf_target.resize(n_bytes / sizeof(float));
            ggml_backend_tensor_get(t, col->stages[s].buf_target.data(), 0, n_bytes);
            col->stages[s].has_target = true;
        }

        col->stages[s].check_and_flush(3840);
    }

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

    std::string out_dir = "F:/AI/CascadeLearning/latents";
    {
        std::ofstream test(out_dir + "/.probe");
        if (!test.good()) {
            out_dir = ".";
        } else {
            test.close();
            std::remove((out_dir + "/.probe").c_str());
        }
    }

    cascade_collector collector;
    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        std::string path = out_dir + "/latents_stage" + std::to_string(s + 1) + ".bin";
        collector.stages[s].out_file.open(path, std::ios::binary);
        if (!collector.stages[s].out_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", path.c_str());
            return 1;
        }
    }

    params.cb_eval           = latent_cb_eval;
    params.cb_eval_user_data = &collector;
    params.warmup            = false;
    if (params.n_ctx == 0) {
        params.n_ctx = 8192;
    }
    if (params.n_batch < params.n_ctx) {
        params.n_batch = params.n_ctx;
    }
    if (params.n_ubatch < 2048) {
        params.n_ubatch = 2048;
    }

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

    std::string dataset_file = "F:/AI/CascadeLearning/rich_train_dataset.txt";
    std::ifstream file(dataset_file);
    if (!file.is_open()) {
        dataset_file = "rich_train_dataset.txt";
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

    LOG_INF("Loaded %zu items from %s. Extracting %zu Cascade stages on GPU...\n",
            problems.size(), dataset_file.c_str(), llama_cascade_config::NUM_STAGES);

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

        LOG_INF("Extracted item %3zu/%3zu | tokens: %zu | S1: %lld | S2: %lld | S3: %lld | S4: %lld | S5: %lld | S6: %lld | S7: %lld\n",
                i + 1, problems.size(), tokens.size(),
                (long long) collector.stages[0].total_tokens,
                (long long) collector.stages[1].total_tokens,
                (long long) collector.stages[2].total_tokens,
                (long long) collector.stages[3].total_tokens,
                (long long) collector.stages[4].total_tokens,
                (long long) collector.stages[5].total_tokens,
                (long long) collector.stages[6].total_tokens);
    }

    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        collector.stages[s].out_file.close();
        LOG_INF("Saved Stage %zu latents to %s/latents_stage%zu.bin (%lld tokens)\n",
                s + 1, out_dir.c_str(), s + 1, (long long) collector.stages[s].total_tokens);
    }

    llama_backend_free();
    return 0;
}

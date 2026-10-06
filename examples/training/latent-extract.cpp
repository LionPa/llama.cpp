#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
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

            const size_t total = (size_t) n_tok * dim;
            std::vector<ggml_fp16_t> fp16_buf(total);

            for (size_t i = 0; i < total; ++i) {
                fp16_buf[i] = ggml_fp32_to_fp16(buf_main[i]);
            }
            out_file.write(reinterpret_cast<const char *>(fp16_buf.data()), total * sizeof(ggml_fp16_t));

            for (size_t i = 0; i < total; ++i) {
                fp16_buf[i] = ggml_fp32_to_fp16(buf_alt[i]);
            }
            out_file.write(reinterpret_cast<const char *>(fp16_buf.data()), total * sizeof(ggml_fp16_t));

            for (size_t i = 0; i < total; ++i) {
                fp16_buf[i] = ggml_fp32_to_fp16(buf_target[i]);
            }
            out_file.write(reinterpret_cast<const char *>(fp16_buf.data()), total * sizeof(ggml_fp16_t));

            total_tokens += n_tok;
            has_main   = false;
            has_alt    = false;
            has_target = false;
        }
    }
};

struct cascade_collector {
    stage_collector stages[llama_cascade_config::NUM_STAGES];
    const llama_model * model = nullptr;
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
    const int32_t n_layer = col->model ? llama_model_n_layer(col->model) : 0;
    const uint32_t dim    = col->model ? (uint32_t) llama_model_n_embd(col->model) : (uint32_t) t->ne[0];

    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        const int32_t merge_l  = llama_cascade_config::get_merge_layer(s, n_layer);
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

        col->stages[s].check_and_flush(dim);
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    std::string out_dir = "latents";
    std::string dataset_file = "";
    int64_t max_tokens = 100000;

    std::vector<char *> clean_argv;
    clean_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_file = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoll(argv[++i]);
        } else {
            clean_argv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) clean_argv.size(), clean_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    std::filesystem::create_directories(out_dir);

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

    collector.model = model;

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    if (dataset_file.empty()) {
        LOG_ERR("Dataset file not specified. Use --dataset <path>\n");
        return 1;
    }

    std::ifstream file(dataset_file);
    if (!file.is_open()) {
        LOG_ERR("Failed to open dataset file: %s\n", dataset_file.c_str());
        return 1;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<std::string> problems;

    if (content.find("<|turn>user") != std::string::npos) {
        const std::string tag = "<|turn>user";
        size_t start = 0;
        while ((start = content.find(tag, start)) != std::string::npos) {
            size_t next = content.find(tag, start + tag.length());
            std::string item = (next == std::string::npos) ?
                content.substr(start) : content.substr(start, next - start);
            while (!item.empty() && (item.back() == '\r' || item.back() == '\n' || item.back() == ' ')) {
                item.pop_back();
            }
            if (!item.empty()) {
                problems.push_back(item);
            }
            if (next == std::string::npos) {
                break;
            }
            start = next;
        }
    } else if (content.find("<|im_start|>") != std::string::npos) {
        const std::string tag = "<|im_start|>";
        size_t start = 0;
        while ((start = content.find(tag, start)) != std::string::npos) {
            size_t next = content.find(tag, start + tag.length());
            std::string item = (next == std::string::npos) ?
                content.substr(start) : content.substr(start, next - start);
            while (!item.empty() && (item.back() == '\r' || item.back() == '\n' || item.back() == ' ')) {
                item.pop_back();
            }
            if (!item.empty()) {
                problems.push_back(item);
            }
            if (next == std::string::npos) {
                break;
            }
            start = next;
        }
    } else {
        std::string delim = "\n---\n";
        if (content.find(delim) == std::string::npos) {
            delim = content.find("\r\n---\r\n") != std::string::npos ? "\r\n---\r\n" : "---";
        }
        size_t start = 0;
        while (start < content.size()) {
            size_t pos = content.find(delim, start);
            std::string item = (pos == std::string::npos) ?
                content.substr(start) : content.substr(start, pos - start);
            while (!item.empty() && (item.front() == '\r' || item.front() == '\n' || item.front() == ' ')) {
                item.erase(item.begin());
            }
            while (!item.empty() && (item.back() == '\r' || item.back() == '\n' || item.back() == ' ')) {
                item.pop_back();
            }
            if (!item.empty()) {
                problems.push_back(item);
            }
            if (pos == std::string::npos) {
                break;
            }
            start = pos + delim.length();
        }
    }

    LOG_INF("Loaded %zu items from %s. Extracting %zu Cascade stages on GPU...\n",
            problems.size(), dataset_file.c_str(), llama_cascade_config::NUM_STAGES);

    auto * mem = llama_get_memory(ctx);

    for (size_t i = 0; i < problems.size(); ++i) {
        if (max_tokens > 0 && collector.stages[0].total_tokens >= max_tokens) {
            LOG_INF("\n>> Reached target limit of %lld tokens. Stopping extraction.\n", (long long) max_tokens);
            break;
        }

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

        LOG_INF("Extracted item %3zu/%3zu | tokens: %zu | S1: %lld | S7: %lld\n",
                i + 1, problems.size(), tokens.size(),
                (long long) collector.stages[0].total_tokens,
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

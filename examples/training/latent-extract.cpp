#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "llama-cpt.h"

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

struct hub_collector {
    std::ofstream out_file;
    std::vector<float> buf_x;
    std::vector<float> buf_y;
    int64_t n_tokens_curr = 0;
    bool has_x = false;
    bool has_y = false;
    int64_t total_tokens = 0;

    void check_and_flush(uint32_t dim) {
        if (has_x && has_y) {
            const uint32_t n_tok = (uint32_t) n_tokens_curr;
            out_file.write(reinterpret_cast<const char *>(&n_tok), sizeof(n_tok));
            out_file.write(reinterpret_cast<const char *>(&dim),   sizeof(dim));

            const size_t total = (size_t) n_tok * dim;
            std::vector<ggml_fp16_t> fp16_buf(total);

            for (size_t i = 0; i < total; ++i) {
                fp16_buf[i] = ggml_fp32_to_fp16(buf_x[i]);
            }
            out_file.write(reinterpret_cast<const char *>(fp16_buf.data()), total * sizeof(ggml_fp16_t));

            for (size_t i = 0; i < total; ++i) {
                fp16_buf[i] = ggml_fp32_to_fp16(buf_y[i]);
            }
            out_file.write(reinterpret_cast<const char *>(fp16_buf.data()), total * sizeof(ggml_fp16_t));

            total_tokens += n_tok;
            has_x = false;
            has_y = false;
        }
    }
};

struct cpt_collector {
    hub_collector hubs[llama_cpt_config::NUM_HUBS];
    const llama_model * model = nullptr;
};

static bool latent_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!t || !t->name || !user_data) {
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

    auto * col = (cpt_collector *) user_data;
    const std::string name(t->name);
    const uint32_t dim = col->model ? (uint32_t) llama_model_n_embd(col->model) : (uint32_t) t->ne[0];

    int hub_num = t->name[10] - '1';
    if (hub_num < 0 || hub_num >= static_cast<int>(llama_cpt_config::NUM_HUBS)) {
        return true;
    }

    size_t h = static_cast<size_t>(hub_num);

    if (is_x) {
        const size_t n_bytes = ggml_nbytes(t);
        col->hubs[h].buf_x.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->hubs[h].buf_x.data(), 0, n_bytes);
        col->hubs[h].has_x = true;
        col->hubs[h].n_tokens_curr = t->ne[1];
    } else if (is_y) {
        const size_t n_bytes = ggml_nbytes(t);
        col->hubs[h].buf_y.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, col->hubs[h].buf_y.data(), 0, n_bytes);
        col->hubs[h].has_y = true;
    }

    col->hubs[h].check_and_flush(dim);

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

    cpt_collector collector;
    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        std::string path = out_dir + "/latents_hub" + std::to_string(h + 1) + ".bin";
        collector.hubs[h].out_file.open(path, std::ios::binary);
        if (!collector.hubs[h].out_file.is_open()) {
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

    LOG_INF("Loaded %zu items from %s. Extracting %zu CPT hubs on GPU...\n",
            problems.size(), dataset_file.c_str(), llama_cpt_config::NUM_HUBS);

    collector.model = model;
    auto * mem = llama_get_memory(ctx);

    for (size_t i = 0; i < problems.size(); ++i) {
        if (max_tokens > 0 && collector.hubs[0].total_tokens >= max_tokens) {
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

        LOG_INF("Extracted item %3zu/%3zu | tokens: %zu | H1: %lld | H5: %lld\n",
                i + 1, problems.size(), tokens.size(),
                (long long) collector.hubs[0].total_tokens,
                (long long) collector.hubs[4].total_tokens);
    }

    for (size_t h = 0; h < llama_cpt_config::NUM_HUBS; ++h) {
        collector.hubs[h].out_file.close();
        LOG_INF("Saved Hub %zu latents to %s/latents_hub%zu.bin (%lld tokens)\n",
                h + 1, out_dir.c_str(), h + 1, (long long) collector.hubs[h].total_tokens);
    }

    llama_backend_free();
    return 0;
}

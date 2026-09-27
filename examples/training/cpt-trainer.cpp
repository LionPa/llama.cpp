#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

struct stage_data {
    std::vector<float> x;
    std::vector<float> y;
    int64_t total_tokens = 0;
};

struct extraction_stage {
    std::vector<float> buf_main;
    std::vector<float> buf_alt;
    std::vector<float> buf_target;
    int64_t n_tokens_curr = 0;
    bool has_main = false;
    bool has_alt = false;
    bool has_target = false;
};

struct trainer_collector {
    extraction_stage stages[llama_cascade_config::NUM_STAGES];
    stage_data data[llama_cascade_config::NUM_STAGES];
    const llama_model * model = nullptr;
    float gamma = 0.08f;
    uint32_t dim = 0;

    void check_and_accumulate(size_t s) {
        if (stages[s].has_main && stages[s].has_alt && stages[s].has_target) {
            const int64_t n_tok = stages[s].n_tokens_curr;
            const size_t total_floats = (size_t) n_tok * dim;

            const size_t cur_size = data[s].x.size();
            data[s].x.resize(cur_size + total_floats);
            data[s].y.resize(cur_size + total_floats);

            std::memcpy(data[s].x.data() + cur_size, stages[s].buf_alt.data(), total_floats * sizeof(float));

            const float * m_ptr = stages[s].buf_main.data();
            const float * t_ptr = stages[s].buf_target.data();
            float * y_dst = data[s].y.data() + cur_size;

            for (size_t i = 0; i < total_floats; ++i) {
                y_dst[i] = (t_ptr[i] - m_ptr[i]) * gamma;
            }

            data[s].total_tokens += n_tok;
            stages[s].has_main = false;
            stages[s].has_alt = false;
            stages[s].has_target = false;
        }
    }
};

static bool trainer_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }
    if (!t || !t->name || !user_data) {
        return true;
    }

    auto * col = (trainer_collector *) user_data;
    const std::string name(t->name);
    const int32_t n_layer = col->model ? llama_model_n_layer(col->model) : 0;

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

        col->check_and_accumulate(s);
    }

    return true;
}

static bool load_stage_data_from_file(const std::string & path, uint32_t dim, float gamma, stage_data & out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    const size_t file_size = (size_t) file.tellg();
    file.seekg(0, std::ios::beg);

    size_t bytes_read = 0;
    while (bytes_read < file_size) {
        uint32_t n_tok = 0;
        uint32_t file_dim = 0;
        file.read(reinterpret_cast<char *>(&n_tok), sizeof(n_tok));
        file.read(reinterpret_cast<char *>(&file_dim), sizeof(file_dim));
        bytes_read += sizeof(n_tok) + sizeof(file_dim);

        if (file_dim != dim) {
            return false;
        }

        const size_t chunk_floats = (size_t) n_tok * dim;
        const size_t chunk_bytes  = chunk_floats * sizeof(float);

        std::vector<float> buf_m(chunk_floats);
        std::vector<float> buf_a(chunk_floats);
        std::vector<float> buf_t(chunk_floats);

        file.read(reinterpret_cast<char *>(buf_m.data()), chunk_bytes);
        file.read(reinterpret_cast<char *>(buf_a.data()), chunk_bytes);
        file.read(reinterpret_cast<char *>(buf_t.data()), chunk_bytes);
        bytes_read += 3 * chunk_bytes;

        const size_t cur_sz = out.x.size();
        out.x.resize(cur_sz + chunk_floats);
        out.y.resize(cur_sz + chunk_floats);

        std::memcpy(out.x.data() + cur_sz, buf_a.data(), chunk_bytes);
        for (size_t i = 0; i < chunk_floats; ++i) {
            out.y[cur_sz + i] = (buf_t[i] - buf_m[i]) * gamma;
        }

        out.total_tokens += n_tok;
    }

    return true;
}

static void train_single_stage(
        size_t stage_num,
        const stage_data & sdata,
        uint32_t dim,
        const std::string & out_w,
        const std::string & out_g,
        int epochs,
        int batch_size,
        float lr0,
        float lr_min,
        float weight_decay,
        float grad_clip_norm) {

    const int64_t n_tokens = sdata.total_tokens;
    if (n_tokens < batch_size) {
        LOG_ERR("Stage %zu: not enough tokens (%lld < %d)\n", stage_num, (long long) n_tokens, batch_size);
        return;
    }

    const size_t w_size = (size_t) dim * dim;
    const size_t g_size = (size_t) dim;

    std::vector<float> w_merge(w_size, 0.0f);
    std::vector<float> gate_proj(g_size, 0.0f);

    std::vector<float> m_w(w_size, 0.0f);
    std::vector<float> v_w(w_size, 0.0f);
    std::vector<float> m_g(g_size, 0.0f);
    std::vector<float> v_g(g_size, 0.0f);

    std::vector<float> grad_w(w_size, 0.0f);
    std::vector<float> grad_g(g_size, 0.0f);

    const int64_t n_batches = n_tokens / batch_size;
    std::vector<int64_t> indices(n_tokens);
    for (int64_t i = 0; i < n_tokens; ++i) {
        indices[i] = i;
    }

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps   = 1e-8f;
    int64_t opt_step  = 0;

    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (int64_t i = n_tokens - 1; i > 0; --i) {
            const int64_t j = (int64_t) (rand() % (i + 1));
            std::swap(indices[i], indices[j]);
        }

        const float epoch_ratio = (float)(epoch - 1) / (float) std::max(1, epochs - 1);
        const float lr = lr_min + 0.5f * (lr0 - lr_min) * (1.0f + std::cos(3.1415926535f * epoch_ratio));

        float total_loss = 0.0f;

        for (int64_t b = 0; b < n_batches; ++b) {
            std::fill(grad_w.begin(), grad_w.end(), 0.0f);
            std::fill(grad_g.begin(), grad_g.end(), 0.0f);

            std::vector<float> bx((size_t) batch_size * dim);
            std::vector<float> by((size_t) batch_size * dim);

            for (int i = 0; i < batch_size; ++i) {
                const int64_t tok_idx = indices[b * batch_size + i];
                std::memcpy(bx.data() + (size_t) i * dim, sdata.x.data() + (size_t) tok_idx * dim, dim * sizeof(float));
                std::memcpy(by.data() + (size_t) i * dim, sdata.y.data() + (size_t) tok_idx * dim, dim * sizeof(float));
            }

            std::vector<float> dp((size_t) batch_size * dim);
            std::vector<float> dz(batch_size);
            float batch_loss = 0.0f;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+:batch_loss) schedule(static)
#endif
            for (int i = 0; i < batch_size; ++i) {
                const float * xi = bx.data() + (size_t) i * dim;
                const float * yi = by.data() + (size_t) i * dim;
                float * dpi      = dp.data() + (size_t) i * dim;

                float z = 0.0f;
                for (uint32_t k = 0; k < dim; ++k) {
                    z += gate_proj[k] * xi[k];
                }
                const float g = 1.0f / (1.0f + std::exp(-z));

                float z_grad_accum = 0.0f;
                const float inv_scale = 1.0f / (float)(batch_size * dim);

                for (uint32_t j = 0; j < dim; ++j) {
                    float p = 0.0f;
                    const float * w_row = w_merge.data() + (size_t) j * dim;
                    for (uint32_t k = 0; k < dim; ++k) {
                        p += w_row[k] * xi[k];
                    }

                    const float y_hat = p * g;
                    const float err = y_hat - yi[j];
                    const float abs_err = std::abs(err);

                    float loss_val = 0.0f;
                    float d_yhat   = 0.0f;

                    if (abs_err < 1.0f) {
                        loss_val = 0.5f * err * err;
                        d_yhat   = err * inv_scale;
                    } else {
                        loss_val = abs_err - 0.5f;
                        d_yhat   = (err > 0.0f ? 1.0f : -1.0f) * inv_scale;
                    }

                    batch_loss += loss_val;
                    dpi[j] = d_yhat * g;
                    z_grad_accum += d_yhat * p;
                }

                dz[i] = z_grad_accum * g * (1.0f - g);
            }

            total_loss += batch_loss / (float)(batch_size * dim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int64_t j = 0; j < (int64_t) dim; ++j) {
                float * gw_row = grad_w.data() + (size_t) j * dim;
                for (uint32_t k = 0; k < dim; ++k) {
                    float sum = 0.0f;
                    for (int i = 0; i < batch_size; ++i) {
                        sum += dp[(size_t) i * dim + j] * bx[(size_t) i * dim + k];
                    }
                    gw_row[k] = sum;
                }
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int64_t k = 0; k < (int64_t) dim; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < batch_size; ++i) {
                    sum += dz[i] * bx[(size_t) i * dim + k];
                }
                grad_g[k] = sum;
            }

            float grad_norm_sq = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:grad_norm_sq) schedule(static)
#endif
            for (int64_t idx = 0; idx < (int64_t) w_size; ++idx) {
                grad_norm_sq += grad_w[idx] * grad_w[idx];
            }
            for (uint32_t k = 0; k < dim; ++k) {
                grad_norm_sq += grad_g[k] * grad_g[k];
            }

            const float total_norm = std::sqrt(grad_norm_sq);
            if (total_norm > grad_clip_norm && total_norm > 0.0f) {
                const float scale = grad_clip_norm / total_norm;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int64_t idx = 0; idx < (int64_t) w_size; ++idx) {
                    grad_w[idx] *= scale;
                }
                for (uint32_t k = 0; k < dim; ++k) {
                    grad_g[k] *= scale;
                }
            }

            opt_step++;
            const float beta1_t = std::pow(beta1, (float) opt_step);
            const float beta2_t = std::pow(beta2, (float) opt_step);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int64_t idx = 0; idx < (int64_t) w_size; ++idx) {
                w_merge[idx] -= lr * weight_decay * w_merge[idx];
                m_w[idx] = beta1 * m_w[idx] + (1.0f - beta1) * grad_w[idx];
                v_w[idx] = beta2 * v_w[idx] + (1.0f - beta2) * grad_w[idx] * grad_w[idx];
                const float m_hat = m_w[idx] / (1.0f - beta1_t);
                const float v_hat = v_w[idx] / (1.0f - beta2_t);
                w_merge[idx] -= lr * (m_hat / (std::sqrt(v_hat) + eps));
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int64_t k = 0; k < (int64_t) dim; ++k) {
                gate_proj[k] -= lr * weight_decay * gate_proj[k];
                m_g[k] = beta1 * m_g[k] + (1.0f - beta1) * grad_g[k];
                v_g[k] = beta2 * v_g[k] + (1.0f - beta2) * grad_g[k] * grad_g[k];
                const float m_hat = m_g[k] / (1.0f - beta1_t);
                const float v_hat = v_g[k] / (1.0f - beta2_t);
                gate_proj[k] -= lr * (m_hat / (std::sqrt(v_hat) + eps));
            }
        }

        const float avg_loss = total_loss / (float) std::max((int64_t)1, n_batches);
        fprintf(stderr, "\r[Stage %zu/7] Epoch %2d/%d | Loss: %.6f | LR: %.6f",
                stage_num, epoch, epochs, avg_loss, lr);
        fflush(stderr);
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    fprintf(stderr, "\n>> Stage %zu completed in %.2f sec\n", stage_num, elapsed_sec);

    std::vector<ggml_fp16_t> w_fp16(w_size);
    for (size_t idx = 0; idx < w_size; ++idx) {
        w_fp16[idx] = ggml_fp32_to_fp16(w_merge[idx]);
    }

    std::ofstream fw(out_w, std::ios::binary);
    if (fw.is_open()) {
        fw.write(reinterpret_cast<const char *>(w_fp16.data()), w_size * sizeof(ggml_fp16_t));
        fw.close();
        LOG_INF("Saved %s (%.2f MB, FP16)\n", out_w.c_str(), (w_size * sizeof(ggml_fp16_t)) / (1024.0 * 1024.0));
    } else {
        LOG_ERR("Failed to save %s\n", out_w.c_str());
    }

    std::vector<ggml_fp16_t> g_fp16(g_size);
    for (size_t idx = 0; idx < g_size; ++idx) {
        g_fp16[idx] = ggml_fp32_to_fp16(gate_proj[idx]);
    }

    std::ofstream fg(out_g, std::ios::binary);
    if (fg.is_open()) {
        fg.write(reinterpret_cast<const char *>(g_fp16.data()), g_size * sizeof(ggml_fp16_t));
        fg.close();
        LOG_INF("Saved %s (%.2f KB, FP16)\n", out_g.c_str(), (g_size * sizeof(ggml_fp16_t)) / 1024.0);
    } else {
        LOG_ERR("Failed to save %s\n", out_g.c_str());
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    common_init();

    std::string out_dir = "weights";
    std::string data_dir = "";
    std::string dataset_file = "";
    int epochs = 40;
    int batch_size = 2048;
    float lr0 = 1e-3f;
    float lr_min = 1e-5f;
    float weight_decay = 1e-4f;
    float gamma = 0.08f;
    float grad_clip = 1.0f;
    int target_stage = 0;

    std::vector<char *> clean_argv;
    clean_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_file = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr0 = std::stof(argv[++i]);
        } else if (arg == "--lr-min" && i + 1 < argc) {
            lr_min = std::stof(argv[++i]);
        } else if (arg == "--gamma" && i + 1 < argc) {
            gamma = std::stof(argv[++i]);
        } else if (arg == "--wd" && i + 1 < argc) {
            weight_decay = std::stof(argv[++i]);
        } else if (arg == "--clip" && i + 1 < argc) {
            grad_clip = std::stof(argv[++i]);
        } else if (arg == "--stage" && i + 1 < argc) {
            target_stage = std::stoi(argv[++i]);
        } else {
            clean_argv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) clean_argv.size(), clean_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    trainer_collector collector;
    collector.gamma = gamma;

    bool have_files = false;
    uint32_t model_dim = 0;

    if (!data_dir.empty()) {
        std::string test_path = data_dir + "/latents_stage1.bin";
        std::ifstream tf(test_path, std::ios::binary);
        if (tf.is_open()) {
            uint32_t n_tok = 0;
            uint32_t dim = 0;
            tf.read(reinterpret_cast<char *>(&n_tok), sizeof(n_tok));
            tf.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            if (dim > 0) {
                model_dim = dim;
                have_files = true;
            }
        }
    }

    if (have_files) {
        LOG_INF("Loading existing stage latents from %s (dim=%u)...\n", data_dir.c_str(), model_dim);
        for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
            if (target_stage > 0 && (int)(s + 1) != target_stage) {
                continue;
            }
            std::string path = data_dir + "/latents_stage" + std::to_string(s + 1) + ".bin";
            if (!load_stage_data_from_file(path, model_dim, gamma, collector.data[s])) {
                LOG_ERR("Failed to load stage data from %s\n", path.c_str());
            } else {
                LOG_INF("Stage %zu: loaded %lld tokens\n", s + 1, (long long) collector.data[s].total_tokens);
            }
        }
    } else {
        params.cb_eval = trainer_cb_eval;
        params.cb_eval_user_data = &collector;
        params.warmup = false;
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
        collector.dim = (uint32_t) llama_model_n_embd(model);
        model_dim = collector.dim;

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

        LOG_INF("Loaded %zu items from %s. Extracting into RAM on GPU...\n", problems.size(), dataset_file.c_str());

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

            LOG_INF("Extracted item %3zu/%3zu | tokens: %zu | S1: %lld | S7: %lld\n",
                    i + 1, problems.size(), tokens.size(),
                    (long long) collector.data[0].total_tokens,
                    (long long) collector.data[6].total_tokens);
        }

        llama_backend_free();
    }

    std::filesystem::create_directories(out_dir);

    for (size_t s = 0; s < llama_cascade_config::NUM_STAGES; ++s) {
        if (target_stage > 0 && (int)(s + 1) != target_stage) {
            continue;
        }

        if (collector.data[s].total_tokens == 0) {
            LOG_INF("Stage %zu has no tokens, skipping.\n", s + 1);
            continue;
        }

        std::string out_w = out_dir + "/w_merge_" + std::to_string(s + 1) + ".bin";
        std::string out_g = out_dir + "/gate_" + std::to_string(s + 1) + ".bin";

        LOG_INF("\n==================== TRAINING STAGE %zu/7 (%lld tokens) ====================\n",
                s + 1, (long long) collector.data[s].total_tokens);

        train_single_stage(
            s + 1,
            collector.data[s],
            model_dim,
            out_w,
            out_g,
            epochs,
            batch_size,
            lr0,
            lr_min,
            weight_decay,
            grad_clip
        );

        collector.data[s].x.clear();
        collector.data[s].x.shrink_to_fit();
        collector.data[s].y.clear();
        collector.data[s].y.shrink_to_fit();
    }

    LOG_INF("\n>> CPT Training finished successfully!\n");
    return 0;
}

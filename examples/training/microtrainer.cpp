#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

static bool filter_adapter_params(const struct ggml_tensor * tensor, void * /*userdata*/) {
    if (!tensor || !tensor->name) {
        return false;
    }
    return std::strcmp(tensor->name, "w_merge") == 0 || std::strcmp(tensor->name, "gate_proj") == 0;
}

static void save_tensor_to_binary(struct ggml_tensor * tensor, const std::string & filename) {
    if (!tensor) {
        LOG_ERR("%s: tensor is null, cannot save to %s\n", __func__, filename.c_str());
        return;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<char> buffer(nbytes);
    ggml_backend_tensor_get(tensor, buffer.data(), 0, nbytes);

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open %s for writing\n", __func__, filename.c_str());
        return;
    }

    file.write(buffer.data(), nbytes);
    LOG_INF("%s: successfully saved %s (%.2f MB)\n", __func__, filename.c_str(), nbytes / (1024.0 * 1024.0));
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.load_mode != LLAMA_LOAD_MODE_NONE) {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        params.cache_type_v = GGML_TYPE_F32;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model or initialize context\n", __func__);
        return 1;
    }

    model->hparams.f_final_logit_softcapping = 0.0f;

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.empty()) {
        LOG_ERR("%s: prompt tokens are empty\n", __func__);
        return 1;
    }

    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("Training Sinusoidal Latent Exploration Adapter on GPU...\n");
    LOG_INF("Optimizer: %s | Base LR: %.2g | Epochs: %d\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (int) lr.epochs);

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ filter_adapter_params,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &params.lr,
        /*optimizer_type  =*/ params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    save_tensor_to_binary(model->w_merge, "w_merge.bin");
    save_tensor_to_binary(model->gate_proj, "gate.bin");

    llama_backend_free();

    return 0;
}

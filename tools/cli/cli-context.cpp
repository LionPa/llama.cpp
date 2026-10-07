#include "cli-context.h"
#include "cli-ui.h"

#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "console.h"

#include "json.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <thread>

using json = common_json;

struct brainstorm_config {
    bool enabled = false;
    int num_batches = 0;
    std::vector<float> batch_sins;
    std::string merger_prompt_template;
    float merger_sin = 0.05f;
};

static bool parse_brainstorm_command(const std::string & cmd, brainstorm_config & cfg, std::string & error_msg) {
    std::string s = string_strip(cmd);
    if (string_starts_with(s, "/brainstorm")) {
        s = string_strip(s.substr(11));
    }
    if (s.empty()) {
        if (cfg.enabled) {
            std::string sins;
            for (size_t i = 0; i < cfg.batch_sins.size(); ++i) {
                if (i > 0) sins += ", ";
                sins += std::to_string(cfg.batch_sins[i]);
            }
            error_msg = "Brainstorm mode is ON: " + std::to_string(cfg.num_batches) + " batches [" + sins + "], merger sin = " + std::to_string(cfg.merger_sin) + "\nUse /brainstorm off to disable.";
        } else {
            error_msg = "Brainstorm mode is OFF.\nUsage: /brainstorm <batches> <sin_1> ... <sin_N> \"<merger_prompt>\" <sin_merger>\nExample: /brainstorm 3, 0.065, 0.3, 10, \"Merge %1 %2 %3\", 0.065";
        }
        return false;
    }
    if (s == "off" || s == "disable" || s == "0") {
        cfg.enabled = false;
        error_msg = "Brainstorm mode disabled.";
        return false;
    }

    std::string clean;
    bool in_q = false;
    char q_ch = 0;
    bool in_paren = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_q) {
            clean += c;
            if (c == q_ch && (i == 0 || s[i - 1] != '\\')) {
                in_q = false;
            }
        } else {
            if (c == '"' || c == '\'') {
                in_q = true;
                q_ch = c;
                clean += c;
            } else if (c == '(') {
                in_paren = true;
            } else if (c == ')') {
                in_paren = false;
            } else if (!in_paren) {
                if (c == ',') {
                    clean += ' ';
                } else {
                    clean += c;
                }
            }
        }
    }

    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < clean.size()) {
        while (i < clean.size() && (clean[i] == ' ' || clean[i] == '\t')) {
            ++i;
        }
        if (i >= clean.size()) break;
        if (clean[i] == '"' || clean[i] == '\'') {
            char quote = clean[i++];
            std::string tok;
            while (i < clean.size() && clean[i] != quote) {
                if (clean[i] == '\\' && i + 1 < clean.size()) {
                    char next = clean[i + 1];
                    if (next == 'n') tok += '\n';
                    else if (next == 't') tok += '\t';
                    else if (next == quote) tok += quote;
                    else if (next == '\\') tok += '\\';
                    else { tok += clean[i]; tok += next; }
                    i += 2;
                } else {
                    tok += clean[i++];
                }
            }
            if (i < clean.size() && clean[i] == quote) {
                ++i;
            }
            tokens.push_back(tok);
        } else {
            std::string tok;
            while (i < clean.size() && clean[i] != ' ' && clean[i] != '\t') {
                tok += clean[i++];
            }
            tokens.push_back(tok);
        }
    }

    if (tokens.empty()) {
        error_msg = "Invalid /brainstorm arguments.";
        return false;
    }

    int n_batches = 0;
    try {
        n_batches = std::stoi(tokens[0]);
    } catch (...) {
        error_msg = "Invalid number of batches: " + tokens[0];
        return false;
    }

    if (n_batches < 1) {
        error_msg = "Number of batches must be >= 1.";
        return false;
    }

    if (tokens.size() < (size_t)(1 + n_batches + 2)) {
        error_msg = "Not enough arguments for /brainstorm. Expected: " + std::to_string(n_batches) + " sin values, merger prompt in quotes, and merger sin value.";
        return false;
    }

    std::vector<float> sins;
    for (int b = 0; b < n_batches; ++b) {
        try {
            sins.push_back(std::stof(tokens[1 + b]));
        } catch (...) {
            error_msg = "Invalid cascade-sin value for batch " + std::to_string(b + 1) + ": " + tokens[1 + b];
            return false;
        }
    }

    std::string merger_prompt = tokens[1 + n_batches];
    float merger_sin = 0.05f;
    try {
        merger_sin = std::stof(tokens[1 + n_batches + 1]);
    } catch (...) {
        error_msg = "Invalid cascade-sin value for merger: " + tokens[1 + n_batches + 1];
        return false;
    }

    cfg.enabled = true;
    cfg.num_batches = n_batches;
    cfg.batch_sins = sins;
    cfg.merger_prompt_template = merger_prompt;
    cfg.merger_sin = merger_sin;
    return true;
}

struct cli_context_impl {
    json messages      = json::array();
    json pending_media = json::array();
    brainstorm_config bs_cfg;
};

cli_context::cli_context(const common_params & params) : params(params), impl(new cli_context_impl()) {}

cli_context::~cli_context() {
    shutdown();
}

std::atomic<bool> & cli_context::interrupted() {
    static std::atomic<bool> flag = false;
    return flag;
}

static bool should_stop() {
    return cli_context::interrupted().load();
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

// number of values an arg consumes on the command line
static int arg_num_values(const common_arg & opt) {
    if (opt.value_hint_2 != nullptr) {
        return 2;
    }
    if (opt.value_hint != nullptr) {
        return 1;
    }
    return 0;
}

static std::string format_error_message(const json & err) {
    if (err.contains("error") && err.at("error").is_object()) {
        const auto & e = err.at("error");
        if (e.contains("message") && e.at("message").is_string()) {
            return e.at("message").get<std::string>();
        }
    }
    return err.dump();
}

// err is the raw response body of a failed request; it may or may not be JSON
static std::string format_error_message(const std::string & err) {
    json parsed = json::parse_no_throw(err);
    if (!parsed.is_discarded()) {
        return format_error_message(parsed);
    }
    return err;
}

static std::string media_type_from_ext(const std::string & fname) {
    std::string ext = std::filesystem::path(fname).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".wav" || ext == ".mp3") {
        return "audio";
    }
    if (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm") {
        return "video";
    }
    return "image";
}

bool cli_context::init() {
    ui::init(params);

    std::optional<ui::spinner> spinner;

    bool use_external_server = !params.server_base.empty();
    if (use_external_server) {
        std::string base = params.server_base;
        while (!base.empty() && base.back() == '/') {
            base.pop_back();
        }
        client.server_base = base;

        spinner.emplace("Connecting to server at " + base);
    } else {
        if (params.model.path.empty() && params.model.url.empty() &&
                params.model.hf_repo.empty() && params.model.docker_repo.empty()) {
            ui::show_error(
                "no model specified",
                "use -m <file.gguf> or -hf <user/repo> to run a local model,\n"
                "or --server-base <url> to connect to a running llama-server"
            );
            return false;
        }

        spinner.emplace("\n\nLoading model...");

        server.emplace();
        if (!server->start(params)) {
            ui::show_error("server start failed");
            return false;
        }
        if (!server->wait_ready(should_stop)) {
            if (!should_stop()) {
                ui::show_error("the server exited before becoming ready");
            }
            return false;
        }
        client.server_base = server->address();
    }

    // for --server-base this is the main availability check; for a spawned
    // server it is a cheap sanity check on top of the ready signal
    auto is_aborted = [this]() {
        return should_stop() || (server && !server->alive());
    };
    bool healthy = false;
    try {
        healthy = client.wait_health(is_aborted);
    } catch (const std::exception & e) {
        client.last_error = e.what();
    }
    if (!healthy) {
        if (!should_stop()) {
            ui::show_error(client.last_error);
        }
        return false;
    }

    if (use_external_server) {
        spinner.reset();
        try {
            if (!list_and_ask_models()) {
                return false;
            }
        } catch (const common_json_error & e) {
            ui::show_error(e.what());
            ui::show_message("This might be caused by an incorrect server-base endpoint URL");
            return false;
        } catch (const std::exception & e) {
            ui::show_error(e.what());
            return false;
        }

        // restore the spinner for the next step
        spinner.emplace("Waiting for server...");
    }

    fetch_server_props();

    if (!params.out_file.empty()) {
        output_file.emplace(params.out_file);
        if (!output_file->is_open()) {
            ui::show_error(string_format("failed to open output file '%s'", params.out_file.c_str()));
            return false;
        }
    }

    return true;
}

void cli_context::fetch_server_props() {
    try {
        json props = json::parse(client.get("/props"));
        model_name = props.value("model_alias", "");
        if (model_name.empty()) {
            const std::string path = props.value("model_path", "");
            if (!path.empty()) {
                model_name = std::filesystem::path(path).filename().string();
            }
        }
        model_ftype = props.value("model_ftype", "");
        build_info = props.value("build_info", "");
        if (props.contains("modalities") && props.at("modalities").is_object()) {
            const auto & modalities = props.at("modalities");
            has_vision = modalities.value("vision", false);
            has_audio  = modalities.value("audio", false);
            has_video  = modalities.value("video", false);
        }
    } catch (const std::exception & e) {
        // /props can be disabled on remote servers; not fatal
        LOG_DBG("failed to fetch /props: %s\n", e.what());
    }
}

bool cli_context::list_and_ask_models() {
    json resp = json::parse(client.get("/v1/models"));
    if (!resp.contains("data") || !resp.at("data").is_array()) {
        throw std::runtime_error("invalid response from /v1/models");
    }
    std::vector<std::string> models;
    std::vector<std::string> models_display;
    for (const auto & m : resp.at("data")) {
        if (!m.contains("id") || !m.at("id").is_string()) {
            continue;
        }
        std::string name = m.at("id").get<std::string>();
        std::string display = name;
        if (m.contains("aliases") && m.at("aliases").is_array()) {
            std::vector<std::string> aliases;
            for (const auto & a : m.at("aliases")) {
                if (a.is_string()) {
                    aliases.push_back(a.get<std::string>());
                }
            }
            if (!aliases.empty()) {
                display += " (" + string_join(aliases, ", ") + ")";
            }
        }
        models.push_back(name);
        models_display.push_back(display);
    }

    // only one model: use it without asking
    if (models.size() == 1) {
        model_name = models[0];
        client.model = model_name;
        return true;
    }

    std::string message = "\nAvailable models:";
    for (size_t i = 0; i < models_display.size(); ++i) {
        message += "\n  " + std::to_string(i + 1) + ". " + models_display[i];
    }
    message += "\n";
    ui::show_message(message);
    std::string selection;
    while (selection.empty()) {
        if (should_stop()) {
            return false;
        }
        ui::user_turn user_turn;
        selection = user_turn.read_input(false, "Select model by number: ");
        if (selection.empty()) {
            continue;
        }
        try {
            size_t idx = std::stoul(selection);
            if (idx > 0 && idx <= models.size()) {
                model_name = models[idx - 1];
                client.model = model_name;
                ui::show_message("Selected model: " + model_name);
                break;
            }
        } catch (...) {
            // ignore
        }
        ui::show_error("Invalid selection. Please enter a valid number.");
        selection.clear();
        continue;
    }
    return true;
}

void cli_context::add_system_prompt() {
    if (!params.system_prompt.empty()) {
        impl->messages.push_back({
            {"role",    "system"},
            {"content", params.system_prompt}
        });
    }
}

void cli_context::push_user_message(const std::string & text) {
    json content;
    if (impl->pending_media.empty()) {
        content = text;
    } else {
        // multimodal message: media parts first, then the text
        content = impl->pending_media;
        content.push_back({
            {"type", "text"},
            {"text", text}
        });
        impl->pending_media = json::array();
    }
    impl->messages.push_back({
        {"role",    "user"},
        {"content", content}
    });
}

bool cli_context::stage_media_file(const std::string & fname, const std::string & type) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        return false;
    }
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string encoded = base64::encode(data);

    if (type == "audio") {
        std::string ext = std::filesystem::path(fname).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        impl->pending_media.push_back({
            {"type", "input_audio"},
            {"input_audio", {
                {"data",   encoded},
                {"format", ext == ".mp3" ? "mp3" : "wav"}
            }}
        });
    } else if (type == "video") {
        impl->pending_media.push_back({
            {"type", "input_video"},
            {"input_video", {
                {"data", encoded}
            }}
        });
    } else {
        // the server detects the actual image type from the data
        impl->pending_media.push_back({
            {"type", "image_url"},
            {"image_url", {
                {"url", "data:image/unknown;base64," + encoded}
            }}
        });
    }
    return true;
}

void cli_context::write_output_file(const std::string & content) {
    if (output_file) {
        (*output_file) << content;
        output_file->flush();
    }
}

bool cli_context::generate_completion(generated_content & content_out, cli_timings & timings, float custom_sin) {
    json body = {
        {"messages",          impl->messages},
        {"stream",            true},
        {"timings_per_token", true},
    };
    if (!client.model.empty()) {
        body["model"] = client.model;
    }
    if (custom_sin >= 0.0f) {
        body["cascade_sin"] = custom_sin;
        body["cache_prompt"] = false;
    } else if (params.cascade_sin >= 0.0f) {
        body["cascade_sin"] = params.cascade_sin;
    }

    bool stream_error = false;

    ui::assistant_turn a;

    std::string err = client.post_sse("/v1/chat/completions", body.dump(), should_stop, [&](const std::string & payload) {
        json chunk = json::parse_no_throw(payload);
        if (chunk.is_discarded()) {
            return;
        }
        if (chunk.contains("error")) {
            stream_error = true;
            ui::show_error(format_error_message(chunk));
            return;
        }
        if (chunk.contains("timings")) {
            const auto & t = chunk.at("timings");
            timings.prompt_per_second    = t.value("prompt_per_second",    0.0);
            timings.predicted_per_second = t.value("predicted_per_second", 0.0);
        }
        if (!chunk.contains("choices") || !chunk.at("choices").is_array() || chunk.at("choices").empty()) {
            return;
        }
        const auto & choice = chunk.at("choices").at(0);
        if (!choice.contains("delta")) {
            return;
        }
        const auto & delta = choice.at("delta");
        if (delta.contains("reasoning_content") && delta.at("reasoning_content").is_string()) {
            const std::string text = delta.at("reasoning_content").get<std::string>();
            if (!text.empty()) {
                content_out.reasoning += text;
                a.push(ui::ASSISTANT_DISPLAY_MODE_REASONING, text);
            }
        }
        if (delta.contains("content") && delta.at("content").is_string()) {
            const std::string text = delta.at("content").get<std::string>();
            if (!text.empty()) {
                content_out.content += text;
                a.push(ui::ASSISTANT_DISPLAY_MODE_CONTENT, text);
            }
        }
    });

    cli_context::interrupted().store(false);

    if (!err.empty()) {
        ui::show_error(format_error_message(err));
        return false;
    }
    return !stream_error;
}

int cli_context::run() {
    add_system_prompt();

    std::string modalities = "text";
    if (has_vision) {
        modalities += ", vision";
    }
    if (has_audio) {
        modalities += ", audio";
    }
    if (has_video) {
        modalities += ", video";
    }

    std::string banner;
    banner += "\n";
    banner += LLAMA_ASCII_LOGO;
    banner += "\n";
    banner += "build      : " + build_info + "\n";
    banner += "model      : " + model_name + "\n";
    if (!model_ftype.empty()) {
        banner += "ftype      : " + model_ftype + "\n";
    }
    banner += "modalities : " + modalities + "\n";
    if (!params.system_prompt.empty()) {
        banner += "using custom system prompt\n";
    }
    banner += "\n";
    banner += "available commands:\n";
    banner += "  /exit or Ctrl+C     stop or exit\n";
    banner += "  /regen              regenerate the last response\n";
    banner += "  /clear              clear the chat history\n";
    banner += "  /brainstorm ...     toggle brainstorm multi-sin mode\n";
    banner += "  /read <file>        add a text file\n";
    banner += "  /glob <pattern>     add text files using globbing pattern\n";
    if (has_vision) {
        banner += "  /image <file>       add an image file\n";
    }
    if (has_audio) {
        banner += "  /audio <file>       add an audio file\n";
    }
    if (has_video) {
        banner += "  /video <file>       add a video file\n";
    }
    banner += "\n";

    ui::show_message(banner);

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cur_msg += "--- File: ";
        cur_msg += fname;
        cur_msg += " ---\n";
        cur_msg += content;
        ui::show_message(string_format("Loaded text from '%s'", fname.c_str()));
        return true;
    };

    while (true) {
        std::string buffer;
        {
            ui::user_turn user_turn;

            if (params.prompt.empty()) {
                buffer = user_turn.read_input(params.multiline_input);
            } else {
                // process input prompt from args
                for (auto & fname : params.image) {
                    if (!stage_media_file(fname, media_type_from_ext(fname))) {
                        ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
                        break;
                    }
                    ui::show_message(string_format("Loaded media from '%s'", fname.c_str()));
                }
                buffer = params.prompt;
                user_turn.echo(buffer);
                params.prompt.clear(); // only use it once
            }
        }

        if (should_stop()) {
            cli_context::interrupted().store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() && buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (impl->messages.size() >= 2) {
                size_t last_idx = impl->messages.size() - 1;
                impl->messages.erase(last_idx);
                add_user_msg = false;
            } else {
                ui::show_error("No message to regenerate.");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            impl->messages.clear();
            add_system_prompt();

            impl->pending_media = json::array();
            ui::show_message("Chat history cleared.");
            continue;
        } else if (string_starts_with(buffer, "/brainstorm")) {
            std::string err_msg;
            if (parse_brainstorm_command(buffer, impl->bs_cfg, err_msg)) {
                std::string sins;
                for (size_t b = 0; b < impl->bs_cfg.batch_sins.size(); ++b) {
                    if (b > 0) sins += ", ";
                    sins += std::to_string(impl->bs_cfg.batch_sins[b]);
                }
                ui::show_message(string_format("Brainstorm mode ENABLED:\n  - Batches: %d [%s]\n  - Merger template: \"%s\"\n  - Merger sin: %.4f\nType /brainstorm off to disable.",
                    impl->bs_cfg.num_batches, sins.c_str(), impl->bs_cfg.merger_prompt_template.c_str(), impl->bs_cfg.merger_sin));
            } else {
                ui::show_message(err_msg);
            }
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && has_vision) ||
                (string_starts_with(buffer, "/audio ") && has_audio) ||
                (string_starts_with(buffer, "/video ") && has_video)) {
            std::string type = buffer.substr(1, 5);
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            if (!stage_media_file(fname, type)) {
                ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
                continue;
            }
            ui::show_message(string_format("Loaded media from '%s'", fname.c_str()));
            write_output_file(string_format("User: Added media: %s\n", fname.c_str()));
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            write_output_file(string_format("User: Added text file: %s\n", fname.c_str()));
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, '~')) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = home + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                const std::string full_path = (curdir / rel).string();
                if (!add_text_file(full_path)) {
                    continue;
                }
                write_output_file(string_format("User: Added text file: %s\n", full_path.c_str()));

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    ui::show_error(string_format("Maximum number of globbed files allowed (%zu) reached.", FILE_GLOB_MAX_RESULTS));
                    break;
                }
            }
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        if (impl->bs_cfg.enabled && add_user_msg) {
            std::string user_prompt = cur_msg;
            push_user_message(user_prompt);
            write_output_file(string_format("User:\n%s\n\n", user_prompt.c_str()));
            cur_msg.clear();

            json base_messages = impl->messages;
            std::vector<std::string> branch_responses(impl->bs_cfg.num_batches);
            std::vector<generated_content> branch_contents(impl->bs_cfg.num_batches);
            std::vector<cli_timings> branch_timings(impl->bs_cfg.num_batches);
            std::vector<bool> branch_success(impl->bs_cfg.num_batches, false);
            std::mutex out_mutex;

            ui::show_message(string_format("\n>>> [Brainstorm: Launching %d batches in parallel...] <<<\n", impl->bs_cfg.num_batches));

            struct batch_live_stat {
                std::atomic<int> n_tokens{0};
                std::atomic<float> speed{0.0f};
                std::atomic<bool> done{false};
            };
            std::vector<std::unique_ptr<batch_live_stat>> live_stats;
            live_stats.reserve(impl->bs_cfg.num_batches);
            for (int i = 0; i < impl->bs_cfg.num_batches; ++i) {
                live_stats.push_back(std::make_unique<batch_live_stat>());
            }

            std::atomic<bool> ticker_active{true};
            std::thread ticker([&]() {
                while (ticker_active.load() && !should_stop()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    if (!ticker_active.load()) {
                        break;
                    }
                    std::string status = "\r\033[K[Brainstorm] ";
                    for (int i = 0; i < impl->bs_cfg.num_batches; ++i) {
                        int tok = live_stats[i]->n_tokens.load();
                        float spd = live_stats[i]->speed.load();
                        bool d = live_stats[i]->done.load();
                        if (i > 0) {
                            status += " | ";
                        }
                        if (d) {
                            status += string_format("B%d: done (%d t)", i + 1, tok);
                        } else if (tok > 0) {
                            status += string_format("B%d: %d t (%.1f t/s)", i + 1, tok, spd);
                        } else {
                            status += string_format("B%d: starting...", i + 1);
                        }
                    }
                    console::log("%s", status.c_str());
                    console::flush();
                }
            });

            std::vector<std::thread> workers;
            workers.reserve(impl->bs_cfg.num_batches);

            for (int b = 0; b < impl->bs_cfg.num_batches; ++b) {
                workers.emplace_back([this, b, &base_messages, &branch_responses, &branch_contents, &branch_timings, &branch_success, &out_mutex, &live_stats]() {
                    float cur_sin = impl->bs_cfg.batch_sins[b];

                    json body = {
                        {"messages",          base_messages},
                        {"stream",            true},
                        {"timings_per_token", true},
                        {"cascade_sin",       cur_sin},
                        {"cache_prompt",      false},
                    };
                    if (!client.model.empty()) {
                        body["model"] = client.model;
                    }

                    cli_client worker_client = client;
                    generated_content content_out;
                    cli_timings timings;
                    bool stream_error = false;

                    std::string err = worker_client.post_sse("/v1/chat/completions", body.dump(), should_stop, [&](const std::string & payload) {
                        json chunk = json::parse_no_throw(payload);
                        if (chunk.is_discarded()) {
                            return;
                        }
                        if (chunk.contains("error")) {
                            stream_error = true;
                            std::lock_guard<std::mutex> lock(out_mutex);
                            ui::show_error(format_error_message(chunk));
                            return;
                        }
                        if (chunk.contains("timings")) {
                            const auto & t = chunk.at("timings");
                            timings.prompt_per_second    = t.value("prompt_per_second",    0.0);
                            timings.predicted_per_second = t.value("predicted_per_second", 0.0);
                            live_stats[b]->speed.store((float)timings.predicted_per_second);
                        }
                        if (!chunk.contains("choices") || !chunk.at("choices").is_array() || chunk.at("choices").empty()) {
                            return;
                        }
                        const auto & choice = chunk.at("choices").at(0);
                        if (!choice.contains("delta")) {
                            return;
                        }
                        const auto & delta = choice.at("delta");
                        if (delta.contains("reasoning_content") && delta.at("reasoning_content").is_string()) {
                            const std::string text = delta.at("reasoning_content").get<std::string>();
                            if (!text.empty()) {
                                content_out.reasoning += text;
                                live_stats[b]->n_tokens.fetch_add(1);
                            }
                        }
                        if (delta.contains("content") && delta.at("content").is_string()) {
                            const std::string text = delta.at("content").get<std::string>();
                            if (!text.empty()) {
                                content_out.content += text;
                                live_stats[b]->n_tokens.fetch_add(1);
                            }
                        }
                    });

                    if (!err.empty()) {
                        std::lock_guard<std::mutex> lock(out_mutex);
                        ui::show_error(format_error_message(err));
                    } else if (!stream_error) {
                        branch_success[b] = true;
                    }

                    branch_contents[b] = std::move(content_out);
                    branch_timings[b] = timings;

                    if (!branch_contents[b].content.empty()) {
                        branch_responses[b] = branch_contents[b].content;
                    } else {
                        branch_responses[b] = "[Ответ отсутствует: размышление не закончилось / No response: thinking was not completed]";
                    }
                    live_stats[b]->done.store(true);
                });
            }

            for (auto & th : workers) {
                if (th.joinable()) {
                    th.join();
                }
            }

            ticker_active.store(false);
            if (ticker.joinable()) {
                ticker.join();
            }
            console::log("\r\033[K");
            console::flush();

            for (int b = 0; b < impl->bs_cfg.num_batches; ++b) {
                float cur_sin = impl->bs_cfg.batch_sins[b];
                ui::show_message(string_format("\n>>> [Brainstorm Batch %d/%d | cascade-sin = %.4f] <<<\n", b + 1, impl->bs_cfg.num_batches, cur_sin));
                if (!branch_contents[b].reasoning.empty()) {
                    ui::show_message("[Start thinking]\n" + branch_contents[b].reasoning + "\n[End thinking]\n\n");
                }
                if (!branch_contents[b].content.empty()) {
                    ui::show_message(branch_contents[b].content + "\n");
                } else {
                    ui::show_message("[Ответ отсутствует: размышление не закончилось]\n");
                }

                if (output_file) {
                    std::string out_branch = string_format("Assistant [Batch %d (sin=%.4f)]:\n", b + 1, cur_sin);
                    if (!branch_contents[b].reasoning.empty()) {
                        out_branch += "[Start thinking]\n\n" + branch_contents[b].reasoning + "[End thinking]\n\n";
                    }
                    if (!branch_contents[b].content.empty()) {
                        out_branch += branch_contents[b].content + "\n\n";
                    } else {
                        out_branch += "[Ответ отсутствует: размышление не закончилось]\n\n";
                    }
                    write_output_file(out_branch);
                }

                if (params.show_timings) {
                    ui::show_info(string_format(
                        "[ Batch %d: %.1f t/s (prompt: %.1f t/s) ]",
                        b + 1,
                        branch_timings[b].predicted_per_second,
                        branch_timings[b].prompt_per_second
                    ));
                }
            }

            cli_context::interrupted().store(false);

            bool all_success = !should_stop();
            for (bool s : branch_success) {
                if (!s) {
                    all_success = false;
                    break;
                }
            }

            if (all_success) {
                std::string merger_input = impl->bs_cfg.merger_prompt_template;
                for (int b = 0; b < impl->bs_cfg.num_batches; ++b) {
                    std::string placeholder = "%" + std::to_string(b + 1);
                    size_t pos = 0;
                    while ((pos = merger_input.find(placeholder, pos)) != std::string::npos) {
                        merger_input.replace(pos, placeholder.length(), branch_responses[b]);
                        pos += branch_responses[b].length();
                    }
                    std::string sin_placeholder = "%sin" + std::to_string(b + 1);
                    std::string sin_val_str = string_format("%.4f", impl->bs_cfg.batch_sins[b]);
                    size_t spos = 0;
                    while ((spos = merger_input.find(sin_placeholder, spos)) != std::string::npos) {
                        merger_input.replace(spos, sin_placeholder.length(), sin_val_str);
                        spos += sin_val_str.length();
                    }
                }
                size_t ppos = 0;
                while ((ppos = merger_input.find("%prompt%", ppos)) != std::string::npos) {
                    merger_input.replace(ppos, 8, user_prompt);
                    ppos += user_prompt.length();
                }
                size_t ppos0 = 0;
                while ((ppos0 = merger_input.find("%0", ppos0)) != std::string::npos) {
                    merger_input.replace(ppos0, 2, user_prompt);
                    ppos0 += user_prompt.length();
                }

                ui::show_message(string_format("\n>>> [Brainstorm Merger | cascade-sin = %.4f] <<<\n", impl->bs_cfg.merger_sin));

                impl->messages = base_messages;
                impl->messages.push_back({
                    {"role", "user"},
                    {"content", merger_input}
                });

                cli_timings merger_timings;
                generated_content merger_content;
                generate_completion(merger_content, merger_timings, impl->bs_cfg.merger_sin);

                impl->messages = base_messages;
                json final_msg = {
                    {"role", "assistant"},
                    {"content", merger_content.content}
                };
                if (!merger_content.reasoning.empty()) {
                    final_msg["reasoning_content"] = merger_content.reasoning;
                }
                impl->messages.push_back(std::move(final_msg));

                if (output_file) {
                    std::string out_merger = string_format("Assistant [Merger (sin=%.4f)]:\n", impl->bs_cfg.merger_sin);
                    if (!merger_content.reasoning.empty()) {
                        out_merger += "[Start thinking]\n\n" + merger_content.reasoning + "[End thinking]\n\n";
                    }
                    out_merger += merger_content.content + "\n\n";
                    write_output_file(out_merger);
                }

                if (params.show_timings) {
                    ui::show_info(string_format(
                        "\n[ Prompt: %.1f t/s | Generation: %.1f t/s | \033[1;36m\xE2\x9A\xA1 Cascade: 7-Stage Lookahead (L24..L30 -> L26..L32, FP16)\033[0m ]",
                        merger_timings.prompt_per_second,
                        merger_timings.predicted_per_second
                    ));
                }
            } else {
                ui::show_error("Brainstorm batches incomplete or aborted.");
            }

            if (params.single_turn) {
                break;
            }
            continue;
        }

        if (add_user_msg) {
            push_user_message(cur_msg);
            write_output_file(string_format("User:\n%s\n\n", cur_msg.c_str()));
            cur_msg.clear();
        }

        cli_timings timings;
        generated_content content;
        generate_completion(content, timings);

        json assistant_msg = {
            {"role",    "assistant"},
            {"content", content.content}
        };
        if (!content.reasoning.empty()) {
            assistant_msg["reasoning_content"] = content.reasoning;
        }
        impl->messages.push_back(std::move(assistant_msg));

        if (output_file) {
            std::string out_content = "Assistant:\n";
            if (!content.reasoning.empty()) {
                out_content += "[Start thinking]\n\n";
                out_content += content.reasoning;
                out_content += "[End thinking]\n\n";
            }
            out_content += content.content;
            if (!out_content.empty() && out_content.back() != '\n') {
                out_content += "\n";
            }
            out_content += "\n";
            write_output_file(out_content);
        }

        if (params.show_timings) {
            ui::show_info(string_format(
                "\n[ Prompt: %.1f t/s | Generation: %.1f t/s | \033[1;36m\xE2\x9A\xA1 Cascade: 7-Stage Lookahead (L24..L30 -> L26..L32, FP16)\033[0m ]",
                timings.prompt_per_second,
                timings.predicted_per_second
            ));
        }

        if (params.single_turn) {
            break;
        }
    }

    ui::show_message("\n\nExiting...");

    return 0;
}

void cli_context::shutdown() {
    if (server) {
        server->stop();
        server.reset();
    }
    if (output_file) {
        output_file->close();
        output_file.reset();
    }
}

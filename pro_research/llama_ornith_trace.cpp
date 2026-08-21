#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct captured_tensor {
    std::string name;
    std::string type;
    std::array<int64_t, GGML_MAX_DIMS> shape{};
    std::vector<int32_t> integers;
    std::vector<float> floats;
};

struct capture_state {
    std::vector<captured_tensor> tensors;
};

bool indexed_name(const std::string & name, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if (name.compare(0, n, prefix) != 0 || name.size() == n) {
        return false;
    }
    return std::all_of(name.begin() + static_cast<std::ptrdiff_t>(n), name.end(),
        [](unsigned char value) { return std::isdigit(value) != 0; });
}

bool wanted_tensor(const ggml_tensor * tensor) {
    const std::string name = tensor->name;
    return indexed_name(name, "ffn_moe_topk-") ||
           indexed_name(name, "ffn_moe_weights_norm-") ||
           name == "result_norm";
}

float read_float(const uint8_t * data, ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float *>(data + offset);
    }
    if (type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + offset));
    }
    if (type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + offset));
    }
    throw std::runtime_error("unsupported floating capture type");
}

bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    if (!wanted_tensor(tensor)) {
        return false;
    }
    if (ask) {
        return true;
    }

    captured_tensor item;
    item.name = tensor->name;
    item.type = ggml_type_name(tensor->type);
    std::copy_n(tensor->ne, GGML_MAX_DIMS, item.shape.begin());

    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());

    const size_t count = static_cast<size_t>(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_I32) {
        item.integers.reserve(count);
    } else if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 ||
               tensor->type == GGML_TYPE_BF16) {
        item.floats.reserve(count);
    } else {
        throw std::runtime_error(item.name + ": unsupported tensor type " + item.type);
    }

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset = static_cast<size_t>(i3) * tensor->nb[3] +
                                          static_cast<size_t>(i2) * tensor->nb[2] +
                                          static_cast<size_t>(i1) * tensor->nb[1] +
                                          static_cast<size_t>(i0) * tensor->nb[0];
                    if (tensor->type == GGML_TYPE_I32) {
                        item.integers.push_back(
                            *reinterpret_cast<const int32_t *>(raw.data() + offset));
                    } else {
                        item.floats.push_back(read_float(raw.data(), tensor->type, offset));
                    }
                }
            }
        }
    }

    auto * state = static_cast<capture_state *>(user_data);
    state->tensors.push_back(std::move(item));
    return true;
}

std::string json_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 16);
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    result += "\\u00";
                    result += hex[ch >> 4];
                    result += hex[ch & 0x0f];
                } else {
                    result += static_cast<char>(ch);
                }
        }
    }
    return result;
}

void write_trace(
    const std::filesystem::path & path,
    const common_params & params,
    const std::vector<llama_token> & tokens,
    capture_state state) {
    std::sort(state.tensors.begin(), state.tensors.end(),
        [](const captured_tensor & lhs, const captured_tensor & rhs) {
            return lhs.name < rhs.name;
        });

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open trace output: " + path.string());
    }
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"ervf.ornith.llama_trace.v1\",\n";
    out << "  \"model\": \"" << json_escape(params.model.path) << "\",\n";
    out << "  \"prompt\": \"" << json_escape(params.prompt) << "\",\n";
    out << "  \"tokens\": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) out << ", ";
        out << tokens[i];
    }
    out << "],\n  \"tensors\": [\n";
    for (size_t i = 0; i < state.tensors.size(); ++i) {
        const captured_tensor & item = state.tensors[i];
        out << "    {\"name\": \"" << json_escape(item.name) << "\", \"type\": \""
            << item.type << "\", \"shape\": [";
        for (size_t d = 0; d < item.shape.size(); ++d) {
            if (d != 0) out << ", ";
            out << item.shape[d];
        }
        out << "], \"values\": [";
        if (!item.integers.empty()) {
            for (size_t j = 0; j < item.integers.size(); ++j) {
                if (j != 0) out << ", ";
                out << item.integers[j];
            }
        } else {
            for (size_t j = 0; j < item.floats.size(); ++j) {
                if (j != 0) out << ", ";
                out << item.floats[j];
            }
        }
        out << "]}" << (i + 1 == state.tensors.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

void print_usage(int, char ** argv) {
    LOG("usage: %s -m model.gguf -p prompt [common llama options]\n", argv[0]);
    LOG("set ORNITH_TRACE_OUT to the destination JSON path\n");
}

}  // namespace

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }

    const char * output_env = std::getenv("ORNITH_TRACE_OUT");
    if (output_env == nullptr || output_env[0] == '\0') {
        LOG_ERR("ORNITH_TRACE_OUT is required\n");
        return 1;
    }

    params.warmup = false;
    capture_state state;
    params.cb_eval = capture_callback;
    params.cb_eval_user_data = &state;

    llama_backend_init();
    llama_numa_init(params.numa);

    try {
        auto llama_init = common_init_from_params(params);
        llama_model * model = llama_init->model();
        llama_context * ctx = llama_init->context();
        if (model == nullptr || ctx == nullptr) {
            throw std::runtime_error("failed to initialize model/context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);
        if (tokens.empty()) {
            throw std::runtime_error("prompt tokenized to zero tokens");
        }
        if (const char * limit_env = std::getenv("ORNITH_TRACE_MAX_TOKENS")) {
            const long limit = std::strtol(limit_env, nullptr, 10);
            if (limit <= 0) {
                throw std::runtime_error("ORNITH_TRACE_MAX_TOKENS must be positive");
            }
            if (tokens.size() > static_cast<size_t>(limit)) {
                tokens.resize(static_cast<size_t>(limit));
            }
        }

        bool all_outputs = true;
        if (const char * all_env = std::getenv("ORNITH_TRACE_ALL_OUTPUTS")) {
            all_outputs = std::string(all_env) != "0";
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool output = all_outputs || i + 1 == tokens.size();
            common_batch_add(batch, tokens[i], static_cast<llama_pos>(i), {0}, output);
        }
        const int decode_status = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (decode_status != 0) {
            throw std::runtime_error("llama_decode failed with status " +
                                     std::to_string(decode_status));
        }

        write_trace(output_env, params, tokens, std::move(state));
        LOG("captured %zu tokens to %s\n", tokens.size(), output_env);
    } catch (const std::exception & error) {
        LOG_ERR("trace failure: %s\n", error.what());
        llama_backend_free();
        return 2;
    }

    llama_backend_free();
    return 0;
}

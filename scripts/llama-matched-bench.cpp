// Matched teacher-forced workload for shared_activation_bench ... forced.
// Link against the installed llama/ggml libraries; no sampling or tokenization.
#include <llama.h>
#include <ggml-backend.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

// Diagnostic only: observing nodes splits the graph and inhibits fusion.
// These synchronized wall spans locate expensive operations, not production
// GPU kernel durations. Leave LLAMA_MATCHED_PROFILE unset for throughput.
struct Profile {
    bool measured = false;
    std::string phase;
    std::chrono::steady_clock::time_point start;
    std::map<std::string, std::pair<int, double>> spans;
};
static bool observe(ggml_tensor * t, bool ask, void * data) {
    auto & p = *static_cast<Profile *>(data);
    if (ask) {
        if (!p.measured || p.phase == "seed" || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW ||
            t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) return false;
        p.start = std::chrono::steady_clock::now();
    } else {
        // Strip the layer suffix to accumulate the same operation across layers.
        std::string name = ggml_get_name(t);
        auto suffix = name.find_last_of('-');
        if (suffix != std::string::npos) name.resize(suffix);
        auto & span = p.spans[p.phase + ":" + ggml_op_name(t->op) + ":" + name];
        ++span.first;
        span.second += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p.start).count();
    }
    return true;
}

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void decode(llama_context * ctx, std::vector<llama_token> & tokens) {
    check(llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0, "llama_decode failed");
    llama_synchronize(ctx);
}
static std::vector<float> logits(llama_context * ctx, int n) {
    auto p = llama_get_logits_ith(ctx, -1);
    check(p != nullptr, "missing logits");
    std::vector<float> out(p, p + n);
    check(std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }), "nonfinite logits");
    return out;
}
template<class T> static void array(const std::vector<T> & v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}
int main(int argc, char ** argv) try {
    check(argc == 4, "MODEL PP ROUNDS");
    int pp = std::stoi(argv[2]), rounds = std::stoi(argv[3]);
    check(pp > 0 && pp <= 512 && rounds > 0, "PP must be 1..512, ROUNDS > 0");
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto model = llama_model_load_from_file(argv[1], mp); check(model, "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = pp + 33; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 16; cp.n_threads_batch = 16;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    Profile profile;
    if (std::getenv("LLAMA_MATCHED_PROFILE")) { cp.cb_eval = observe; cp.cb_eval_user_data = &profile; }
    auto ctx = llama_init_from_model(model, cp); check(ctx, "context creation failed");
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto token = [vocab](int i) { return (llama_token)(((i + 32) * 733 + 17) % std::min(vocab, 30000)); };
    std::vector<llama_token> prompt(pp), inputs(32), one(1, 1);
    for (int i = 0; i < pp; ++i) prompt[i] = token(i);
    for (int i = 0; i < 32; ++i) inputs[i] = token(i);
    std::vector<double> prefill, generation;
    std::vector<float> prefill_logits, decode_logits;
    using clock = std::chrono::steady_clock;
    for (int r = 0; r <= rounds; ++r) {
        profile.measured = r > 0; profile.phase = "prefill";
        llama_memory_clear(llama_get_memory(ctx), true);
        auto t = clock::now(); decode(ctx, prompt);
        double sec = std::chrono::duration<double>(clock::now() - t).count();
        if (r) prefill.push_back(pp / sec);
        prefill_logits = logits(ctx, vocab);
        llama_memory_clear(llama_get_memory(ctx), true);
        profile.phase = "seed";
        one[0] = 1; decode(ctx, one);
        profile.phase = "decode";
        t = clock::now();
        for (auto input : inputs) { one[0] = input; decode(ctx, one); }
        sec = std::chrono::duration<double>(clock::now() - t).count();
        if (r) generation.push_back(32 / sec);
        decode_logits = logits(ctx, vocab);
    }
    std::cout << std::setprecision(12)
        << "{\"pp\":" << pp << ",\"rounds\":" << rounds
        << ",\"warmup_rounds\":1,\"decode_context_tokens\":1,\"decode_tokens\":32"
        << ",\"decode_kind\":\"forced_full_logits\",\"kv_format\":\"f16\",\"prefill_tokens_per_second\":";
    array(prefill); std::cout << ",\"decode_tokens_per_second\":"; array(generation);
    std::cout << ",\"forced_decode_inputs\":"; array(inputs);
    std::cout << ",\"prefill_logits\":"; array(prefill_logits);
    std::cout << ",\"decode_final_logits\":"; array(decode_logits);
    std::cout << ",\"profile\":[";
    bool first = true;
    for (const auto & span : profile.spans) {
        if (!first) std::cout << ','; first = false;
        std::cout << "{\"op\":\"" << span.first << "\",\"calls\":" << span.second.first
            << ",\"synchronized_wall_ms\":" << span.second.second << '}';
    }
    std::cout << ']';
    std::cout << "}\n";
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }

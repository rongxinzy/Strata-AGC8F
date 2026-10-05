// tools/test_stage_dense_scope.cpp - host regression for the opt-in STRATA_STAGE_DENSE_SCOPE rule.
//
// The expectations below are written as LITERAL answers (name -> owned/skipped), not by re-running the
// header's own logic: the 48-layer eight-way split of the real model, its boundary layers, the globals
// that must survive on every stage, the base-skip union, the exact-"1" env gate, the overflow and grammar
// refusals, and the invalid-range fail-safe. Standalone: clang++ -std=c++17 -Iinclude this_file.cpp
#include "strata/core/stage_dense_scope.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;
static void check_failed(const std::string& msg, const char* file, int line) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++failures;
}
#define CHECK(cond, msg) do { if (!(cond)) check_failed((msg), __FILE__, __LINE__); } while (0)

using strata::core::dense_scope_env_on;
using strata::core::dense_scope_layer_of;
using strata::core::dense_scope_owned;
using strata::core::dense_scope_range_ok;
using strata::core::dense_scope_stage_skip;

// The real split shape: 48 layers, eight devices, six layers each. CUDA0 runs [0, 6) on its FULL pool
// (the integration never scopes the main device); CUDA1..7 run [6,12) .. [42,48) and are the scoping
// targets. Three dense families present on every layer, the layer-1 PLE key, the four non-blk globals.
static std::vector<std::string> real_rows() {
    std::vector<std::string> names;
    for (int64_t l = 0; l < 48; ++l) {
        names.push_back("blk." + std::to_string(l) + ".attn_qkv.weight");
        names.push_back("blk." + std::to_string(l) + ".ssm_out.weight");
        names.push_back("blk." + std::to_string(l) + ".ffn_down_shexp.weight");
    }
    names.push_back("blk.1.ple_key.weight");
    names.push_back("token_embd.weight");
    names.push_back("output.weight");
    names.push_back("output_hc_0.weight");
    names.push_back("per_layer_token_embd.weight");
    return names;   // 149 rows
}

int main() {
    // ---- grammar: what parses as blk.<L>. and what does not (all failures mean OWNED)
    CHECK(dense_scope_layer_of("blk.0.attn_qkv.weight") == 0, "layer 0 parses");
    CHECK(dense_scope_layer_of("blk.1.ple_key.weight") == 1, "PLE key is layer 1");
    CHECK(dense_scope_layer_of("blk.47.ffn_down_shexp.weight") == 47, "layer 47 parses");
    CHECK(dense_scope_layer_of("blk.1weight") == -1, "no dot after the digits");
    CHECK(dense_scope_layer_of("blk..weight") == -1, "no digits");
    CHECK(dense_scope_layer_of("blk.-1.weight") == -1, "negative layer");
    CHECK(dense_scope_layer_of("blk.abc.weight") == -1, "non-numeric layer");
    CHECK(dense_scope_layer_of("blk.5") == -1, "dot required");
    CHECK(dense_scope_layer_of("blk.5.") == -1, "empty suffix stays owned");
    CHECK(dense_scope_layer_of("blk.") == -1, "prefix alone");
    CHECK(dense_scope_layer_of("") == -1, "empty name");
    CHECK(dense_scope_layer_of("token_embd.weight") == -1, "global");
    CHECK(dense_scope_layer_of("output.weight") == -1, "global");
    CHECK(dense_scope_layer_of("output_hc_0.weight") == -1, "global");
    CHECK(dense_scope_layer_of("blkx.5.weight") == -1, "blk prefix must be exact");
    CHECK(dense_scope_layer_of("xxblk.5.weight") == -1, "prefix must be at the start");

    // ---- overflow: the digit run is checked BEFORE it wraps, and a refusal means OWNED
    CHECK(dense_scope_layer_of("blk.9223372036854775807.weight") == 9223372036854775807LL, "INT64_MAX parses");
    CHECK(dense_scope_layer_of("blk.9223372036854775808.weight") == -1, "INT64_MAX + 1 refuses");
    CHECK(dense_scope_layer_of("blk.99999999999999999999.weight") == -1, "20 digits refuses");
    CHECK(dense_scope_layer_of("blk.05.weight") == -1, "leading zero is not canonical");
    CHECK(dense_scope_layer_of("blk.0.weight") == 0, "a single zero is layer 0");
    CHECK(dense_scope_owned("blk.9223372036854775808.weight", 6, 12), "overflow row stays owned");
    CHECK(dense_scope_owned("blk.05.weight", 6, 12), "leading-zero row stays owned");

    // ---- the real 48-layer, 8-stage partition: CUDA1..7 own [6,12) .. [42,48)
    const std::vector<std::string> rows = real_rows();
    const std::set<std::string> all(rows.begin(), rows.end());
    CHECK(all.size() == 149, "48*3 layer rows + PLE key + 4 globals = 149");
    for (int64_t stage = 1; stage <= 7; ++stage) {
        const int64_t lb = stage * 6, le = lb + 6;
        const std::string tag = "stage [" + std::to_string(lb) + "," + std::to_string(le) + ")";
        std::set<std::string> sk;
        std::string why;
        CHECK(dense_scope_stage_skip(all, {}, lb, le, sk, why), tag + " scopes");
        CHECK(why.empty(), tag + " scopes without a complaint");
        CHECK(sk.size() == 126, tag + ": 149 rows minus 18 owned layer rows minus PLE minus 4 globals = 126 skipped");
        // literal boundary answers, one layer either side of each edge, all three families
        for (const char* family : {".attn_qkv.weight", ".ssm_out.weight", ".ffn_down_shexp.weight"}) {
            CHECK(sk.count("blk." + std::to_string(lb - 1) + family) == 1, tag + " skips the layer below");
            CHECK(sk.count("blk." + std::to_string(lb) + family) == 0, tag + " keeps its first layer");
            CHECK(sk.count("blk." + std::to_string(le - 1) + family) == 0, tag + " keeps its last layer");
            if (le < 48)   // stage 7's upper neighbour is layer 48, past the end of the model
                CHECK(sk.count("blk." + std::to_string(le) + family) == 1, tag + " skips the layer above");
        }
        CHECK(sk.count("blk.1.ple_key.weight") == 0, tag + " retains PLE on every stage");
        CHECK(sk.count("token_embd.weight") == 0, tag + " keeps token_embd");
        CHECK(sk.count("output.weight") == 0, tag + " keeps output");
        CHECK(sk.count("output_hc_0.weight") == 0, tag + " keeps output_hc");
        CHECK(sk.count("per_layer_token_embd.weight") == 0, tag + " keeps per_layer_token_embd");
    }
    // the main device range also retains PLE
    {
        std::set<std::string> sk;
        std::string why;
        CHECK(dense_scope_stage_skip(all, {}, 0, 6, sk, why), "main range scopes");
        CHECK(sk.count("blk.1.ple_key.weight") == 0, "main range keeps the layer-1 PLE key");
        CHECK(sk.count("blk.6.attn_qkv.weight") == 1, "main range skips layer 6");
    }

    // ---- invalid ranges: refuse to scope (empty out, a reason), and own everything per-name
    const std::vector<std::pair<int64_t, int64_t>> bad = {{-1, 6}, {6, 6}, {7, 3}, {0, -5}, {-3, -1}};
    for (const auto& r : bad) {
        std::set<std::string> sk;
        std::string why;
        CHECK(!dense_scope_stage_skip(all, {"token_embd.weight"}, r.first, r.second, sk, why),
              "an invalid range refuses to scope");
        CHECK(sk.empty(), "a refused scope leaves the skip set empty (full pool)");
        CHECK(!why.empty(), "a refused scope says why");
        CHECK(dense_scope_owned("blk.9.attn_qkv.weight", r.first, r.second),
              "an invalid range owns every row (NativeDense default [-1,-1) behaves as before)");
    }
    CHECK(!dense_scope_range_ok(-1, -1), "the no-scope default is not a range");
    CHECK(dense_scope_owned("blk.47.attn_qkv.weight", -1, -1), "default range owns layer 47");

    // ---- base-skip union: native-served rows stay skipped even when their layer is owned
    {
        const std::set<std::string> base = {"token_embd.weight", "blk.7.attn_qkv.weight", "blk.5.attn_qkv.weight"};
        std::set<std::string> sk;
        std::string why;
        CHECK(dense_scope_stage_skip(all, base, 6, 12, sk, why), "union scopes");
        CHECK(sk.count("token_embd.weight") == 1, "an owned global in the base skip stays skipped");
        CHECK(sk.count("blk.7.attn_qkv.weight") == 1, "an owned layer's base-skipped row stays skipped");
        CHECK(sk.count("blk.5.attn_qkv.weight") == 1, "an unowned row is skipped regardless");
        CHECK(sk.count("blk.7.ssm_out.weight") == 0, "an owned row outside the base skip is kept");
        CHECK(sk.size() == 128, "126 unowned + token_embd + blk.7.attn_qkv (blk.5.attn_qkv already unowned)");
    }

    // ---- the env gate: exactly "1", nothing else
    CHECK(dense_scope_env_on("1"), "exactly 1 enables");
    const char* off_values[] = {"", "0", "01", "1x", "10", " 1", "1 ", "true", "2", "-1", "11", "1\n"};
    for (const char* v : off_values) {
        const std::string shown = v[0] ? "\"" + std::string(v) + "\"" : "empty";
        CHECK(!dense_scope_env_on(v), shown + " must NOT enable the scope");
    }
    CHECK(!dense_scope_env_on(nullptr), "unset must NOT enable the scope");

    if (failures == 0) std::printf("stage_dense_scope_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}

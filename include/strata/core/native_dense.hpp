#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With a [scope_lb, scope_le) stage range, projections of layers outside the range are validated but
    /// not uploaded; an unparseable name is uploaded (fail-safe, same rule as the canonical scope in
    /// strata/core/stage_dense_scope.hpp). The defaulted [-1, -1) range uploads everything, as before.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t scope_lb = -1, int64_t scope_le = -1);
    /// Frees the uploaded matrices and the shared Q8_1 scratch, leaving the object exactly as constructed
    /// (the destructor stays correct on a reset object). CALLER ORDERING: the old WeightRefs' native_data /
    /// native_q8_1 fields dangle from this point until the caller replaces the table and re-runs load - no
    /// kernel, graph or captured context may still reference them. The boot-time caller runs this before any
    /// session, head, verifier or prefill graph exists, and stops the boot if the scoped re-load fails.
    bool reset(std::string& err);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
};
} // namespace strata::core

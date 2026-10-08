// include/strata/core/stage_dense_scope.hpp - which dense weight rows ONE layer-split stage must own.
//
// A split stage runs layers [lb, le) and nothing else: every runtime consumer of a `blk.<L>.*` row is
// layer-indexed (LayerView::name, include/strata/core/layout.hpp), an earlier stage's verifier returns
// before the head (src/core/verify.cpp), and the embedding paths run only where `hand_in_ == nullptr`
// (src/prefill/prefill.cpp, src/core/verify.cpp). Non-blk globals (token_embd, output, output_hc_*,
// per_layer_token_embd) are NOT layer-indexed, so this filter keeps every one of them on every stage
// rather than reasoning about which device dereferences which global.
//
// Skipped rows keep their metadata with `data == nullptr` (WeightTable::load, src/core/weights.cpp),
// so `find()` lookups that only read shapes (`output.weight->ne1` at generate.cpp, NativeDense::load's
// "absent from canonical table" check) still pass on a scoped stage.
//
// FAIL-SAFE BY CONSTRUCTION: anything this file cannot prove owned by the stage - a name it cannot
// parse, a number that overflows, a range it cannot validate - is OWNED. A wrong extra row costs VRAM;
// a wrongly dropped row costs correctness. The one deliberate exception is a layer index that parses
// cleanly but sits outside [lb, le): that is the rule doing its job, not a doubt.
#pragma once

#include <cstdint>
#include <limits>
#include <set>
#include <string>

namespace strata::core {

/// The opt-in is EXACTLY the string "1": "1x", "10", "01", " 1", "1 " and "" are all OFF, and so is an
/// unset variable. A prefix test ("v[0] == '1'") would silently enable the scope on typos and on any
/// future "1<suffix>" value, which is the wrong direction for a memory-layout change.
inline bool dense_scope_env_on(const char* value) { return value != nullptr && std::string(value) == "1"; }

/// The layer index of a canonical row name, or -1 for a global or a name this cannot parse.
/// "blk." + digits + "." only; `blk.1weight`, `blk..weight`, `blk.-1.weight` and `blk.abc.weight` are
/// all -1 (owned). A digit run that would overflow int64 is -1 (owned), checked BEFORE each multiply
/// rather than detected afterwards, when the silent wrap has already happened. "blk.05.weight" is not a
/// canonical name either (the pack writes no leading zeros) and is treated as unparseable: owned.
inline int64_t dense_scope_layer_of(const std::string& name) {
    if (name.compare(0, 4, "blk.") != 0) return -1;
    size_t i = 4;
    if (i >= name.size() || name[i] < '0' || name[i] > '9') return -1;
    if (name[i] == '0' && i + 1 < name.size() && name[i + 1] >= '0' && name[i + 1] <= '9') return -1;
    int64_t v = 0;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
        const int64_t d = name[i] - '0';
        if (v > (std::numeric_limits<int64_t>::max() - d) / 10) return -1;  // overflow: keep the row
        v = v * 10 + d;
    }
    if (i + 1 >= name.size() || name[i] != '.') return -1;
    return v;
}

/// True when [lb, le) is a usable stage range (the filter may run at all).
inline bool dense_scope_range_ok(int64_t lb, int64_t le) { return lb >= 0 && le > lb; }

/// True when the row's BYTES must be on the stage. Globals and unparseable names are always owned.
/// An invalid range (also the [-1, -1) default, which means "no scope") owns everything, so
/// NativeDense::load's defaulted parameters behave exactly as before this filter existed.
inline bool dense_scope_owned(const std::string& name, int64_t lb, int64_t le) {
    if (!dense_scope_range_ok(lb, le)) return true;
    const int64_t l = dense_scope_layer_of(name);
    return l < 0 || (l >= lb && l < le);
}

/// The native dense uploads use THE SAME rule (`NativeDense::load` passes its tensor names here): one
/// ownership decision per name, so the canonical skip set and the native upload filter cannot disagree
/// about a layer, and an unparseable native name is uploaded rather than dropped.

/// The stage's canonical skip set: `all` minus what the stage owns, UNIONED with `base_skip` (the
/// native-pack/native-head skips, which apply on every device regardless of stage - a base-skipped row
/// stays skipped even when its layer is owned, because its bytes are served natively). A name both
/// base-skipped and unowned appears once; `std::set` makes that automatic.
///
/// Returns false ONLY for an invalid range, with `why` set and an EMPTY out set: the caller then keeps
/// the full pool it already loaded and carries on, exactly as with the option off. Allocation or load
/// failures are NOT this function's business; the caller stops the boot on those.
inline bool dense_scope_stage_skip(const std::set<std::string>& all, const std::set<std::string>& base_skip,
                                   int64_t lb, int64_t le, std::set<std::string>& out, std::string& why) {
    out.clear();
    why.clear();
    if (!dense_scope_range_ok(lb, le)) {
        why = "dense scope: the stage range [" + std::to_string(lb) + "," + std::to_string(le) + ") is not a range";
        return false;
    }
    out = base_skip;
    for (const std::string& n : all)
        if (!dense_scope_owned(n, lb, le)) out.insert(n);
    return true;
}

}  // namespace strata::core

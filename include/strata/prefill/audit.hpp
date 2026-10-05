// Host-only helpers for diagnostic fingerprints; no buffers proportional to state size.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::prefill::detail {
inline constexpr uint64_t audit_seed = 1469598103934665603ull; // same seed as generate STATE_HASH
inline constexpr size_t audit_scratch_bytes = 65536;
inline uint64_t audit_fnv(const uint8_t* bytes, size_t n, uint64_t hash = audit_seed) {
    for (size_t i = 0; i < n; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
    return hash;
}
template<class Read>
bool audit_bytes(uint64_t bytes, uint64_t& hash, std::array<uint8_t, audit_scratch_bytes>& scratch,
                 Read&& read, uint64_t* aggregate = nullptr) {
    for (uint64_t at = 0; at < bytes;) {
        const size_t n = (size_t) std::min<uint64_t>(scratch.size(), bytes - at);
        if (!read(at, scratch.data(), n)) return false;
        hash = audit_fnv(scratch.data(), n, hash);
        if (aggregate) *aggregate = audit_fnv(scratch.data(), n, *aggregate);
        at += n;
    }
    return true;
}
// A downstream stage receives the final chunk of its parent, not the full n.
// Apply once per boundary; valid even when the legacy path has unequal chunks.
inline int64_t audit_last_chunk(int64_t input_n, int64_t chunk) {
    return input_n > 0 && chunk > 0 ? (input_n - 1) % chunk + 1 : 0;
}
} // namespace strata::prefill::detail

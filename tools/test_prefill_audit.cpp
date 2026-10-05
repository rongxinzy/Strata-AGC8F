#include "strata/prefill/audit.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using namespace strata::prefill::detail;
int main() {
    std::vector<uint8_t> data(200003);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) ((i * 37 + i / 19) & 255);
    std::array<uint8_t, audit_scratch_bytes> scratch;
    for (size_t bytes : {size_t(0), size_t(1), size_t(40960), size_t(65536), size_t(65537), data.size()}) {
        uint64_t hash = audit_seed, aggregate = audit_seed;
        size_t copied = 0, calls = 0, largest = 0;
        assert(audit_bytes(bytes, hash, scratch, [&](uint64_t at, uint8_t* host, size_t n) {
            assert(at == copied && n <= 65536 && at + n <= bytes);
            std::memcpy(host, data.data() + at, n);
            copied += n; ++calls; largest = std::max(largest, n);
            return true;
        }, &aggregate));
        assert(hash == audit_fnv(data.data(), bytes) && aggregate == hash && copied == bytes);
        assert(calls == (bytes + 65535) / 65536);
        std::cout << "fingerprint bytes=" << bytes << " copies=" << calls << " largest=" << largest << "\n";
    }
    uint64_t hash = audit_seed;
    int calls = 0;
    assert(!audit_bytes(data.size(), hash, scratch, [&](uint64_t, uint8_t*, size_t) { ++calls; return false; }));
    assert(calls == 1 && hash == audit_seed);
    assert(audit_last_chunk(0, 1024) == 0);
    assert(audit_last_chunk(1, 1024) == 1);
    assert(audit_last_chunk(1024, 1024) == 1024);
    assert(audit_last_chunk(1025, 1024) == 1);
    assert(audit_last_chunk(16383, 1024) == 1023);
    assert(audit_last_chunk(audit_last_chunk(1023, 1024), 512) == 511);
    assert(audit_last_chunk(audit_last_chunk(2050, 1024), 512) == 2);
    assert(audit_last_chunk(1, 0) == 0);
    std::cout << "copy failure stops hashing; full/partial/empty/unequal-stage last rows match\n";
}

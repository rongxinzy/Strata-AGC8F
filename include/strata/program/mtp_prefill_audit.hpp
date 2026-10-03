// Host-only serializer for diagnostic MTP KV, not a performance or accuracy test.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace strata::program::mtp_audit {

constexpr size_t copy_limit = 65536;
constexpr uint64_t hash_seed = 1469598103934665603ull;

struct Geometry {
    int64_t kv_heads = 0, head_dim = 0, page_size = 0;
    int64_t n_pages = 0, n_slots = 0, max_cells = 0;
};
struct Request {
    int device = -1;
    int64_t cell0 = 0, rows = 0, first_needed = 0;
    bool batched = false;
};
struct Range { int64_t begin = 0, end = 0; };

inline bool active_range(const Request& r, int64_t max_cells, Range& out, std::string& err) {
    if (r.cell0 < 0 || r.rows < 0 || max_cells < 0 || r.cell0 > max_cells || r.rows > max_cells - r.cell0) {
        err = "MTP audit: invalid cell range";
        return false;
    }
    out.end = r.cell0 + r.rows;
    out.begin = std::min(out.end, std::max(r.cell0, std::max<int64_t>(0, r.first_needed)));
    return true;
}

// The finite values of binary16 are exact in double, including signed zero.
inline double half_value(uint16_t bits) {
    const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
    double value;
    if (exponent == 31) value = fraction ? std::numeric_limits<double>::quiet_NaN()
                                       : std::numeric_limits<double>::infinity();
    else if (exponent == 0) value = std::ldexp((double) fraction, -24);
    else value = std::ldexp((double) (1024 + fraction), exponent - 25);
    return (bits & 0x8000) ? -value : value;
}

struct Stats {
    uint64_t hash = hash_seed, elements = 0, finite = 0, nan = 0, inf = 0, negative_zero = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity(), max_abs = 0;
    void add(uint16_t bits) {
        // Hash exactly the canonical little-endian bytes written below.
        hash = (hash ^ (uint8_t) bits) * 1099511628211ull;
        hash = (hash ^ (uint8_t) (bits >> 8)) * 1099511628211ull;
        ++elements;
        const double x = half_value(bits);
        if (std::isnan(x)) ++nan;
        else if (std::isinf(x)) ++inf;
        else {
            ++finite;
            minimum = std::min(minimum, x); maximum = std::max(maximum, x);
            max_abs = std::max(max_abs, std::fabs(x));
            if (bits == 0x8000) ++negative_zero;
        }
    }
};

inline void json_stats(std::ostream& f, const Stats& s) {
    f << "{\"bytes\":" << s.elements * 2 << ",\"elements\":" << s.elements
      << ",\"finite\":" << s.finite << ",\"nan\":" << s.nan << ",\"inf\":" << s.inf
      << ",\"negative_zero\":" << s.negative_zero << ",\"min\":";
    if (s.finite) f << s.minimum; else f << "null";
    f << ",\"max\":"; if (s.finite) f << s.maximum; else f << "null";
    f << ",\"max_abs\":" << s.max_abs << ",\"hash\":\"" << std::hex << std::setw(16)
      << std::setfill('0') << s.hash << std::dec << "\"}";
}

// read(key, element_offset, host, elements, err) copies from one physical pool.
// table is only the logical page interval starting at table_first. No unwritten
// rows, page padding, or cells earlier than first_needed are read or serialized.
template<class Reader>
bool write(const std::filesystem::path& root, uint64_t sequence, const Geometry& g, const Request& r,
           int64_t table_first, const std::vector<int32_t>& table, Reader read, std::string& err) {
    Range range;
    if (!active_range(r, g.max_cells, range, err)) return false;
    if (root.empty() || g.kv_heads <= 0 || g.head_dim <= 0 || g.page_size <= 0 || g.n_pages <= 0 ||
        g.n_slots <= 0 || g.head_dim > (int64_t) (copy_limit / 2) ||
        g.kv_heads > (int64_t) (copy_limit / 2) / g.head_dim) {
        err = "MTP audit: invalid geometry or output directory"; return false;
    }
    const uint64_t row_elements = (uint64_t) g.kv_heads * g.head_dim;
    // Bound every physical offset, not only the size of each D2H copy.
    const uint64_t limit = (std::numeric_limits<uint64_t>::max)() / 2;
    if ((uint64_t) g.n_slots > limit / row_elements ||
        (uint64_t) g.page_size > limit / ((uint64_t) g.n_slots * row_elements)) {
        err = "MTP audit: physical pool extent overflow"; return false;
    }
    const int64_t first_page = range.begin / g.page_size;
    const int64_t last_page = range.begin < range.end ? (range.end - 1) / g.page_size + 1 : first_page;
    if (last_page > g.n_pages || table_first != first_page ||
        (uint64_t) (last_page - first_page) != table.size()) {
        err = "MTP audit: page table coverage differs from active interval"; return false;
    }
    std::set<int32_t> used;
    for (int32_t page : table) {
        if (page < 0 || page >= g.n_slots || !used.insert(page).second) {
            err = "MTP audit: missing, aliased, or out-of-range physical page"; return false;
        }
    }
    try {
        std::filesystem::create_directories(root);
        const auto directory = root / ("prefill-" + std::to_string(sequence));
        if (!std::filesystem::create_directory(directory)) {
            err = "MTP audit: refusing to overwrite an existing chunk directory"; return false;
        }
        const auto k_tmp = directory / "k.fp16le.partial", v_tmp = directory / "v.fp16le.partial";
        std::ofstream k(k_tmp, std::ios::binary), v(v_tmp, std::ios::binary);
        k.exceptions(std::ios::badbit | std::ios::failbit); v.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<uint16_t, copy_limit / 2> buffer{};
        std::vector<uint8_t> raw((size_t) g.head_dim * 2);
        Stats stats[2];
        const int64_t block_rows = (int64_t) (copy_limit / (row_elements * 2));
        for (int pool = 0; pool < 2; ++pool) {
            std::ostream& f = pool == 0 ? (std::ostream&) k : (std::ostream&) v;
            for (int64_t cell = range.begin; cell < range.end;) {
                const int64_t logical = cell / g.page_size, within = cell % g.page_size;
                const int64_t rows = std::min({block_rows, g.page_size - within, range.end - cell});
                const uint64_t physical = (uint64_t) table[(size_t) (logical - first_page)];
                for (int64_t head = 0; head < g.kv_heads; ++head) {
                    const uint64_t at = ((physical * g.kv_heads + head) * g.page_size + within) * g.head_dim;
                    if (!read(pool == 0, at, buffer.data() + head * rows * g.head_dim,
                              (size_t) (rows * g.head_dim), err)) return false;
                }
                // Convert the physical head-major page layout to [cell][head][dimension].
                for (int64_t row = 0; row < rows; ++row) for (int64_t head = 0; head < g.kv_heads; ++head) {
                    const uint16_t* source = buffer.data() + (head * rows + row) * g.head_dim;
                    for (int64_t dim = 0; dim < g.head_dim; ++dim) {
                        const uint16_t bits = source[dim];
                        raw[(size_t) dim * 2] = (uint8_t) bits; raw[(size_t) dim * 2 + 1] = (uint8_t) (bits >> 8);
                        stats[pool].add(bits);
                    }
                    f.write((const char*) raw.data(), (std::streamsize) raw.size());
                }
                cell += rows;
            }
        }
        k.close(); v.close();
        std::filesystem::rename(k_tmp, directory / "k.fp16le");
        std::filesystem::rename(v_tmp, directory / "v.fp16le");
        std::ofstream meta(directory / "metadata.json.partial");
        meta.exceptions(std::ios::badbit | std::ios::failbit);
        meta << std::setprecision(17) << "{\n\"schema\":1,\"complete\":true,\"device\":" << r.device
             << ",\"sequence\":" << sequence << ",\"path\":\"" << (r.batched ? "batched" : "token")
             << "\",\"cell0\":" << r.cell0 << ",\"rows\":" << r.rows
             << ",\"first_needed\":" << r.first_needed << ",\"active_begin\":" << range.begin
             << ",\"active_end\":" << range.end << ",\"active_rows\":" << range.end - range.begin
             << ",\"kv_heads\":" << g.kv_heads << ",\"head_dim\":" << g.head_dim
             << ",\"page_size\":" << g.page_size << ",\"max_cells\":" << g.max_cells
             << ",\"n_pages\":" << g.n_pages << ",\"n_slots\":" << g.n_slots
             << ",\"table_first_page\":" << first_page << ",\"physical_pages\":[";
        for (size_t i = 0; i < table.size(); ++i) meta << (i ? "," : "") << table[i];
        meta << "],\"dtype\":\"fp16-le\",\"layout\":\"cell,kv_head,head_dim\","
                "\"copy_limit\":65536,\"hash_algorithm\":\"strata-fnv1a64\","
                "\"k_file\":\"k.fp16le\",\"v_file\":\"v.fp16le\",\"k\":";
        json_stats(meta, stats[0]); meta << ",\"v\":"; json_stats(meta, stats[1]); meta << "\n}\n";
        meta.close();
        std::filesystem::rename(directory / "metadata.json.partial", directory / "metadata.json");
        return true;
    } catch (const std::exception& e) {
        err = std::string("MTP audit output: ") + e.what(); return false;
    }
}

} // namespace strata::program::mtp_audit

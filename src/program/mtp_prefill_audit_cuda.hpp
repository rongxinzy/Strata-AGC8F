#pragma once

#include "strata/program/mtp_prefill_audit.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/qsa.hpp"
#include <cstdio>

namespace strata::program {

// Called only by the owning Prefill's synchronous on_chunk after successful KV
// production. The flag is independent of MULTI, so the old path can be audited.
inline bool audit_mtp_prefill(core::MtpDrafter& mtp, const core::ModelGeometry& g, int64_t rows, int64_t cell0,
                              bool batched, const std::string& directory, uint64_t sequence, std::string& err) {
    const core::OnDevice on(mtp.device());
    const core::QsaState& st = mtp.kv_state();
    if (st.kv_mode != 0 || st.kv_hybrid || st.kv_int8 || st.kv_q4 || st.kv_rot ||
        st.k_pool == nullptr || st.v_pool == nullptr || st.page_table == nullptr) {
        err = "MTP audit: only unrotated resident paged fp16 K/V is supported"; return false;
    }
    if (!mtp.idle(err)) return false;
    // draft_kv has already synchronized its Prefill stream; the legacy prefill
    // has synchronized MTP's stream. This diagnostic barrier also rejects any
    // asynchronous device failure before extracting the written interval.
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) { err = std::string("MTP audit sync: ") + cudaGetErrorString(sync); return false; }
    const auto shape = strata::kernels::qsa_real_shapes();
    const mtp_audit::Geometry geometry{g.n_head_kv, g.head_dim, shape.page_size, st.n_pages, st.n_slots, st.max_cells};
    const mtp_audit::Request request{mtp.device(), cell0, rows, mtp.first_needed(), batched};
    mtp_audit::Range range;
    if (!mtp_audit::active_range(request, st.max_cells, range, err)) return false;
    if (geometry.page_size <= 0) { err = "MTP audit: invalid page size"; return false; }
    const int64_t first = range.begin / geometry.page_size;
    const int64_t last = range.begin < range.end ? (range.end - 1) / geometry.page_size + 1 : first;
    if (first < 0 || last > st.n_pages) { err = "MTP audit: logical page outside allocation"; return false; }
    std::vector<int32_t> table((size_t) (last - first));
    // Bound diagnostic page-table copies too, even for unusually large contexts.
    for (size_t at = 0; at < table.size();) {
        const size_t count = std::min(table.size() - at, mtp_audit::copy_limit / sizeof(int32_t));
        const auto status = cudaMemcpy(table.data() + at, st.page_table + first + at,
                                       count * sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) { err = std::string("MTP audit table: ") + cudaGetErrorString(status); return false; }
        at += count;
    }
    const bool ok = mtp_audit::write(directory, sequence, geometry, request, first, table,
        [&](bool key, uint64_t offset, uint16_t* host, size_t elements, std::string& e) {
            const auto status = cudaMemcpy(host, (key ? st.k_pool : st.v_pool) + offset,
                                           elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            if (status == cudaSuccess) return true;
            e = std::string("MTP audit K/V copy: ") + cudaGetErrorString(status); return false;
        }, err);
    if (ok) std::fprintf(stderr, "strata MTP prefill audit: sequence=%llu device=%d cell0=%lld rows=%lld "
                               "active=%lld:%lld path=%s complete=1\n", (unsigned long long) sequence,
                               mtp.device(), (long long) cell0, (long long) rows, (long long) range.begin,
                               (long long) range.end, batched ? "batched" : "token");
    return ok;
}

} // namespace strata::program

#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
namespace strata::core {
struct Ep24Plan {
    std::array<std::array<int, 3>, 8> layers{}, helpers{};
};
template<class BlobBytes> Ep24Plan ep24_plan(BlobBytes bytes) {
    Ep24Plan p;
    for (int st = 0; st < 8; ++st) {
        std::array<int, 6> order{};
        for (int i = 0; i < 6; ++i) order[i] = st * 6 + i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return bytes(a) != bytes(b) ? bytes(a) > bytes(b) : a < b;
        });
        for (int i = 0; i < 3; ++i) p.layers[st][i] = order[i];
        // Start clockwise after this stage, skip GPU0 and the original stage.
        int n = 0;
        for (int step = 1; n < 3; ++step) {
            const int dev = (st + step) % 8;
            if (dev && dev != st) p.helpers[st][n++] = dev;
        }
    }
    return p;
}
} // namespace strata::core

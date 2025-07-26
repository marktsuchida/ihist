#pragma once

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

#include <array>
#include <cstddef>
#include <cstdint>

#if defined _WIN32 || defined __CYGWIN__
#ifdef IHIST_BUILDING_SHARED
#define IHIST_PUBLIC __declspec(dllexport)
#else
#define IHIST_PUBLIC __declspec(dllimport)
#endif
#else
#ifdef IHIST_BUILDING_SHARED
#define IHIST_PUBLIC __attribute__((visibility("default")))
#else
#define IHIST_PUBLIC
#endif
#endif

#ifdef _MSC_VER
#define IHIST_RESTRICT __restrict
#else
#define IHIST_RESTRICT __restrict__
#endif

namespace ihist {

inline void hist8_naive(std::uint8_t const *IHIST_RESTRICT data,
                        std::size_t size,
                        std::uint32_t *IHIST_RESTRICT histogram) {
    for (std::size_t i = 0; i < size; ++i) {
        ++histogram[data[i]];
    }
}

template <std::size_t P>
void hist8_striped(std::uint8_t const *IHIST_RESTRICT data, std::size_t size,
                   std::uint32_t *IHIST_RESTRICT histogram) {
    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    static constexpr std::size_t NLANES = 1 << P;
    static constexpr std::size_t NBINS = 1 << 8;

    std::array<std::array<std::uint32_t, NBINS>, NLANES> lanes{};

    // The #pragma unroll makes a big difference on Apple M1. TODO Others?
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        ++lanes[i & (NLANES - 1)][data[i]];
    }

    for (std::size_t bin = 0; bin < NBINS; ++bin) {
        std::uint32_t sum = 0;
        for (std::size_t n = 0; n < NLANES; ++n) {
            sum += lanes[n][bin];
        }
        histogram[bin] += sum;
    }
}

namespace internal {

template <auto Hist8>
void hist8_mt(std::uint8_t const *IHIST_RESTRICT data, std::size_t size,
              std::uint32_t *IHIST_RESTRICT histogram) {
    tbb::combinable<std::array<uint32_t, 256>> local_hists(
        [] { return std::array<uint32_t, 256>{}; });

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, 1 << 14),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          Hist8(data + r.begin(), r.size(), h.data());
                      });

    local_hists.combine_each([&](const std::array<uint32_t, 1 << 8> &h) {
        for (int bin = 0; bin < (1 << 8); ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

} // namespace internal

inline void hist8_naive_mt(std::uint8_t const *IHIST_RESTRICT data,
                           std::size_t size,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist8_mt<hist8_naive>(data, size, histogram);
}

template <std::size_t P>
void hist8_striped_mt(std::uint8_t const *IHIST_RESTRICT data,
                      std::size_t size,
                      std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist8_mt<hist8_striped<P>>(data, size, histogram);
}

} // namespace ihist

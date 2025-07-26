#pragma once

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

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

template <typename T> constexpr std::size_t bin_count() {
    return 1 << (8 * sizeof(T));
}

template <typename T>
void hist_naive(T const *IHIST_RESTRICT data, std::size_t size,
                std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t i = 0; i < size; ++i) {
        ++histogram[data[i]];
    }
}

template <typename T, std::size_t P>
void hist_striped(T const *IHIST_RESTRICT data, std::size_t size,
                  std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(std::is_unsigned_v<T>);
    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    static constexpr std::size_t NLANES = 1 << P;
    static constexpr std::size_t NBINS = bin_count<T>();

    assert(size < std::numeric_limits<std::uint32_t>::max());

    std::vector<std::uint32_t> hists(NLANES * NBINS, 0);

    // The #pragma unroll makes a big difference on Apple M1. TODO Others?
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        auto const lane = i & (NLANES - 1);
        ++hists[lane * NBINS + data[i]];
    }

    for (std::size_t bin = 0; bin < NBINS; ++bin) {
        std::uint32_t sum = 0;
        for (std::size_t lane = 0; lane < NLANES; ++lane) {
            sum += hists[lane * NBINS + bin];
        }
        histogram[bin] += sum;
    }
}

namespace internal {

template <typename T, auto Hist>
void hist_mt(T const *IHIST_RESTRICT data, std::size_t size,
             std::uint32_t *IHIST_RESTRICT histogram) {
    static constexpr std::size_t NBINS = bin_count<T>();

    tbb::combinable<std::array<uint32_t, NBINS>> local_hists(
        [] { return std::array<uint32_t, NBINS>{}; });

    // TODO Grain size is empirical on Apple M1; investigate elsewhere.
    // u8 -> 1 << 14
    // u16 -> 1 << 17
    static constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          Hist(data + r.begin(), r.size(), h.data());
                      });

    local_hists.combine_each([&](const std::array<uint32_t, NBINS> &h) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

} // namespace internal

template <typename T>
void hist_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                   std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_naive<T>>(data, size, histogram);
}

template <typename T, std::size_t P>
void hist_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                     std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_striped<T, P>>(data, size, histogram);
}

} // namespace ihist

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

namespace internal {

// N.B. This will discard high bits of value, effectively wrapping around if
// values exceed the (BITS + LO_BIT)-bit range. To histogram data while leaving
// out samples that exceed the expected range, masking should be used (once we
// support it).
// (Checking for such high bits, and skipping the histogramming of such values,
// can cause a ~10% overhead, at least under some conditions.)
template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
constexpr auto bin_index(T value) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    static_assert(BITS > 0);
    static_assert(BITS <= 8 * sizeof(T));
    static_assert(LO_BIT < 8 * sizeof(T));
    static_assert(BITS + LO_BIT <= 8 * sizeof(T));
    constexpr T MASK = ((1uLL << BITS) - 1) << LO_BIT;
    return (value & MASK) >> LO_BIT;
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_naive(T const *IHIST_RESTRICT data, std::size_t size,
                std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());
    for (std::size_t i = 0; i < size; ++i) {
        ++histogram[internal::bin_index<T, BITS, LO_BIT>(data[i])];
    }
}

template <typename T, std::size_t P, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_striped(T const *IHIST_RESTRICT data, std::size_t size,
                  std::uint32_t *IHIST_RESTRICT histogram) {
    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    constexpr std::size_t NLANES = 1 << P;
    constexpr std::size_t NBINS = 1 << BITS;

    assert(size < std::numeric_limits<std::uint32_t>::max());

    std::vector<std::uint32_t> hists(NLANES * NBINS, 0);

    // The #pragma unroll makes a big difference on Apple M1. TODO Others?
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        auto const lane = i & (NLANES - 1);
        ++hists[lane * NBINS + internal::bin_index<T, BITS, LO_BIT>(data[i])];
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

template <typename T, auto Hist, unsigned BITS = 8 * sizeof(T)>
void hist_mt(T const *IHIST_RESTRICT data, std::size_t size,
             std::uint32_t *IHIST_RESTRICT histogram) {
    constexpr std::size_t NBINS = 1 << BITS;

    tbb::combinable<std::array<uint32_t, NBINS>> local_hists(
        [] { return std::array<uint32_t, NBINS>{}; });

    // TODO Grain size is empirical on Apple M1; investigate elsewhere.
    // u8 -> 1 << 14
    // u16 -> 1 << 17
    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
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

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                   std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_naive<T, BITS, LO_BIT>, BITS>(data, size,
                                                            histogram);
}

template <typename T, std::size_t P, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                     std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_striped<T, P, BITS, LO_BIT>, BITS>(data, size,
                                                                 histogram);
}

} // namespace ihist

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

// Value to bin index, disregarding high bits.
template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
constexpr auto bin_index(T value) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    constexpr auto TYPE_BITS = 8 * sizeof(T);
    constexpr auto SAMP_BITS = BITS + LO_BIT;
    static_assert(BITS > 0);
    static_assert(BITS <= TYPE_BITS);
    static_assert(LO_BIT < TYPE_BITS);
    static_assert(SAMP_BITS <= TYPE_BITS);

    constexpr T SIGNIF_MASK = (1uLL << BITS) - 1;
    std::size_t const bin = (value >> LO_BIT) & SIGNIF_MASK;
    return bin;
}

// Value to bin index, but masked by matching high bits.
template <typename T, unsigned BITS, unsigned LO_BIT = 0>
constexpr auto bin_index_himask(T value, T hi_mask) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    constexpr auto TYPE_BITS = 8 * sizeof(T);
    constexpr auto SAMP_BITS = BITS + LO_BIT;
    static_assert(BITS > 0);
    static_assert(BITS <= TYPE_BITS);
    static_assert(LO_BIT < TYPE_BITS);
    static_assert(SAMP_BITS < TYPE_BITS);

    constexpr T SIGNIF_MASK = (1uLL << BITS) - 1;
    std::size_t const bin = (value >> LO_BIT) & SIGNIF_MASK;

    constexpr auto HI_BITS = TYPE_BITS - SAMP_BITS;
    constexpr T HI_BITS_MASK = (1uLL << HI_BITS) - 1;
    constexpr std::size_t MASKED_BIN = 1uLL << BITS;
    auto const hi_bits = (value >> SAMP_BITS) & HI_BITS_MASK;
    bool const keep = hi_bits == hi_mask;

    return keep ? bin : MASKED_BIN;
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive(T const *IHIST_RESTRICT data, std::size_t size,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());
    for (std::size_t i = 0; i < size; ++i) {
        ++histogram[internal::bin_index<T, BITS, LO_BIT>(data[i])];
    }
}

template <typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_naive(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());
    constexpr std::size_t MASKED_BIN = 1uLL << (BITS + LO_BIT);
    for (std::size_t i = 0; i < size; ++i) {
        auto const bin =
            internal::bin_index_himask<T, BITS, LO_BIT>(data[i], hi_mask);
        if (bin != MASKED_BIN) {
            ++histogram[bin];
        }
    }
}

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_filtered_naive(T const *IHIST_RESTRICT data, std::size_t size,
                         std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_naive<T, BITS, LO_BIT>(data, size, histogram);
    } else {
        hist_himask_naive<T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped(T const *IHIST_RESTRICT data, std::size_t size,
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

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_striped(T const *IHIST_RESTRICT data, std::size_t size,
                         T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(P < 16, "P should not be too big");
    constexpr std::size_t NLANES = 1 << P;
    constexpr std::size_t NBINS = 1 << BITS;
    constexpr std::size_t MASKED_BIN = 1uLL << (BITS + LO_BIT);

    assert(size < std::numeric_limits<std::uint32_t>::max());

    std::vector<std::uint32_t> hists(NLANES * NBINS, 0);

#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        auto const lane = i & (NLANES - 1);
        auto const bin =
            internal::bin_index_himask<T, BITS, LO_BIT>(data[i], hi_mask);
        if (bin != MASKED_BIN) {
            ++hists[lane * NBINS + bin];
        }
    }

    for (std::size_t bin = 0; bin < NBINS; ++bin) {
        std::uint32_t sum = 0;
        for (std::size_t lane = 0; lane < NLANES; ++lane) {
            sum += hists[lane * NBINS + bin];
        }
        histogram[bin] += sum;
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_filtered_striped(T const *IHIST_RESTRICT data, std::size_t size,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_striped<P, T, BITS, LO_BIT>(data, size, histogram);
    } else {
        hist_himask_striped<P, T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

namespace internal {

template <typename T, auto Hist, unsigned BITS = 8 * sizeof(T)>
void hist_mt(T const *IHIST_RESTRICT data, std::size_t size,
             std::uint32_t *IHIST_RESTRICT histogram) {
    constexpr std::size_t NBINS = 1 << BITS;
    using hist_array = std::array<std::uint32_t, NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    // TODO Grain size is empirical on Apple M1; investigate elsewhere.
    // u8 -> 1 << 14
    // u16 -> 1 << 17
    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          Hist(data + r.begin(), r.size(), h.data());
                      });

    local_hists.combine_each([&](const hist_array &h) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

template <typename T, auto HistHiMasked, unsigned BITS>
void hist_himask_mt(T const *IHIST_RESTRICT data, std::size_t size, T hi_mask,
                    std::uint32_t *IHIST_RESTRICT histogram) {
    constexpr std::size_t NBINS = 1 << BITS;
    using hist_array = std::array<std::uint32_t, NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          HistHiMasked(data + r.begin(), r.size(), hi_mask,
                                       h.data());
                      });

    local_hists.combine_each([&](const hist_array &h) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_unfiltered_naive<T, BITS, LO_BIT>, BITS>(
        data, size, histogram);
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<T, hist_unfiltered_striped<P, T, BITS, LO_BIT>, BITS>(
        data, size, histogram);
}

template <typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                          T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_himask_mt<T, hist_himask_naive<T, BITS, LO_BIT>, BITS>(
        data, size, hi_mask, histogram);
}

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                            T hi_mask,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_himask_mt<T, hist_himask_striped<P, T, BITS, LO_BIT>, BITS>(
        data, size, hi_mask, histogram);
}

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_filtered_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_naive_mt<T, BITS, LO_BIT>(data, size, histogram);
    } else {
        hist_himask_naive_mt<T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_filtered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_striped_mt<P, T, BITS, LO_BIT>(data, size, histogram);
    } else {
        hist_himask_striped_mt<P, T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

} // namespace ihist

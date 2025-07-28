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

template <typename T, unsigned BITS, unsigned LO_BIT, bool HI_MASK>
void hist_naive_impl(T const *IHIST_RESTRICT data, std::size_t size, T hi_mask,
                     std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());
    constexpr std::size_t MASKED_BIN = 1uLL << BITS;
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (HI_MASK) {
            auto const bin =
                bin_index_himask<T, BITS, LO_BIT>(data[i], hi_mask);
            if (bin != MASKED_BIN) {
                ++histogram[bin];
            }

        } else {
            auto const bin = internal::bin_index<T, BITS, LO_BIT>(data[i]);
            ++histogram[bin];
        }
    }
}

template <typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_naive(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    hist_naive_impl<T, BITS, LO_BIT, true>(data, size, hi_mask, histogram);
}

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT,
          bool HI_MASK>
void hist_striped_impl(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    constexpr std::size_t NLANES = 1 << P;
    constexpr std::size_t NBINS = 1 << BITS;
    constexpr std::size_t MASKED_BIN = 1uLL << BITS;

    assert(size < std::numeric_limits<std::uint32_t>::max());

    std::vector<std::uint32_t> hists(NLANES * NBINS, 0);

#if defined(__APPLE__) && defined(__aarch64__)
// Improves performance on Apple M1:
#pragma unroll
#endif
    for (std::size_t i = 0; i < size; ++i) {
        auto const lane = i & (NLANES - 1);
        if constexpr (HI_MASK) {
            auto const bin =
                bin_index_himask<T, BITS, LO_BIT>(data[i], hi_mask);
            if (bin != MASKED_BIN) {
                ++hists[lane * NBINS + bin];
            }
        } else {
            auto const bin = internal::bin_index<T, BITS, LO_BIT>(data[i]);
            ++hists[lane * NBINS + bin];
        }
    }

    // Clang and GCC typically vectorize this loop. Making the lane the outer
    // loop (and bin the inner loop) appears to have either a negative or
    // negligible/unpredictable effect on performance, for most cases.
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
    hist_striped_impl<P, T, BITS, LO_BIT, true>(data, size, hi_mask,
                                                histogram);
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive(T const *IHIST_RESTRICT data, std::size_t size,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_naive_impl<T, BITS, LO_BIT, false>(data, size, 0,
                                                      histogram);
}

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_filtered_naive(T const *IHIST_RESTRICT data, std::size_t size,
                         std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_naive<T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_naive<T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped(T const *IHIST_RESTRICT data, std::size_t size,
                             std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_striped_impl<P, T, BITS, LO_BIT, false>(data, size, 0, histogram);
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_filtered_striped(T const *IHIST_RESTRICT data, std::size_t size,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_striped<P, T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_striped<P, T, BITS, LO_BIT>(data, size, 0,
                                                          histogram);
    }
}

template <std::size_t R, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_radixmultipass(T const *IHIST_RESTRICT data,
                                    std::size_t size,
                                    std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(BITS + LO_BIT <= 8 * sizeof(T));
    static_assert(LO_BIT < 8 * sizeof(T));
    static_assert(R > 0);
    static_assert(R < BITS);
    constexpr std::size_t N_HI_BINS = 1 << (BITS - R);
    assert(size <= std::numeric_limits<std::uint32_t>::max());

    for (std::size_t hi = 0; hi < N_HI_BINS; ++hi) {
        // TODO Selectable func. Consider passing in a class template via
        // template template parameter, from which the function (pointer) can
        // be obtained.
        internal::hist_himask_striped<3, T, R, LO_BIT>(
            data, size, hi, histogram + hi * (1 << R));
    }
}

// Split BITS into (BITS - R) high bits and R low bits; partition the latter by
// the former.
template <std::size_t R, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_radixpartition(T const *IHIST_RESTRICT data,
                                    std::size_t size,
                                    std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(BITS + LO_BIT <= 8 * sizeof(T));
    static_assert(LO_BIT < 8 * sizeof(T));
    static_assert(R > 0);
    static_assert(R < BITS);
    constexpr std::size_t N_HI_BINS = 1 << (BITS - R);
    assert(size <= std::numeric_limits<std::uint32_t>::max());

    using hihist_array = std::array<std::uint32_t, N_HI_BINS>;
    hihist_array const hi_counts = [&] {
        hihist_array h{};
        // TODO Selectable func
        hist_unfiltered_striped<3, T, BITS - R, R + LO_BIT>(data, size,
                                                            h.data());
        return h;
    }();

    hihist_array const offsets = [&] {
        hihist_array o;
        o[0] = 0;
        for (std::size_t i = 1; i < N_HI_BINS; ++i) {
            o[i] = o[i - 1] + hi_counts[i - 1];
        }
        return o;
    }();

    constexpr T LO_MASK = (1 << R) - 1;
    static_assert(R <= 16); // For now, at least.
    using U = std::conditional_t<(R <= 8), std::uint8_t, std::uint16_t>;

    auto partitioned = std::vector<U>(size);
    hihist_array cur_offsets(offsets);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        T const hi = data[i] >> (R + LO_BIT);
        T const lo = (data[i] >> LO_BIT) & LO_MASK;
        partitioned[cur_offsets[hi]++] = lo;
    }

    for (std::size_t hi = 0; hi < N_HI_BINS; ++hi) {
        std::size_t const start = offsets[hi];
        std::size_t const count = hi_counts[hi];
        if (count == 0) // TODO Use fallback wrapper instead
            continue;
        // TODO Selectable func
        hist_unfiltered_striped<3, U, R>(partitioned.data() + start, count,
                                         histogram + hi * (1 << R));
    }
}

namespace internal {

template <typename T> struct first_parameter;

template <typename R, typename First, typename... Args>
struct first_parameter<R(First, Args...)> {
    using type = First;
};

template <typename R, typename First, typename... Args>
struct first_parameter<R (*)(First, Args...)> {
    using type = First;
};

template <typename T>
using first_parameter_t = typename first_parameter<T>::type;

template <auto Hist, typename T, unsigned BITS = 8 * sizeof(T)>
void hist_mt(T const *IHIST_RESTRICT data, std::size_t size,
             std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(Hist)>>);
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

template <auto HistHimask, typename T, unsigned BITS = 8 * sizeof(T)>
void hist_himask_mt(T const *IHIST_RESTRICT data, std::size_t size, T hi_mask,
                    std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(HistHimask)>>);
    constexpr std::size_t NBINS = 1 << BITS;
    using hist_array = std::array<std::uint32_t, NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          HistHimask(data + r.begin(), r.size(), hi_mask,
                                     h.data());
                      });

    local_hists.combine_each([&](const hist_array &h) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

template <auto Hist, typename T,
          std::size_t CHUNK_SIZE = (1 << 12) / sizeof(T)>
void hist_chunked(T const *IHIST_RESTRICT data, std::size_t size,
                  std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(Hist)>>);
    for (std::size_t i = 0; i < size; i += CHUNK_SIZE) {
        auto const siz = std::min(CHUNK_SIZE, size - i);
        Hist(data + i, siz, histogram);
    }
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<hist_unfiltered_naive<T, BITS, LO_BIT>, T, BITS>(
        data, size, histogram);
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<hist_unfiltered_striped<P, T, BITS, LO_BIT>, T, BITS>(
        data, size, histogram);
}

namespace internal {

template <typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                          T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    hist_himask_mt<hist_himask_naive<T, BITS, LO_BIT>, T, BITS>(
        data, size, hi_mask, histogram);
}

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                            T hi_mask,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    hist_himask_mt<hist_himask_striped<P, T, BITS, LO_BIT>, T, BITS>(
        data, size, hi_mask, histogram);
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_filtered_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_naive_mt<T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_naive_mt<T, BITS, LO_BIT>(data, size, 0,
                                                        histogram);
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_filtered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_striped_mt<P, T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_striped_mt<P, T, BITS, LO_BIT>(data, size, 0,
                                                             histogram);
    }
}

} // namespace ihist

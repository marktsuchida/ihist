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
#if defined(__APPLE__) && defined(__aarch64__)
    bool const keep = hi_bits == hi_mask;
    return keep ? bin : MASKED_BIN;
#else
    if (hi_bits == hi_mask) {
        return bin;
    }
    return MASKED_BIN;
#endif
}

template <typename T, unsigned BITS, unsigned LO_BIT, bool HI_MASK,
          std::size_t STRIDE, std::size_t... COMPONENT_OFFSETS>
void hist_naive_impl(T const *IHIST_RESTRICT data, std::size_t size, T hi_mask,
                     std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({COMPONENT_OFFSETS...}) < STRIDE);

    constexpr std::size_t NBINS = 1uLL << BITS;
    constexpr std::size_t NCOMPONENTS = sizeof...(COMPONENT_OFFSETS);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        COMPONENT_OFFSETS...};

    for (std::size_t j = 0; j < size; ++j) {
        auto const i = j * STRIDE;
        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            if constexpr (HI_MASK) {
                auto const bin = bin_index_himask<T, BITS, LO_BIT>(
                    data[i + offset], hi_mask);
                if (bin != NBINS) {
                    ++histogram[c * NBINS + bin];
                }
            } else {
                auto const bin = bin_index<T, BITS, LO_BIT>(data[i + offset]);
                ++histogram[c * NBINS + bin];
            }
        }
    }
}

template <typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_naive(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    hist_naive_impl<T, BITS, LO_BIT, true, 1, 0>(data, size, hi_mask,
                                                 histogram);
}

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT,
          bool HI_MASK, std::size_t STRIDE, std::size_t... COMPONENT_OFFSETS>
void hist_striped_impl(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    static_assert(std::max({COMPONENT_OFFSETS...}) < STRIDE);

    constexpr std::size_t NSTRIPES = 1 << P;
    constexpr std::size_t NBINS = 1 << BITS;
    constexpr std::size_t NCOMPONENTS = sizeof...(COMPONENT_OFFSETS);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        COMPONENT_OFFSETS...};

    std::vector<std::uint32_t> stripes(NSTRIPES * NCOMPONENTS * NBINS, 0);

    // TODO NUNROLL (or its dependence on NCOMPONENTS) should be injected
    // together with P as a strategy parameter.
#if defined(__APPLE__) && defined(__aarch64__)
    constexpr std::size_t NUNROLL = sizeof(T) > 2 ? 1 : sizeof(T) > 1 ? 4 : 16;
#elif defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) ||       \
    defined(__amd64) || defined(_M_X64)
    constexpr std::size_t NUNROLL = sizeof(T) > 1 ? 1 : 4;
#else
    constexpr std::size_t NUNROLL = 1;
#endif

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), NUNROLL / NCOMPONENTS);
    std::size_t const n_blocks = size / BLOCKSIZE;
    std::size_t const n_remainder = size % BLOCKSIZE;

    for (std::size_t block = 0; block < n_blocks; ++block) {
        // We pre-compute all the bin indices for the block here, which
        // facilitates experimenting with potential optimizations, but the
        // compiler may well interleave this with the bin increments below.
        std::array<std::size_t, BLOCKSIZE> bins;
        for (std::size_t n = 0; n < BLOCKSIZE * STRIDE; ++n) {
            if constexpr (HI_MASK) {
                bins[n] = bin_index_himask<T, BITS, LO_BIT>(
                    data[block * BLOCKSIZE * STRIDE + n], hi_mask);
            } else {
                bins[n] = bin_index<T, BITS, LO_BIT>(
                    data[block * BLOCKSIZE * STRIDE + n]);
            }
        }

        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                auto const bin = bins[k * STRIDE + offset];
                if constexpr (HI_MASK) {
#if defined(__APPLE__) && defined(__aarch64__)
                    auto const b = bin % NBINS;
                    stripes[(stripe * NCOMPONENTS + c) * NBINS + b] +=
                        (bin != NBINS);
#else
                    if (bin != NBINS) {
                        ++stripes[(stripe * NCOMPONENTS + c) * NBINS + bin];
                    }
#endif
                } else {
                    ++stripes[(stripe * NCOMPONENTS + c) * NBINS + bin];
                }
            }
        }
    }

    // Leftover
    for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
        auto const offset = offsets[c];
        for (std::size_t k = 0; k < n_remainder; ++k) {
            auto const i = (n_blocks * BLOCKSIZE + k) * STRIDE + offset;
            auto const stripe = (n_blocks * BLOCKSIZE + k) % NSTRIPES;
            if constexpr (HI_MASK) {
                auto const bin =
                    bin_index_himask<T, BITS, LO_BIT>(data[i], hi_mask);
                auto const b = bin % NBINS;
                stripes[(stripe * NCOMPONENTS + c) * NBINS + b] +=
                    (bin != NBINS);
            } else {
                auto const bin = bin_index<T, BITS, LO_BIT>(data[i]);
                ++stripes[(stripe * NCOMPONENTS + c) * NBINS + bin];
            }
        }
    }

    // Clang and GCC typically vectorize this loop.
    for (std::size_t bin = 0; bin < NCOMPONENTS * NBINS; ++bin) {
        std::uint32_t sum = 0;
        for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
            sum += stripes[stripe * NCOMPONENTS * NBINS + bin];
        }
        histogram[bin] += sum;
    }
}

template <std::size_t P, typename T, unsigned BITS, unsigned LO_BIT = 0>
void hist_himask_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                            T hi_mask,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    hist_striped_impl<P, T, BITS, LO_BIT, true, 1, 0>(data, size, hi_mask,
                                                      histogram);
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive_st(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_naive_impl<T, BITS, LO_BIT, false, 1, 0>(data, size, 0,
                                                            histogram);
}

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_filtered_naive_st(T const *IHIST_RESTRICT data, std::size_t size,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_naive_st<T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_naive<T, BITS, LO_BIT>(data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_striped_impl<P, T, BITS, LO_BIT, false, 1, 0>(data, size, 0,
                                                                 histogram);
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_filtered_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (BITS + LO_BIT == 8 * sizeof(T)) {
        hist_unfiltered_striped_st<P, T, BITS, LO_BIT>(data, size, histogram);
    } else {
        internal::hist_himask_striped_st<P, T, BITS, LO_BIT>(data, size, 0,
                                                             histogram);
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
    hist_himask_mt<hist_himask_striped_st<P, T, BITS, LO_BIT>, T, BITS>(
        data, size, hi_mask, histogram);
}

} // namespace internal

template <typename T, unsigned BITS = 8 * sizeof(T), unsigned LO_BIT = 0>
void hist_unfiltered_naive_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<hist_unfiltered_naive_st<T, BITS, LO_BIT>, T, BITS>(
        data, size, histogram);
}

template <std::size_t P, typename T, unsigned BITS = 8 * sizeof(T),
          unsigned LO_BIT = 0>
void hist_unfiltered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<hist_unfiltered_striped_st<P, T, BITS, LO_BIT>, T, BITS>(
        data, size, histogram);
}

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

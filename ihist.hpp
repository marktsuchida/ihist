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
template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0>
constexpr auto bin_index(T value) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    constexpr auto TYPE_BITS = 8 * sizeof(T);
    constexpr auto SAMP_BITS = Bits + LoBit;
    static_assert(Bits > 0);
    static_assert(Bits <= TYPE_BITS);
    static_assert(LoBit < TYPE_BITS);
    static_assert(SAMP_BITS <= TYPE_BITS);

    constexpr T SIGNIF_MASK = (1uLL << Bits) - 1;
    std::size_t const bin = (value >> LoBit) & SIGNIF_MASK;
    return bin;
}

// Value to bin index, but masked by matching high bits.
template <typename T, unsigned Bits, unsigned LoBit = 0>
constexpr auto bin_index_himask(T value, T hi_mask) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    constexpr auto TYPE_BITS = 8 * sizeof(T);
    constexpr auto SAMP_BITS = Bits + LoBit;
    static_assert(Bits > 0);
    static_assert(Bits <= TYPE_BITS);
    static_assert(LoBit < TYPE_BITS);
    static_assert(SAMP_BITS < TYPE_BITS);

    constexpr T SIGNIF_MASK = (1uLL << Bits) - 1;
    std::size_t const bin = (value >> LoBit) & SIGNIF_MASK;

    constexpr auto HI_BITS = TYPE_BITS - SAMP_BITS;
    constexpr T HI_BITS_MASK = (1uLL << HI_BITS) - 1;
    constexpr std::size_t MASKED_BIN = 1uLL << Bits;
    auto const hi_bits = (value >> SAMP_BITS) & HI_BITS_MASK;

    // Intel is faster with a predictable branch than with branchless. Since
    // our only current use of the himask is to filter for zero high bits, we
    // optimize for that case.  For Apple M1, branchless appears to be slightly
    // faster when Bits is low enough and appropriately striped.
    // TODO This choice should be injected as a strategy parameter.
#if defined(__APPLE__) && defined(__aarch64__)
    return (hi_bits == hi_mask) ? bin : MASKED_BIN;
#else
    if (hi_bits == hi_mask) {
        return bin;
    }
    return MASKED_BIN;
#endif
}

template <typename T, unsigned Bits, unsigned LoBit, bool UseHiMask,
          std::size_t Stride, std::size_t... ComponentOffsets>
void hist_unoptimized_impl(T const *IHIST_RESTRICT data, std::size_t size,
                           T hi_mask,
                           std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({ComponentOffsets...}) < Stride);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NCOMPONENTS = sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        ComponentOffsets...};

    for (std::size_t j = 0; j < size; ++j) {
        auto const i = j * Stride;
        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            if constexpr (UseHiMask) {
                auto const bin = bin_index_himask<T, Bits, LoBit>(
                    data[i + offset], hi_mask);
                if (bin != NBINS) {
                    ++histogram[c * NBINS + bin];
                }
            } else {
                auto const bin = bin_index<T, Bits, LoBit>(data[i + offset]);
                ++histogram[c * NBINS + bin];
            }
        }
    }
}

template <typename T, unsigned Bits, unsigned LoBit, std::size_t Stride,
          std::size_t... ComponentOffsets>
void hist_himask_unoptimized(T const *IHIST_RESTRICT data, std::size_t size,
                             T hi_mask,
                             std::uint32_t *IHIST_RESTRICT histogram) {
    hist_unoptimized_impl<T, Bits, LoBit, true, Stride, ComponentOffsets...>(
        data, size, hi_mask, histogram);
}

template <std::size_t P, typename T, unsigned Bits, unsigned LoBit,
          bool UseHiMask, std::size_t Stride, std::size_t... ComponentOffsets>
void hist_striped_impl(T const *IHIST_RESTRICT data, std::size_t size,
                       T hi_mask, std::uint32_t *IHIST_RESTRICT histogram) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    // 4 * 2^P needs to comfortably fit in L1D cache.
    static_assert(P < 16, "P should not be too big");
    static_assert(std::max({ComponentOffsets...}) < Stride);

    constexpr std::size_t NSTRIPES = 1 << P;
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NCOMPONENTS = sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        ComponentOffsets...};

    // Use extra bin for overflows (when UseHiMask).
    std::vector<std::uint32_t> stripes(NSTRIPES * NCOMPONENTS * (NBINS + 1));

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
        std::array<std::size_t, BLOCKSIZE * Stride> bins;
        for (std::size_t n = 0; n < BLOCKSIZE * Stride; ++n) {
            auto const i = block * BLOCKSIZE * Stride + n;
            if constexpr (UseHiMask) {
                bins[n] = bin_index_himask<T, Bits, LoBit>(data[i], hi_mask);
            } else {
                bins[n] = bin_index<T, Bits, LoBit>(data[i]);
            }
        }

        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                auto const bin = bins[k * Stride + offset];
                ++stripes[(stripe * NCOMPONENTS + c) * (NBINS + 1) + bin];
            }
        }
    }

    // Leftover
    for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
        auto const offset = offsets[c];
        for (std::size_t k = 0; k < n_remainder; ++k) {
            auto const i = (n_blocks * BLOCKSIZE + k) * Stride + offset;
            auto const stripe = (n_blocks * BLOCKSIZE + k) % NSTRIPES;
            auto const bin = [&] {
                if constexpr (UseHiMask) {
                    return bin_index_himask<T, Bits, LoBit>(data[i], hi_mask);
                } else {
                    return bin_index<T, Bits, LoBit>(data[i]);
                }
            }();
            ++stripes[(stripe * NCOMPONENTS + c) * (NBINS + 1) + bin];
        }
    }

    for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            std::uint32_t sum = 0;
            for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                sum += stripes[(stripe * NCOMPONENTS + c) * (NBINS + 1) + bin];
            }
            histogram[c * NBINS + bin] += sum;
        }
    }
}

template <std::size_t P, typename T, unsigned Bits, unsigned LoBit,
          std::size_t Stride, std::size_t... ComponentOffsets>
void hist_himask_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                            T hi_mask,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    hist_striped_impl<P, T, Bits, LoBit, true, Stride, ComponentOffsets...>(
        data, size, hi_mask, histogram);
}

} // namespace internal

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_unfiltered_unoptimized_st(T const *IHIST_RESTRICT data,
                                    std::size_t size,
                                    std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_unoptimized_impl<T, Bits, LoBit, false, Stride,
                                    Component0Offset, ComponentOffsets...>(
        data, size, 0, histogram);
}

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_filtered_unoptimized_st(T const *IHIST_RESTRICT data,
                                  std::size_t size,
                                  std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (Bits + LoBit == 8 * sizeof(T)) {
        hist_unfiltered_unoptimized_st<T, Bits, LoBit, Stride,
                                       Component0Offset, ComponentOffsets...>(
            data, size, histogram);
    } else {
        internal::hist_himask_unoptimized<
            T, Bits, LoBit, Stride, Component0Offset, ComponentOffsets...>(
            data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void hist_unfiltered_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_striped_impl<P, T, Bits, LoBit, false, Stride,
                                Component0Offset, ComponentOffsets...>(
        data, size, 0, histogram);
}

template <std::size_t P, typename T, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void hist_filtered_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (Bits + LoBit == 8 * sizeof(T)) {
        hist_unfiltered_striped_st<P, T, Bits, LoBit, Stride, Component0Offset,
                                   ComponentOffsets...>(data, size, histogram);
    } else {
        internal::hist_himask_striped_st<
            P, T, Bits, LoBit, Stride, Component0Offset, ComponentOffsets...>(
            data, size, 0, histogram);
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

template <auto Hist, typename T, unsigned Bits, std::size_t Stride,
          std::size_t NComponents>
void hist_mt(T const *IHIST_RESTRICT data, std::size_t size,
             std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(Hist)>>);
    constexpr std::size_t NBINS = 1 << Bits;
    using hist_array = std::array<std::uint32_t, NComponents * NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    // TODO Grain size is empirical on Apple M1, single-component; investigate
    // elsewhere.
    // u8 -> 1 << 14
    // u16 -> 1 << 17
    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          Hist(data + r.begin() * Stride, r.size(), h.data());
                      });

    local_hists.combine_each([&](const hist_array &h) {
        for (std::size_t bin = 0; bin < NComponents * NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

template <auto HistHimask, typename T, unsigned Bits, std::size_t Stride,
          std::size_t NComponents>
void hist_himask_mt(T const *IHIST_RESTRICT data, std::size_t size, T hi_mask,
                    std::uint32_t *IHIST_RESTRICT histogram) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(HistHimask)>>);
    constexpr std::size_t NBINS = 1 << Bits;
    using hist_array = std::array<std::uint32_t, NComponents * NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    // TODO Grain size (see non-himask variant).
    constexpr auto grain_size = 1 << (sizeof(T) > 1 ? 17 : 14);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](const tbb::blocked_range<std::size_t> &r) {
                          auto &h = local_hists.local();
                          HistHimask(data + r.begin() * Stride, r.size(),
                                     hi_mask, h.data());
                      });

    local_hists.combine_each([&](const hist_array &h) {
        for (std::size_t bin = 0; bin < NComponents * NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

template <typename T, unsigned Bits, unsigned LoBit, std::size_t Stride,
          std::size_t... ComponentOffsets>
void hist_himask_unoptimized_mt(T const *IHIST_RESTRICT data, std::size_t size,
                                T hi_mask,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    hist_himask_mt<
        hist_himask_unoptimized<T, Bits, LoBit, Stride, ComponentOffsets...>,
        T, Bits, Stride, sizeof...(ComponentOffsets)>(data, size, hi_mask,
                                                      histogram);
}

template <std::size_t P, typename T, unsigned Bits, unsigned LoBit,
          std::size_t Stride, std::size_t... ComponentOffsets>
void hist_himask_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                            T hi_mask,
                            std::uint32_t *IHIST_RESTRICT histogram) {
    hist_himask_mt<
        hist_himask_striped_st<P, T, Bits, LoBit, Stride, ComponentOffsets...>,
        T, Bits, Stride, sizeof...(ComponentOffsets)>(data, size, hi_mask,
                                                      histogram);
}

} // namespace internal

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_unfiltered_unoptimized_mt(T const *IHIST_RESTRICT data,
                                    std::size_t size,
                                    std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<
        hist_unfiltered_unoptimized_st<T, Bits, LoBit, Stride,
                                       Component0Offset, ComponentOffsets...>,
        T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(data, size,
                                                          histogram);
}

template <std::size_t P, typename T, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void hist_unfiltered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                                std::uint32_t *IHIST_RESTRICT histogram) {
    internal::hist_mt<
        hist_unfiltered_striped_st<P, T, Bits, LoBit, Stride, Component0Offset,
                                   ComponentOffsets...>,
        T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(data, size,
                                                          histogram);
}

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_filtered_unoptimized_mt(T const *IHIST_RESTRICT data,
                                  std::size_t size,
                                  std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (Bits + LoBit == 8 * sizeof(T)) {
        hist_unfiltered_unoptimized_mt<T, Bits, LoBit, Stride,
                                       Component0Offset, ComponentOffsets...>(
            data, size, histogram);
    } else {
        internal::hist_himask_unoptimized_mt<
            T, Bits, LoBit, Stride, Component0Offset, ComponentOffsets...>(
            data, size, 0, histogram);
    }
}

template <std::size_t P, typename T, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t Stride = 1,
          std::size_t Component0Offset = 0, std::size_t... ComponentOffsets>
void hist_filtered_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                              std::uint32_t *IHIST_RESTRICT histogram) {
    if constexpr (Bits + LoBit == 8 * sizeof(T)) {
        hist_unfiltered_striped_mt<P, T, Bits, LoBit, Stride, Component0Offset,
                                   ComponentOffsets...>(data, size, histogram);
    } else {
        internal::hist_himask_striped_mt<
            P, T, Bits, LoBit, Stride, Component0Offset, ComponentOffsets...>(
            data, size, 0, histogram);
    }
}

} // namespace ihist

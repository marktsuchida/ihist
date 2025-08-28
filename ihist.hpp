/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

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

struct tuning_parameters {
    // Number of separate histograms to iterate over (to tune for store-to-load
    // latency hiding vs spatial locality).
    std::size_t n_stripes = 1;

    // Approximate samples processed per main loop iteration (divided by
    // component count to determine pixels processed per iteration).
    std::size_t n_unroll = 1;
};

inline constexpr tuning_parameters untuned_parameters;

template <typename T, unsigned Bits>
constexpr tuning_parameters default_tuning_parameters{
#if defined(__APPLE__) && defined(__aarch64__)
    sizeof(T) > 1 ? 2 : 8, // TODO Tune the default
    sizeof(T) > 2   ? 1
    : sizeof(T) > 1 ? 4
                    : 16,
#elif defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) ||       \
    defined(__amd64) || defined(_M_X64)
    sizeof(T) > 1 ? 1 : 2, // TODO Tune the default
    sizeof(T) > 1 ? 1 : 4,
#endif
};

namespace internal {

// Value to bin index. If value is out of range, return 1 + max bin index.
template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0>
constexpr auto bin_index(T value) -> std::size_t {
    static_assert(std::is_unsigned_v<T>);
    constexpr auto TYPE_BITS = 8 * sizeof(T);
    constexpr auto SAMP_BITS = Bits + LoBit;
    static_assert(Bits > 0);
    static_assert(SAMP_BITS <= TYPE_BITS);

    std::size_t const bin = value >> LoBit;
    if constexpr (SAMP_BITS < TYPE_BITS) {
        constexpr std::size_t OVERFLOW_BIN = 1uLL << Bits;
        return value >> SAMP_BITS ? OVERFLOW_BIN : bin;
    } else {
        return bin;
    }
}

// Note that the first aligned index may be beyond the end of the buffer.
template <typename T, std::size_t Alignment>
constexpr auto first_aligned_index_impl(std::uintptr_t addr) -> std::size_t {
    static_assert((Alignment & (Alignment - 1)) == 0,
                  "Alignment must be a power of 2");
    if constexpr (Alignment <= alignof(T)) {
        return 0;
    } else {
        auto const aligned_addr = (addr + Alignment - 1) & ~(Alignment - 1);
        std::size_t const byte_offset = aligned_addr - addr;
        return byte_offset / sizeof(T);
    }
}

template <typename T, std::size_t Alignment>
constexpr auto first_aligned_index(T const *buffer) -> std::size_t {
    return first_aligned_index_impl<T, Alignment>(
        reinterpret_cast<std::uintptr_t>(buffer));
}

} // namespace internal

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_unoptimized_st(T const *IHIST_RESTRICT data, std::size_t size,
                         std::uint32_t *IHIST_RESTRICT histogram,
                         std::size_t = 0) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({Component0Offset, ComponentOffsets...}) < Stride);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        Component0Offset, ComponentOffsets...};

#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (std::size_t j = 0; j < size; ++j) {
        auto const i = j * Stride;
        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            auto const bin =
                internal::bin_index<T, Bits, LoBit>(data[i + offset]);
            if (bin != NBINS) {
                ++histogram[c * NBINS + bin];
            }
        }
    }
}

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void histxy_unoptimized_st(T const *IHIST_RESTRICT data, std::size_t width,
                           std::size_t height, std::size_t roi_x,
                           std::size_t roi_y, std::size_t roi_width,
                           std::size_t roi_height,
                           std::uint32_t *IHIST_RESTRICT histogram,
                           std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(roi_x + roi_width <= width);
    assert(roi_y + roi_height <= height);

    static_assert(std::max({Component0Offset, ComponentOffsets...}) < Stride);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        Component0Offset, ComponentOffsets...};

#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (std::size_t y = roi_y; y < roi_y + roi_height; ++y) {
#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
        for (std::size_t x = roi_x; x < roi_x + roi_width; ++x) {
            auto const j = x + y * width;
            auto const i = j * Stride;
            for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
                auto const offset = offsets[c];
                auto const bin =
                    internal::bin_index<T, Bits, LoBit>(data[i + offset]);
                if (bin != NBINS) {
                    ++histogram[c * NBINS + bin];
                }
            }
        }
    }
}

template <tuning_parameters const &Tuning, typename T,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_striped_st(T const *IHIST_RESTRICT data, std::size_t size,
                     std::uint32_t *IHIST_RESTRICT histogram,
                     std::size_t = 0) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({Component0Offset, ComponentOffsets...}) < Stride);

    constexpr std::size_t NSTRIPES =
        std::max(std::size_t(1), Tuning.n_stripes);
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        Component0Offset, ComponentOffsets...};

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    std::vector<std::uint32_t> stripes(NSTRIPES * NCOMPONENTS * STRIPE_LEN);

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), Tuning.n_unroll / NCOMPONENTS);
    constexpr std::size_t BLOCKSIZE_BYTES = BLOCKSIZE * Stride * sizeof(T);
    constexpr bool BLOCKSIZE_BYTES_IS_POWER_OF_2 =
        (BLOCKSIZE_BYTES & (BLOCKSIZE_BYTES - 1)) == 0;
    constexpr std::size_t BLOCK_ALIGNMENT =
        BLOCKSIZE_BYTES_IS_POWER_OF_2 ? BLOCKSIZE_BYTES : alignof(T);

    std::size_t const prolog_size = [&] {
        if constexpr (BLOCKSIZE_BYTES_IS_POWER_OF_2) {
            return std::min(
                size, internal::first_aligned_index<T, BLOCKSIZE_BYTES>(data));
        } else {
            return 0;
        }
    }();

    hist_unoptimized_st<T, Bits, LoBit, Stride, Component0Offset,
                        ComponentOffsets...>(data, prolog_size, histogram);

    std::size_t const size_after_prolog = size - prolog_size;
    if (size_after_prolog == 0) {
        return;
    }

    T const *block_data =
#if defined(__GNUC__) || defined(__clang__)
        (T const *)__builtin_assume_aligned(
#endif
            data + prolog_size * Stride
#if defined(__GNUC__) || defined(__clang__)
            ,
            BLOCK_ALIGNMENT)
#endif
        ;

    std::size_t const n_blocks = size_after_prolog / BLOCKSIZE;
    std::size_t const epilog_size = size_after_prolog % BLOCKSIZE;
    T const *epilog_data = block_data + n_blocks * BLOCKSIZE * Stride;

#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (std::size_t block = 0; block < n_blocks; ++block) {
        // We pre-compute all the bin indices for the block here, which
        // facilitates experimenting with potential optimizations, but the
        // compiler may well interleave this with the bin increments below.
        std::array<std::size_t, BLOCKSIZE * Stride> bins;
        for (std::size_t n = 0; n < BLOCKSIZE * Stride; ++n) {
            auto const i = block * BLOCKSIZE * Stride + n;
            bins[n] = internal::bin_index<T, Bits, LoBit>(block_data[i]);
        }

        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                auto const bin = bins[k * Stride + offset];
                ++stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
            }
        }
    }

    for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            std::uint32_t sum = 0;
            for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                sum += stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
            }
            histogram[c * NBINS + bin] += sum;
        }
    }

    hist_unoptimized_st<T, Bits, LoBit, Stride, Component0Offset,
                        ComponentOffsets...>(epilog_data, epilog_size,
                                             histogram);
}

template <tuning_parameters const &Tuning, typename T,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void histxy_striped_st(T const *IHIST_RESTRICT data, std::size_t width,
                       std::size_t height, std::size_t roi_x,
                       std::size_t roi_y, std::size_t roi_width,
                       std::size_t roi_height,
                       std::uint32_t *IHIST_RESTRICT histogram,
                       std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(roi_x + roi_width <= width);
    assert(roi_y + roi_height <= height);

    static_assert(std::max({Component0Offset, ComponentOffsets...}) < Stride);

    constexpr std::size_t NSTRIPES =
        std::max(std::size_t(1), Tuning.n_stripes);
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(ComponentOffsets);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{
        Component0Offset, ComponentOffsets...};

    // Simplify to single row if full-width.
    if (roi_width == width && height > 1) {
        return histxy_striped_st<Tuning, T, Bits, LoBit, Stride,
                                 Component0Offset, ComponentOffsets...>(
            data + roi_y * width * Stride, width * roi_height, 1, 0, 0,
            width * roi_height, 1, histogram);
    }

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    std::vector<std::uint32_t> stripes(NSTRIPES * NCOMPONENTS * STRIPE_LEN);

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), Tuning.n_unroll / NCOMPONENTS);
    std::size_t const n_blocks_per_row = roi_width / BLOCKSIZE;
    std::size_t const row_epilog_size = roi_width % BLOCKSIZE;

    for (std::size_t y = roi_y; y < roi_y + roi_height; ++y) {
        T const *row_data = data + (y * width + roi_x) * Stride;
        T const *row_epilog_data =
            row_data + n_blocks_per_row * BLOCKSIZE * Stride;
#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
        for (std::size_t block = 0; block < n_blocks_per_row; ++block) {
            std::array<std::size_t, BLOCKSIZE * Stride> bins;
            for (std::size_t n = 0; n < BLOCKSIZE * Stride; ++n) {
                auto const i = block * BLOCKSIZE * Stride + n;
                bins[n] = internal::bin_index<T, Bits, LoBit>(row_data[i]);
            }

            for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
                auto const offset = offsets[c];
                for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                    auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                    auto const bin = bins[k * Stride + offset];
                    ++stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
                }
            }
        }

        // Epilog goes straight to the final histogram, at least for now.
        hist_unoptimized_st<T, Bits, LoBit, Stride, Component0Offset,
                            ComponentOffsets...>(row_epilog_data,
                                                 row_epilog_size, histogram);
    }

    for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
        for (std::size_t bin = 0; bin < NBINS; ++bin) {
            std::uint32_t sum = 0;
            for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                sum += stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
            }
            histogram[c * NBINS + bin] += sum;
        }
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
             std::uint32_t *IHIST_RESTRICT histogram,
             std::size_t grain_size = 1) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(Hist)>>);
    constexpr std::size_t NBINS = 1 << Bits;
    using hist_array = std::array<std::uint32_t, NComponents * NBINS>;

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                      [&](tbb::blocked_range<std::size_t> const &r) {
                          auto &h = local_hists.local();
                          Hist(data + r.begin() * Stride, r.size(), h.data(),
                               0);
                      });

    local_hists.combine_each([&](hist_array const &h) {
        for (std::size_t bin = 0; bin < NComponents * NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

template <auto HistXY, typename T, unsigned Bits, std::size_t Stride,
          std::size_t NComponents>
void histxy_mt(T const *IHIST_RESTRICT data, std::size_t width,
               std::size_t height, std::size_t roi_x, std::size_t roi_y,
               std::size_t roi_width, std::size_t roi_height,
               std::uint32_t *IHIST_RESTRICT histogram,
               std::size_t grain_size) {
    static_assert(
        std::is_same_v<T const *, first_parameter_t<decltype(HistXY)>>);
    constexpr std::size_t NBINS = 1 << Bits;
    using hist_array = std::array<std::uint32_t, NComponents * NBINS>;

    auto const height_grain_size =
        std::max(std::size_t(1), grain_size / std::max(std::size_t(1), width));

    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, roi_height, height_grain_size),
        [&](tbb::blocked_range<std::size_t> const &r) {
            auto &h = local_hists.local();
            HistXY(data, width, height, roi_x, roi_y + r.begin(), roi_width,
                   r.size(), h.data(), 0);
        });

    local_hists.combine_each([&](hist_array const &h) {
        for (std::size_t bin = 0; bin < NComponents * NBINS; ++bin) {
            histogram[bin] += h[bin];
        }
    });
}

} // namespace internal

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_unoptimized_mt(T const *IHIST_RESTRICT data, std::size_t size,
                         std::uint32_t *IHIST_RESTRICT histogram,
                         std::size_t grain_size = 1) {
    internal::hist_mt<
        hist_unoptimized_st<T, Bits, LoBit, Stride, Component0Offset,
                            ComponentOffsets...>,
        T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(
        data, size, histogram, grain_size);
}

template <tuning_parameters const &Tuning, typename T,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void hist_striped_mt(T const *IHIST_RESTRICT data, std::size_t size,
                     std::uint32_t *IHIST_RESTRICT histogram,
                     std::size_t grain_size = 1) {
    internal::hist_mt<hist_striped_st<Tuning, T, Bits, LoBit, Stride,
                                      Component0Offset, ComponentOffsets...>,
                      T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(
        data, size, histogram, grain_size);
}

template <typename T, unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void histxy_unoptimized_mt(T const *IHIST_RESTRICT data, std::size_t width,
                           std::size_t height, std::size_t roi_x,
                           std::size_t roi_y, std::size_t roi_width,
                           std::size_t roi_height,
                           std::uint32_t *IHIST_RESTRICT histogram,
                           std::size_t grain_size = 1) {
    internal::histxy_mt<
        histxy_unoptimized_st<T, Bits, LoBit, Stride, Component0Offset,
                              ComponentOffsets...>,
        T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(
        data, width, height, roi_x, roi_y, roi_width, roi_height, histogram,
        grain_size);
}

template <tuning_parameters const &Tuning, typename T,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t Stride = 1, std::size_t Component0Offset = 0,
          std::size_t... ComponentOffsets>
void histxy_striped_mt(T const *IHIST_RESTRICT data, std::size_t width,
                       std::size_t height, std::size_t roi_x,
                       std::size_t roi_y, std::size_t roi_width,
                       std::size_t roi_height,
                       std::uint32_t *IHIST_RESTRICT histogram,
                       std::size_t grain_size = 1) {
    internal::histxy_mt<
        histxy_striped_st<Tuning, T, Bits, LoBit, Stride, Component0Offset,
                          ComponentOffsets...>,
        T, Bits, Stride, 1 + sizeof...(ComponentOffsets)>(
        data, width, height, roi_x, roi_y, roi_width, roi_height, histogram,
        grain_size);
}

} // namespace ihist

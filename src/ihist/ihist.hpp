/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "phys_core_count.hpp"

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef _MSC_VER
#define IHIST_RESTRICT __restrict
#else
#define IHIST_RESTRICT __restrict__
#endif

#ifdef _MSC_VER
#define IHIST_NOINLINE __declspec(noinline)
#else
#define IHIST_NOINLINE [[gnu::noinline]]
#endif

namespace ihist {

struct tuning_parameters {
    // Number of separate histograms to iterate over (to tune for store-to-load
    // latency hiding vs spatial locality).
    std::size_t n_stripes;

    // Approximate samples processed per main loop iteration (divided by
    // component count to determine pixels processed per iteration).
    std::size_t n_unroll;
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

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t SamplesPerPixel = 1,
          std::size_t Sample0Index = 0, std::size_t... SampleIndices>
/* not noinline */ void
hist_unoptimized_st(T const *IHIST_RESTRICT data,
                    std::uint8_t const *IHIST_RESTRICT mask, std::size_t size,
                    std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{Sample0Index,
                                                           SampleIndices...};

#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (std::size_t j = 0; j < size; ++j) {
        auto const i = j * SamplesPerPixel;
        if (!UseMask || mask[j]) {
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

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t SamplesPerPixel = 1,
          std::size_t Sample0Index = 0, std::size_t... SampleIndices>
/* not noinline */ void histxy_unoptimized_st(
    T const *IHIST_RESTRICT data, std::uint8_t const *IHIST_RESTRICT mask,
    std::size_t width, [[maybe_unused]] std::size_t height, std::size_t roi_x,
    std::size_t roi_y, std::size_t roi_width, std::size_t roi_height,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(roi_x + roi_width <= width);
    assert(roi_y + roi_height <= height);

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{Sample0Index,
                                                           SampleIndices...};

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
            auto const i = j * SamplesPerPixel;
            if (!UseMask || mask[j]) {
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
}

template <tuning_parameters const &Tuning, typename T, bool UseMask = false,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t SamplesPerPixel = 1, std::size_t Sample0Index = 0,
          std::size_t... SampleIndices>
IHIST_NOINLINE void
hist_striped_st(T const *IHIST_RESTRICT data,
                std::uint8_t const *IHIST_RESTRICT mask, std::size_t size,
                std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(size < std::numeric_limits<std::uint32_t>::max());

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NSTRIPES =
        std::max(std::size_t(1), Tuning.n_stripes);
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{Sample0Index,
                                                           SampleIndices...};

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    constexpr bool USE_STRIPES = NSTRIPES > 1 || STRIPE_LEN > NBINS;

    std::vector<std::uint32_t> stripes_storage;
    std::uint32_t *stripes = [&]() {
        if constexpr (USE_STRIPES) {
            stripes_storage.resize(NSTRIPES * NCOMPONENTS * STRIPE_LEN);
            return stripes_storage.data();
        } else {
            return histogram;
        }
    }();

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), Tuning.n_unroll);
    constexpr std::size_t BLOCKSIZE_BYTES =
        BLOCKSIZE * SamplesPerPixel * sizeof(T);
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

    hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel, Sample0Index,
                        SampleIndices...>(data, mask, prolog_size, histogram);

    std::size_t const size_after_prolog = size - prolog_size;
    if (size_after_prolog == 0) {
        return;
    }

    T const *blocks_data =
#if defined(__GNUC__) || defined(__clang__)
        (T const *)__builtin_assume_aligned(
#endif
            data + prolog_size * SamplesPerPixel
#if defined(__GNUC__) || defined(__clang__)
            ,
            BLOCK_ALIGNMENT)
#endif
        ;
    std::uint8_t const *blocks_mask = UseMask ? mask + prolog_size : nullptr;

    std::size_t const n_blocks = size_after_prolog / BLOCKSIZE;
    std::size_t const epilog_size = size_after_prolog % BLOCKSIZE;
    T const *epilog_data =
        blocks_data + n_blocks * BLOCKSIZE * SamplesPerPixel;
    std::uint8_t const *epilog_mask =
        UseMask ? blocks_mask + n_blocks * BLOCKSIZE : nullptr;

#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (std::size_t block = 0; block < n_blocks; ++block) {
        // We pre-compute all the bin indices for the block here, which
        // facilitates experimenting with potential optimizations, but the
        // compiler may well interleave this with the bin increments below.
        std::array<std::size_t, BLOCKSIZE * SamplesPerPixel> bins;
        for (std::size_t n = 0; n < BLOCKSIZE * SamplesPerPixel; ++n) {
            auto const i = block * BLOCKSIZE * SamplesPerPixel + n;
            bins[n] = internal::bin_index<T, Bits, LoBit>(blocks_data[i]);
        }
        auto const *block_mask =
            UseMask ? blocks_mask + block * BLOCKSIZE : nullptr;

        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            auto const offset = offsets[c];
            for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                if (!UseMask || block_mask[k]) {
                    auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                    auto const bin = bins[k * SamplesPerPixel + offset];
                    ++stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
                }
            }
        }
    }

    if constexpr (USE_STRIPES) {
        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            for (std::size_t bin = 0; bin < NBINS; ++bin) {
                std::uint32_t sum = 0;
                for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                    sum +=
                        stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
                }
                histogram[c * NBINS + bin] += sum;
            }
        }
    }

    hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel, Sample0Index,
                        SampleIndices...>(epilog_data, epilog_mask,
                                          epilog_size, histogram);
}

template <tuning_parameters const &Tuning, typename T, bool UseMask = false,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t SamplesPerPixel = 1, std::size_t Sample0Index = 0,
          std::size_t... SampleIndices>
IHIST_NOINLINE void
histxy_striped_st(T const *IHIST_RESTRICT data,
                  std::uint8_t const *IHIST_RESTRICT mask, std::size_t width,
                  std::size_t height, std::size_t roi_x, std::size_t roi_y,
                  std::size_t roi_width, std::size_t roi_height,
                  std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(roi_x + roi_width <= width);
    assert(roi_y + roi_height <= height);

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NSTRIPES =
        std::max(std::size_t(1), Tuning.n_stripes);
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NCOMPONENTS = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NCOMPONENTS> offsets{Sample0Index,
                                                           SampleIndices...};

    // Simplify to single row if full-width.
    if (roi_width == width && height > 1) {
        return histxy_striped_st<Tuning, T, UseMask, Bits, LoBit,
                                 SamplesPerPixel, Sample0Index,
                                 SampleIndices...>(
            data + roi_y * width * SamplesPerPixel,
            UseMask ? mask + roi_y * width : nullptr, width * roi_height, 1, 0,
            0, width * roi_height, 1, histogram);
    }

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    constexpr bool USE_STRIPES = NSTRIPES > 1 || STRIPE_LEN > NBINS;

    std::vector<std::uint32_t> stripes_storage;
    std::uint32_t *stripes = [&]() {
        if constexpr (USE_STRIPES) {
            stripes_storage.resize(NSTRIPES * NCOMPONENTS * STRIPE_LEN);
            return stripes_storage.data();
        } else {
            return histogram;
        }
    }();

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), Tuning.n_unroll);
    std::size_t const n_blocks_per_row = roi_width / BLOCKSIZE;
    std::size_t const row_epilog_size = roi_width % BLOCKSIZE;

    for (std::size_t y = roi_y; y < roi_y + roi_height; ++y) {
        T const *row_data = data + (y * width + roi_x) * SamplesPerPixel;
        std::uint8_t const *row_mask =
            UseMask ? mask + y * width + roi_x : nullptr;
        T const *row_epilog_data =
            row_data + n_blocks_per_row * BLOCKSIZE * SamplesPerPixel;
        std::uint8_t const *row_epilog_mask =
            UseMask ? row_mask + n_blocks_per_row * BLOCKSIZE : nullptr;
#ifdef __clang__
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
        for (std::size_t block = 0; block < n_blocks_per_row; ++block) {
            std::array<std::size_t, BLOCKSIZE * SamplesPerPixel> bins;
            for (std::size_t n = 0; n < BLOCKSIZE * SamplesPerPixel; ++n) {
                auto const i = block * BLOCKSIZE * SamplesPerPixel + n;
                bins[n] = internal::bin_index<T, Bits, LoBit>(row_data[i]);
            }
            auto const *block_mask =
                UseMask ? row_mask + block * BLOCKSIZE : nullptr;

            for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
                auto const offset = offsets[c];
                for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                    if (!UseMask || block_mask[k]) {
                        auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                        auto const bin = bins[k * SamplesPerPixel + offset];
                        ++stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN +
                                  bin];
                    }
                }
            }
        }

        // Epilog goes straight to the final histogram.
        hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                            Sample0Index, SampleIndices...>(
            row_epilog_data, row_epilog_mask, row_epilog_size, histogram);
    }

    if constexpr (USE_STRIPES) {
        for (std::size_t c = 0; c < NCOMPONENTS; ++c) {
            for (std::size_t bin = 0; bin < NBINS; ++bin) {
                std::uint32_t sum = 0;
                for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                    sum +=
                        stripes[(stripe * NCOMPONENTS + c) * STRIPE_LEN + bin];
                }
                histogram[c * NBINS + bin] += sum;
            }
        }
    }
}

namespace internal {

template <typename T>
using hist_st_func = void(T const *IHIST_RESTRICT,
                          std::uint8_t const *IHIST_RESTRICT, std::size_t,
                          std::uint32_t *IHIST_RESTRICT, std::size_t);

template <typename T>
using histxy_st_func = void(T const *IHIST_RESTRICT,
                            std::uint8_t const *IHIST_RESTRICT, std::size_t,
                            std::size_t, std::size_t, std::size_t, std::size_t,
                            std::size_t, std::uint32_t *IHIST_RESTRICT,
                            std::size_t);

template <typename T, std::size_t HistSize>
void hist_mt(hist_st_func<T> *hist_func, T const *IHIST_RESTRICT data,
             std::uint8_t const *IHIST_RESTRICT mask, std::size_t size,
             std::size_t samples_per_pixel,
             std::uint32_t *IHIST_RESTRICT histogram,
             std::size_t grain_size = 1) {
    using hist_array = std::array<std::uint32_t, HistSize>;
    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    // Histogramming scales very poorly with simultaneous multithreading
    // (Hyper-Threading), so only schedule 1 thread per physical core.
    int const n_phys_cores = get_physical_core_count();
    auto arena =
        n_phys_cores > 0 ? tbb::task_arena(n_phys_cores) : tbb::task_arena();
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size, grain_size),
                          [&](tbb::blocked_range<std::size_t> const &r) {
                              auto &h = local_hists.local();
                              hist_func(data + r.begin() * samples_per_pixel,
                                        mask == nullptr ? nullptr
                                                        : mask + r.begin(),
                                        r.size(), h.data(), 0);
                          });
    });

    local_hists.combine_each([&](hist_array const &h) {
        std::transform(h.begin(), h.end(), histogram, histogram, std::plus{});
    });
}

template <typename T, std::size_t HistSize>
void histxy_mt(histxy_st_func<T> *histxy_func, T const *IHIST_RESTRICT data,
               std::uint8_t const *IHIST_RESTRICT mask, std::size_t width,
               std::size_t height, std::size_t roi_x, std::size_t roi_y,
               std::size_t roi_width, std::size_t roi_height,
               std::uint32_t *IHIST_RESTRICT histogram,
               std::size_t grain_size = 1) {
    using hist_array = std::array<std::uint32_t, HistSize>;
    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    auto const h_grain_size = std::max(
        std::size_t(1), grain_size / std::max(std::size_t(1), roi_width));

    // Histogramming scales very poorly with simultaneous multithreading
    // (Hyper-Threading), so only schedule 1 thread per physical core.
    int const n_phys_cores = get_physical_core_count();
    auto arena =
        n_phys_cores > 0 ? tbb::task_arena(n_phys_cores) : tbb::task_arena();
    arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, roi_height, h_grain_size),
            [&](tbb::blocked_range<std::size_t> const &r) {
                auto &h = local_hists.local();
                histxy_func(data, mask, width, height, roi_x,
                            roi_y + r.begin(), roi_width, r.size(), h.data(),
                            0);
            });
    });

    local_hists.combine_each([&](hist_array const &h) {
        std::transform(h.begin(), h.end(), histogram, histogram, std::plus{});
    });
}

} // namespace internal

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t SamplesPerPixel = 1,
          std::size_t Sample0Index = 0, std::size_t... SampleIndices>
IHIST_NOINLINE void
hist_unoptimized_mt(T const *IHIST_RESTRICT data,
                    std::uint8_t const *IHIST_RESTRICT mask, std::size_t size,
                    std::uint32_t *IHIST_RESTRICT histogram,
                    std::size_t grain_size = 1) {
    constexpr auto NCOMPONENTS = 1 + sizeof...(SampleIndices);
    internal::hist_mt<T, (1uLL << Bits) * NCOMPONENTS>(
        hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                            Sample0Index, SampleIndices...>,
        data, mask, size, SamplesPerPixel, histogram, grain_size);
}

template <tuning_parameters const &Tuning, typename T, bool UseMask = false,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t SamplesPerPixel = 1, std::size_t Sample0Index = 0,
          std::size_t... SampleIndices>
IHIST_NOINLINE void hist_striped_mt(T const *IHIST_RESTRICT data,
                                    std::uint8_t const *IHIST_RESTRICT mask,
                                    std::size_t size,
                                    std::uint32_t *IHIST_RESTRICT histogram,
                                    std::size_t grain_size = 1) {
    constexpr auto NCOMPONENTS = 1 + sizeof...(SampleIndices);
    internal::hist_mt<T, (1uLL << Bits) * NCOMPONENTS>(
        hist_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                        Sample0Index, SampleIndices...>,
        data, mask, size, SamplesPerPixel, histogram, grain_size);
}

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t SamplesPerPixel = 1,
          std::size_t Sample0Index = 0, std::size_t... SampleIndices>
IHIST_NOINLINE void histxy_unoptimized_mt(
    T const *IHIST_RESTRICT data, std::uint8_t const *IHIST_RESTRICT mask,
    std::size_t width, std::size_t height, std::size_t roi_x,
    std::size_t roi_y, std::size_t roi_width, std::size_t roi_height,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t grain_size = 1) {
    constexpr auto NCOMPONENTS = 1 + sizeof...(SampleIndices);
    internal::histxy_mt<T, (1uLL << Bits) * NCOMPONENTS>(
        histxy_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                              Sample0Index, SampleIndices...>,
        data, mask, width, height, roi_x, roi_y, roi_width, roi_height,
        histogram, grain_size);
}

template <tuning_parameters const &Tuning, typename T, bool UseMask = false,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t SamplesPerPixel = 1, std::size_t Sample0Index = 0,
          std::size_t... SampleIndices>
IHIST_NOINLINE void histxy_striped_mt(
    T const *IHIST_RESTRICT data, std::uint8_t const *IHIST_RESTRICT mask,
    std::size_t width, std::size_t height, std::size_t roi_x,
    std::size_t roi_y, std::size_t roi_width, std::size_t roi_height,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t grain_size = 1) {
    constexpr auto NCOMPONENTS = 1 + sizeof...(SampleIndices);
    internal::histxy_mt<T, (1uLL << Bits) * NCOMPONENTS>(
        histxy_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                          Sample0Index, SampleIndices...>,
        data, mask, width, height, roi_x, roi_y, roi_width, roi_height,
        histogram, grain_size);
}

} // namespace ihist

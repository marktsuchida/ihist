/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "phys_core_count.hpp"

#ifdef IHIST_USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

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

#ifdef __clang__
#define IHIST_PRAGMA_DISABLE_LOOP_UNROLL _Pragma("clang loop unroll(disable)")
#elif defined(__GNUC__)
#define IHIST_PRAGMA_DISABLE_LOOP_UNROLL _Pragma("GCC unroll 0")
#else
#define IHIST_PRAGMA_DISABLE_LOOP_UNROLL
#endif

namespace ihist {

struct tuning_parameters {
    // Number of separate histograms to iterate over (to tune for store-to-load
    // latency hiding vs spatial locality).
    std::size_t n_stripes;

    // Pixels processed per main loop iteration.
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
    constexpr std::size_t NSAMPLES = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NSAMPLES> s_indices{Sample0Index,
                                                          SampleIndices...};

    IHIST_PRAGMA_DISABLE_LOOP_UNROLL
    for (std::size_t j = 0; j < size; ++j) {
        auto const i = j * SamplesPerPixel;
        if (!UseMask || mask[j]) {
            for (std::size_t s = 0; s < NSAMPLES; ++s) {
                auto const s_index = s_indices[s];
                auto const bin =
                    internal::bin_index<T, Bits, LoBit>(data[i + s_index]);
                if (bin != NBINS) {
                    ++histogram[s * NBINS + bin];
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
    std::size_t height, std::size_t width, std::size_t stride,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(width <= stride);

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NBINS = 1uLL << Bits;
    constexpr std::size_t NSAMPLES = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NSAMPLES> s_indices{Sample0Index,
                                                          SampleIndices...};

    IHIST_PRAGMA_DISABLE_LOOP_UNROLL
    for (std::size_t y = 0; y < height; ++y) {
        IHIST_PRAGMA_DISABLE_LOOP_UNROLL
        for (std::size_t x = 0; x < width; ++x) {
            auto const j = y * stride + x;
            auto const i = j * SamplesPerPixel;
            if (!UseMask || mask[j]) {
                for (std::size_t s = 0; s < NSAMPLES; ++s) {
                    auto const s_index = s_indices[s];
                    auto const bin =
                        internal::bin_index<T, Bits, LoBit>(data[i + s_index]);
                    if (bin != NBINS) {
                        ++histogram[s * NBINS + bin];
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
    constexpr std::size_t NSAMPLES = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NSAMPLES> s_indices{Sample0Index,
                                                          SampleIndices...};

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    constexpr bool USE_STRIPES = NSTRIPES > 1 || STRIPE_LEN > NBINS;

    std::vector<std::uint32_t> stripes_storage;
    std::uint32_t *stripes = [&]() {
        if constexpr (USE_STRIPES) {
            stripes_storage.resize(NSTRIPES * NSAMPLES * STRIPE_LEN);
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

    IHIST_PRAGMA_DISABLE_LOOP_UNROLL
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

        for (std::size_t s = 0; s < NSAMPLES; ++s) {
            auto const s_index = s_indices[s];
            for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                if (!UseMask || block_mask[k]) {
                    auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                    auto const bin = bins[k * SamplesPerPixel + s_index];
                    ++stripes[(stripe * NSAMPLES + s) * STRIPE_LEN + bin];
                }
            }
        }
    }

    if constexpr (USE_STRIPES) {
        for (std::size_t s = 0; s < NSAMPLES; ++s) {
            for (std::size_t bin = 0; bin < NBINS; ++bin) {
                std::uint32_t sum = 0;
                for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                    sum += stripes[(stripe * NSAMPLES + s) * STRIPE_LEN + bin];
                }
                histogram[s * NBINS + bin] += sum;
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
                  std::uint8_t const *IHIST_RESTRICT mask, std::size_t height,
                  std::size_t width, std::size_t stride,
                  std::uint32_t *IHIST_RESTRICT histogram, std::size_t = 0) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(width <= stride);

    static_assert(std::max({Sample0Index, SampleIndices...}) <
                  SamplesPerPixel);

    constexpr std::size_t NSTRIPES =
        std::max(std::size_t(1), Tuning.n_stripes);
    constexpr std::size_t NBINS = 1 << Bits;
    constexpr std::size_t NSAMPLES = 1 + sizeof...(SampleIndices);
    constexpr std::array<std::size_t, NSAMPLES> s_indices{Sample0Index,
                                                          SampleIndices...};

    // Simplify to single row if full-width.
    if (width == stride && height > 1) {
        auto const size = height * width;
        return histxy_striped_st<Tuning, T, UseMask, Bits, LoBit,
                                 SamplesPerPixel, Sample0Index,
                                 SampleIndices...>(
            data, UseMask ? mask : nullptr, 1, size, size, histogram);
    }

    // Use extra bin for overflows if applicable.
    constexpr auto STRIPE_LEN =
        NBINS + static_cast<std::size_t>(Bits + LoBit < 8 * sizeof(T));
    constexpr bool USE_STRIPES = NSTRIPES > 1 || STRIPE_LEN > NBINS;

    std::vector<std::uint32_t> stripes_storage;
    std::uint32_t *stripes = [&]() {
        if constexpr (USE_STRIPES) {
            stripes_storage.resize(NSTRIPES * NSAMPLES * STRIPE_LEN);
            return stripes_storage.data();
        } else {
            return histogram;
        }
    }();

    constexpr std::size_t BLOCKSIZE =
        std::max(std::size_t(1), Tuning.n_unroll);
    std::size_t const n_blocks_per_row = width / BLOCKSIZE;
    std::size_t const row_epilog_size = width % BLOCKSIZE;

    for (std::size_t y = 0; y < height; ++y) {
        T const *row_data = data + y * stride * SamplesPerPixel;
        std::uint8_t const *row_mask = UseMask ? mask + y * stride : nullptr;
        T const *row_epilog_data =
            row_data + n_blocks_per_row * BLOCKSIZE * SamplesPerPixel;
        std::uint8_t const *row_epilog_mask =
            UseMask ? row_mask + n_blocks_per_row * BLOCKSIZE : nullptr;
        IHIST_PRAGMA_DISABLE_LOOP_UNROLL
        for (std::size_t block = 0; block < n_blocks_per_row; ++block) {
            std::array<std::size_t, BLOCKSIZE * SamplesPerPixel> bins;
            for (std::size_t n = 0; n < BLOCKSIZE * SamplesPerPixel; ++n) {
                auto const i = block * BLOCKSIZE * SamplesPerPixel + n;
                bins[n] = internal::bin_index<T, Bits, LoBit>(row_data[i]);
            }
            auto const *block_mask =
                UseMask ? row_mask + block * BLOCKSIZE : nullptr;

            for (std::size_t s = 0; s < NSAMPLES; ++s) {
                auto const s_index = s_indices[s];
                for (std::size_t k = 0; k < BLOCKSIZE; ++k) {
                    if (!UseMask || block_mask[k]) {
                        auto const stripe = (block * BLOCKSIZE + k) % NSTRIPES;
                        auto const bin = bins[k * SamplesPerPixel + s_index];
                        ++stripes[(stripe * NSAMPLES + s) * STRIPE_LEN + bin];
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
        for (std::size_t s = 0; s < NSAMPLES; ++s) {
            for (std::size_t bin = 0; bin < NBINS; ++bin) {
                std::uint32_t sum = 0;
                for (std::size_t stripe = 0; stripe < NSTRIPES; ++stripe) {
                    sum += stripes[(stripe * NSAMPLES + s) * STRIPE_LEN + bin];
                }
                histogram[s * NBINS + bin] += sum;
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
                            std::size_t, std::size_t,
                            std::uint32_t *IHIST_RESTRICT, std::size_t);

template <typename T, std::size_t HistSize>
void hist_mt(hist_st_func<T> *hist_func, T const *IHIST_RESTRICT data,
             std::uint8_t const *IHIST_RESTRICT mask, std::size_t size,
             std::size_t n_components, std::uint32_t *IHIST_RESTRICT histogram,
             std::size_t grain_size = 1) {
#ifdef IHIST_USE_TBB
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
                              hist_func(data + r.begin() * n_components,
                                        mask == nullptr ? nullptr
                                                        : mask + r.begin(),
                                        r.size(), h.data(), 0);
                          });
    });

    local_hists.combine_each([&](hist_array const &h) {
        std::transform(h.begin(), h.end(), histogram, histogram, std::plus{});
    });
#else
    (void)grain_size;
    (void)n_components;
    hist_func(data, mask, size, histogram, 0);
#endif
}

template <typename T, std::size_t SamplesPerPixel, std::size_t HistSize>
void histxy_mt(histxy_st_func<T> *histxy_func, T const *IHIST_RESTRICT data,
               std::uint8_t const *IHIST_RESTRICT mask, std::size_t height,
               std::size_t width, std::size_t stride,
               std::uint32_t *IHIST_RESTRICT histogram,
               std::size_t grain_size = 1) {
#ifdef IHIST_USE_TBB
    using hist_array = std::array<std::uint32_t, HistSize>;
    tbb::combinable<hist_array> local_hists([] { return hist_array{}; });

    auto const h_grain_size =
        std::max(std::size_t(1), grain_size / std::max(std::size_t(1), width));

    // Histogramming scales very poorly with simultaneous multithreading
    // (Hyper-Threading), so only schedule 1 thread per physical core.
    int const n_phys_cores = get_physical_core_count();
    auto arena =
        n_phys_cores > 0 ? tbb::task_arena(n_phys_cores) : tbb::task_arena();
    arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, height, h_grain_size),
            [&](tbb::blocked_range<std::size_t> const &r) {
                auto &h = local_hists.local();
                histxy_func(data + r.begin() * stride * SamplesPerPixel,
                            mask ? mask + r.begin() * stride : nullptr,
                            r.size(), width, stride, h.data(), 0);
            });
    });

    local_hists.combine_each([&](hist_array const &h) {
        std::transform(h.begin(), h.end(), histogram, histogram, std::plus{});
    });
#else
    (void)grain_size;
    histxy_func(data, mask, height, width, stride, histogram, 0);
#endif
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
#ifdef IHIST_USE_TBB
    constexpr auto NSAMPLES = 1 + sizeof...(SampleIndices);
    internal::hist_mt<T, (1uLL << Bits) * NSAMPLES>(
        hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                            Sample0Index, SampleIndices...>,
        data, mask, size, SamplesPerPixel, histogram, grain_size);
#else
    (void)grain_size;
    hist_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel, Sample0Index,
                        SampleIndices...>(data, mask, size, histogram, 0);
#endif
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
#ifdef IHIST_USE_TBB
    constexpr auto NSAMPLES = 1 + sizeof...(SampleIndices);
    internal::hist_mt<T, (1uLL << Bits) * NSAMPLES>(
        hist_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                        Sample0Index, SampleIndices...>,
        data, mask, size, SamplesPerPixel, histogram, grain_size);
#else
    (void)grain_size;
    hist_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                    Sample0Index, SampleIndices...>(data, mask, size,
                                                    histogram, 0);
#endif
}

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0, std::size_t SamplesPerPixel = 1,
          std::size_t Sample0Index = 0, std::size_t... SampleIndices>
IHIST_NOINLINE void histxy_unoptimized_mt(
    T const *IHIST_RESTRICT data, std::uint8_t const *IHIST_RESTRICT mask,
    std::size_t height, std::size_t width, std::size_t stride,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t grain_size = 1) {
#ifdef IHIST_USE_TBB
    constexpr auto NSAMPLES = 1 + sizeof...(SampleIndices);
    internal::histxy_mt<T, SamplesPerPixel, (1uLL << Bits) * NSAMPLES>(
        histxy_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                              Sample0Index, SampleIndices...>,
        data, mask, height, width, stride, histogram, grain_size);
#else
    (void)grain_size;
    histxy_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                          Sample0Index, SampleIndices...>(
        data, mask, height, width, stride, histogram, 0);
#endif
}

template <tuning_parameters const &Tuning, typename T, bool UseMask = false,
          unsigned Bits = 8 * sizeof(T), unsigned LoBit = 0,
          std::size_t SamplesPerPixel = 1, std::size_t Sample0Index = 0,
          std::size_t... SampleIndices>
IHIST_NOINLINE void histxy_striped_mt(T const *IHIST_RESTRICT data,
                                      std::uint8_t const *IHIST_RESTRICT mask,
                                      std::size_t height, std::size_t width,
                                      std::size_t stride,
                                      std::uint32_t *IHIST_RESTRICT histogram,
                                      std::size_t grain_size = 1) {
#ifdef IHIST_USE_TBB
    constexpr auto NSAMPLES = 1 + sizeof...(SampleIndices);
    internal::histxy_mt<T, SamplesPerPixel, (1uLL << Bits) * NSAMPLES>(
        histxy_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                          Sample0Index, SampleIndices...>,
        data, mask, height, width, stride, histogram, grain_size);
#else
    (void)grain_size;
    histxy_striped_st<Tuning, T, UseMask, Bits, LoBit, SamplesPerPixel,
                      Sample0Index, SampleIndices...>(
        data, mask, height, width, stride, histogram, 0);
#endif
}

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0>
/* not noinline */ void
histxy_dynamic_st(T const *IHIST_RESTRICT data,
                  std::uint8_t const *IHIST_RESTRICT mask, std::size_t height,
                  std::size_t width, std::size_t stride,
                  std::size_t n_components, std::size_t n_hist_components,
                  std::size_t const *IHIST_RESTRICT component_indices,
                  std::uint32_t *IHIST_RESTRICT histogram) {
    assert(width * height < std::numeric_limits<std::uint32_t>::max());
    assert(width <= stride);
    assert(component_indices != nullptr || n_hist_components == 0);

    constexpr std::size_t NBINS = 1uLL << Bits;

    // Simplify to single row if full-width.
    if (width == stride && height > 1) {
        auto const size = height * width;
        return histxy_dynamic_st<T, UseMask, Bits, LoBit>(
            data, mask, 1, size, size, n_components, n_hist_components,
            component_indices, histogram);
    }

    // We could implement striping for dynamic components, perhaps only for the
    // cases of 2-4 components being histogrammed. But keep it simple for now
    // because these are meant to be uncommon cases; if a very common case
    // comes to light, we can add a static implementation for it.

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            auto const j = y * stride + x;
            auto const i = j * n_components;
            if (!UseMask || mask[j]) {
                for (std::size_t s = 0; s < n_hist_components; ++s) {
                    auto const s_index = component_indices[s];
                    auto const bin =
                        internal::bin_index<T, Bits, LoBit>(data[i + s_index]);
                    if (bin != NBINS) {
                        ++histogram[s * NBINS + bin];
                    }
                }
            }
        }
    }
}

template <typename T, bool UseMask = false, unsigned Bits = 8 * sizeof(T),
          unsigned LoBit = 0>
IHIST_NOINLINE void histxy_dynamic_mt(
    T const *IHIST_RESTRICT data, std::uint8_t const *IHIST_RESTRICT mask,
    std::size_t height, std::size_t width, std::size_t stride,
    std::size_t n_components, std::size_t n_hist_components,
    std::size_t const *IHIST_RESTRICT component_indices,
    std::uint32_t *IHIST_RESTRICT histogram, std::size_t grain_size = 1) {
#ifdef IHIST_USE_TBB
    constexpr std::size_t NBINS = 1uLL << Bits;
    std::size_t const hist_size = n_hist_components * NBINS;

    using hist_vec = std::vector<std::uint32_t>;
    tbb::combinable<hist_vec> local_hists(
        [hist_size] { return hist_vec(hist_size, 0); });

    auto const h_grain_size =
        std::max(std::size_t(1), grain_size / std::max(std::size_t(1), width));

    // Histogramming scales very poorly with simultaneous multithreading
    // (Hyper-Threading), so only schedule 1 thread per physical core.
    int const n_phys_cores = internal::get_physical_core_count();
    auto arena =
        n_phys_cores > 0 ? tbb::task_arena(n_phys_cores) : tbb::task_arena();

    arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, height, h_grain_size),
            [&](tbb::blocked_range<std::size_t> const &r) {
                auto &h = local_hists.local();
                histxy_dynamic_st<T, UseMask, Bits, LoBit>(
                    data + r.begin() * stride * n_components,
                    mask ? mask + r.begin() * stride : nullptr, r.size(),
                    width, stride, n_components, n_hist_components,
                    component_indices, h.data());
            });
    });

    local_hists.combine_each([&](hist_vec const &h) {
        std::transform(h.begin(), h.end(), histogram, histogram, std::plus{});
    });
#else
    (void)grain_size;
    histxy_dynamic_st<T, UseMask, Bits, LoBit>(
        data, mask, height, width, stride, n_components, n_hist_components,
        component_indices, histogram);
#endif
}

} // namespace ihist

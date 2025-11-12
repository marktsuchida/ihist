/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include "ihist/ihist.h"

#include "ihist.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace {

// TODO Support locally-generated tuning via build option.

// Use the results of automatic tuning for striping and unrolling.
#define TUNE(pixel_type, bits, mask, stripes, unrolls)                        \
    constexpr auto tuning_##bits##bit_##pixel_type##_mask##mask =             \
        ihist::tuning_parameters{stripes, unrolls};

#if defined(__APPLE__) && defined(__aarch64__)
#include "tuning_apple_arm64.h"
#elif defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) ||       \
    defined(__amd64) || defined(_M_X64)
#include "tuning_x86_64.h"
#else
#include "tuning_default.h"
#endif

#undef TUNE

// We tune xabc identically to abcx, at least for now:
constexpr auto tuning_8bit_xabc_mask0 = tuning_8bit_abcx_mask0;
constexpr auto tuning_8bit_xabc_mask1 = tuning_8bit_abcx_mask1;
constexpr auto tuning_12bit_xabc_mask0 = tuning_12bit_abcx_mask0;
constexpr auto tuning_12bit_xabc_mask1 = tuning_12bit_abcx_mask1;
constexpr auto tuning_16bit_xabc_mask0 = tuning_16bit_abcx_mask0;
constexpr auto tuning_16bit_xabc_mask1 = tuning_16bit_abcx_mask1;

// The following parallelism tuning values were manually picked, based on
// benchmarking. (Further work is needed to see if these values work well
// across different machines.)
//
// Tune the grain size, then input size threshold.
//
// If the values are too small, efficiency (CPU time for a given input,
// compared to single-threaded) will decrease. There are two sources of this
// inefficiency: the per-thread stripe reduction (which scales with chunk
// count) and the overhead of thread and task management. The grain size and
// threshold are intended to avoid those two situations, respectively, although
// they are not completely orthogonal to each other.
//
// If the values are too large, latency suffers for "medium" input sizes,
// because we won't parallelize (or use as many cores as possible).
//
// Thus, the tuning requires balancing low latency for slightly larger inputs
// and high efficiency for slightly smaller inputs. Because our main goal is
// histogramming for live image stream visualization, we start caring less
// about absolute latency for a given input size once we are well below 10 ms.
//
// (To get an initial estimate, it is useful to limit thread count to 2 and see
// what grain size and input size are required to prevent excessive
// inefficiency. After picking the values, check efficiency and latency for all
// pixel formats, over input sizes spanning the threshold and grain size. Also
// confirm stability with grain sizes 1/2x and 2x of the chosen value.)
constexpr std::size_t parallel_size_threshold = 1uLL << 20;
constexpr std::size_t parallel_grain_size = 1uLL << 20;

} // namespace

namespace {

// Buffer conversion for different Bits: NSamples = 0 means use run-time
// n_histogram_samples.

template <typename T, std::size_t Bits, std::size_t NSamples = 0>
auto hist_buffer_of_higher_bits(std::size_t sample_bits,
                                std::uint32_t const *histogram,
                                std::size_t n_histogram_samples = NSamples)
    -> std::vector<std::uint32_t> {
    static_assert(Bits <= 8 * sizeof(T));
    auto const n_samples = NSamples != 0 ? NSamples : n_histogram_samples;
    std::vector<std::uint32_t> hist;
    hist.reserve(n_samples << Bits);
    hist.assign(histogram,
                std::next(histogram, std::size_t(1) << sample_bits));
    hist.resize(n_samples << Bits);
    for (std::size_t i = 1; i < n_samples; ++i) {
        std::copy_n(std::next(histogram, i << sample_bits),
                    std::size_t(1) << sample_bits,
                    std::next(hist.begin(), i << Bits));
    }
    return hist;
}

template <typename T, std::size_t Bits, std::size_t NSamples = 0>
void copy_hist_from_higher_bits(std::size_t sample_bits,
                                std::uint32_t *histogram,
                                std::vector<std::uint32_t> const &hist,
                                std::size_t n_histogram_samples = NSamples) {
    static_assert(Bits <= 8 * sizeof(T));
    auto const n_samples = NSamples != 0 ? NSamples : n_histogram_samples;
    for (std::size_t i = 0; i < n_samples; ++i) {
        std::copy_n(std::next(hist.begin(), i << Bits),
                    std::size_t(1) << sample_bits,
                    std::next(histogram, i << sample_bits));
    }
}

template <typename T, std::size_t Bits>
auto hist_buffer_of_higher_bits_dynamic(std::size_t sample_bits,
                                        std::size_t n_histogram_samples,
                                        std::uint32_t const *histogram)
    -> std::vector<std::uint32_t> {
    return hist_buffer_of_higher_bits<T, Bits, 0>(sample_bits, histogram,
                                                  n_histogram_samples);
}

template <typename T, std::size_t Bits>
void copy_hist_from_higher_bits_dynamic(
    std::size_t sample_bits, std::size_t n_histogram_samples,
    std::uint32_t *histogram, std::vector<std::uint32_t> const &hist) {
    copy_hist_from_higher_bits<T, Bits, 0>(sample_bits, histogram, hist,
                                           n_histogram_samples);
}

template <typename T, std::size_t Bits,
          ihist::tuning_parameters const &NomaskTuning,
          ihist::tuning_parameters const &MaskedTuning,
          std::size_t SamplesPerPixel, std::size_t... SampleIndices>
void hist_2d_impl(std::size_t sample_bits, T const *IHIST_RESTRICT image,
                  std::uint8_t const *IHIST_RESTRICT mask, std::size_t height,
                  std::size_t width, std::size_t stride,
                  std::uint32_t *IHIST_RESTRICT histogram,
                  bool maybe_parallel) {
    assert(sample_bits <= Bits);
    assert(image != nullptr);
    assert(histogram != nullptr);

    std::vector<std::uint32_t> buffer;
    std::uint32_t *hist{};
    if (sample_bits == Bits) {
        hist = histogram;
    } else {
        buffer = hist_buffer_of_higher_bits<T, Bits, sizeof...(SampleIndices)>(
            sample_bits, histogram);
        hist = buffer.data();
    }

    if (maybe_parallel && width * height >= parallel_size_threshold) {
        if (mask != nullptr) {
            ihist::histxy_striped_mt<MaskedTuning, T, true, Bits, 0,
                                     SamplesPerPixel, SampleIndices...>(
                image, mask, height, width, stride, hist, parallel_grain_size);
        } else {
            ihist::histxy_striped_mt<NomaskTuning, T, false, Bits, 0,
                                     SamplesPerPixel, SampleIndices...>(
                image, mask, height, width, stride, hist, parallel_grain_size);
        }
    } else {
        if (mask != nullptr) {
            ihist::histxy_striped_st<MaskedTuning, T, true, Bits, 0,
                                     SamplesPerPixel, SampleIndices...>(
                image, mask, height, width, stride, hist);
        } else {
            ihist::histxy_striped_st<NomaskTuning, T, false, Bits, 0,
                                     SamplesPerPixel, SampleIndices...>(
                image, mask, height, width, stride, hist);
        }
    }

    if (sample_bits < Bits) {
        copy_hist_from_higher_bits<T, Bits, sizeof...(SampleIndices)>(
            sample_bits, histogram, buffer);
    }
}

template <typename T, std::size_t Bits>
void hist_2d_dynamic(std::size_t sample_bits, T const *IHIST_RESTRICT image,
                     std::uint8_t const *IHIST_RESTRICT mask,
                     std::size_t height, std::size_t width, std::size_t stride,
                     std::size_t n_components, std::size_t n_histogram_samples,
                     std::size_t const *IHIST_RESTRICT sample_indices,
                     std::uint32_t *IHIST_RESTRICT histogram,
                     bool maybe_parallel) {
    assert(sample_bits <= Bits);
    assert(image != nullptr);
    assert(histogram != nullptr);
    assert(sample_indices != nullptr);
    assert(n_histogram_samples > 0);

    std::vector<std::uint32_t> buffer;
    std::uint32_t *hist{};
    if (sample_bits == Bits) {
        hist = histogram;
    } else {
        buffer = hist_buffer_of_higher_bits_dynamic<T, Bits>(
            sample_bits, n_histogram_samples, histogram);
        hist = buffer.data();
    }

    if (maybe_parallel && width * height >= parallel_size_threshold) {
        if (mask != nullptr) {
            ihist::histxy_dynamic_mt<T, true, Bits, 0>(
                image, mask, height, width, stride, n_components,
                n_histogram_samples, sample_indices, hist,
                parallel_grain_size);
        } else {
            ihist::histxy_dynamic_mt<T, false, Bits, 0>(
                image, mask, height, width, stride, n_components,
                n_histogram_samples, sample_indices, hist,
                parallel_grain_size);
        }
    } else {
        if (mask != nullptr) {
            ihist::histxy_dynamic_st<T, true, Bits, 0>(
                image, mask, height, width, stride, n_components,
                n_histogram_samples, sample_indices, hist);
        } else {
            ihist::histxy_dynamic_st<T, false, Bits, 0>(
                image, mask, height, width, stride, n_components,
                n_histogram_samples, sample_indices, hist);
        }
    }

    if (sample_bits < Bits) {
        copy_hist_from_higher_bits_dynamic<T, Bits>(
            sample_bits, n_histogram_samples, histogram, buffer);
    }
}

bool indices_match(std::size_t n_histogram_samples,
                   std::size_t const *sample_indices,
                   std::initializer_list<std::size_t> expected) {
    if (n_histogram_samples != expected.size())
        return false;
    auto it = expected.begin();
    for (std::size_t i = 0; i < n_histogram_samples; ++i, ++it) {
        if (sample_indices[i] != *it)
            return false;
    }
    return true;
}

// Unified dispatch for common pixel format optimizations
template <typename T, std::size_t Bits,
          ihist::tuning_parameters const &MonoMask0,
          ihist::tuning_parameters const &MonoMask1,
          ihist::tuning_parameters const &AbcMask0,
          ihist::tuning_parameters const &AbcMask1,
          ihist::tuning_parameters const &AbcxMask0,
          ihist::tuning_parameters const &AbcxMask1,
          ihist::tuning_parameters const &XabcMask0,
          ihist::tuning_parameters const &XabcMask1>
void dispatch_common_pixel_formats(
    std::size_t sample_bits, T const *IHIST_RESTRICT image,
    std::uint8_t const *IHIST_RESTRICT mask, std::size_t height,
    std::size_t width, std::size_t stride, std::size_t n_components,
    std::size_t n_histogram_samples,
    std::size_t const *IHIST_RESTRICT sample_indices,
    std::uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel) {

    if (n_components == 1 && n_histogram_samples == 1 &&
        sample_indices[0] == 0) {
        // Mono: optimized path
        hist_2d_impl<T, Bits, MonoMask0, MonoMask1, 1, 0>(
            sample_bits, image, mask, height, width, stride, histogram,
            maybe_parallel);
    } else if (n_components == 3 && n_histogram_samples == 3 &&
               indices_match(n_histogram_samples, sample_indices, {0, 1, 2})) {
        // RGB: optimized path
        hist_2d_impl<T, Bits, AbcMask0, AbcMask1, 3, 0, 1, 2>(
            sample_bits, image, mask, height, width, stride, histogram,
            maybe_parallel);
    } else if (n_components == 4 && n_histogram_samples == 3 &&
               indices_match(n_histogram_samples, sample_indices, {0, 1, 2})) {
        // RGBA (skip last): optimized path
        hist_2d_impl<T, Bits, AbcxMask0, AbcxMask1, 4, 0, 1, 2>(
            sample_bits, image, mask, height, width, stride, histogram,
            maybe_parallel);
    } else if (n_components == 4 && n_histogram_samples == 3 &&
               indices_match(n_histogram_samples, sample_indices, {1, 2, 3})) {
        // ARGB (skip first): optimized path
        hist_2d_impl<T, Bits, XabcMask0, XabcMask1, 4, 1, 2, 3>(
            sample_bits, image, mask, height, width, stride, histogram,
            maybe_parallel);
    } else {
        // General case: dynamic implementation
        hist_2d_dynamic<T, Bits>(sample_bits, image, mask, height, width,
                                 stride, n_components, n_histogram_samples,
                                 sample_indices, histogram, maybe_parallel);
    }
}

} // namespace

extern "C" IHIST_PUBLIC void
ihist_hist8_2d(size_t sample_bits, uint8_t const *IHIST_RESTRICT image,
               uint8_t const *IHIST_RESTRICT mask, size_t height, size_t width,
               size_t stride, size_t n_components, size_t n_histogram_samples,
               size_t const *IHIST_RESTRICT sample_indices,
               uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel) {

    dispatch_common_pixel_formats<
        std::uint8_t, 8, tuning_8bit_mono_mask0, tuning_8bit_mono_mask1,
        tuning_8bit_abc_mask0, tuning_8bit_abc_mask1, tuning_8bit_abcx_mask0,
        tuning_8bit_abcx_mask1, tuning_8bit_xabc_mask0,
        tuning_8bit_xabc_mask1>(sample_bits, image, mask, height, width,
                                stride, n_components, n_histogram_samples,
                                sample_indices, histogram, maybe_parallel);
}

extern "C" IHIST_PUBLIC void
ihist_hist16_2d(size_t sample_bits, uint16_t const *IHIST_RESTRICT image,
                uint8_t const *IHIST_RESTRICT mask, size_t height,
                size_t width, size_t stride, size_t n_components,
                size_t n_histogram_samples,
                size_t const *IHIST_RESTRICT sample_indices,
                uint32_t *IHIST_RESTRICT histogram, bool maybe_parallel) {

    // For 16-bit, use 12-bit path for sample_bits <= 12, otherwise 16-bit
    if (sample_bits <= 12) {
        dispatch_common_pixel_formats<
            std::uint16_t, 12, tuning_12bit_mono_mask0,
            tuning_12bit_mono_mask1, tuning_12bit_abc_mask0,
            tuning_12bit_abc_mask1, tuning_12bit_abcx_mask0,
            tuning_12bit_abcx_mask1, tuning_12bit_xabc_mask0,
            tuning_12bit_xabc_mask1>(
            sample_bits, image, mask, height, width, stride, n_components,
            n_histogram_samples, sample_indices, histogram, maybe_parallel);
    } else {
        dispatch_common_pixel_formats<
            std::uint16_t, 16, tuning_16bit_mono_mask0,
            tuning_16bit_mono_mask1, tuning_16bit_abc_mask0,
            tuning_16bit_abc_mask1, tuning_16bit_abcx_mask0,
            tuning_16bit_abcx_mask1, tuning_16bit_xabc_mask0,
            tuning_16bit_xabc_mask1>(
            sample_bits, image, mask, height, width, stride, n_components,
            n_histogram_samples, sample_indices, histogram, maybe_parallel);
    }
}

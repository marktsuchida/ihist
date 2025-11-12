/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <ihist.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace ihist {

// To facilitate parameterized testing of different tuning parameters and
// implementations, we use this type to map test parameters to functions.
template <typename T, std::size_t Stripes, std::size_t Unroll, bool MT>
struct hist_function_traits {
    using value_type = T;

    using hist_func_type = void(T const *, std::uint8_t const *, std::size_t,
                                std::uint32_t *, std::size_t);

    using histxy_func_type = void(T const *, std::uint8_t const *, std::size_t,
                                  std::size_t, std::size_t, std::uint32_t *,
                                  std::size_t);

    static constexpr tuning_parameters tuning{Stripes, Unroll};

    template <bool UseMask, unsigned Bits, unsigned LoBit,
              std::size_t SamplesPerPixel, std::size_t... SampleIndices>
    static constexpr hist_func_type *hist_func =
        MT ? (Stripes == 0
                  ? hist_unoptimized_mt<T, UseMask, Bits, LoBit,
                                        SamplesPerPixel, SampleIndices...>
                  : hist_striped_mt<tuning, T, UseMask, Bits, LoBit,
                                    SamplesPerPixel, SampleIndices...>)
           : (Stripes == 0
                  ? hist_unoptimized_st<T, UseMask, Bits, LoBit,
                                        SamplesPerPixel, SampleIndices...>
                  : hist_striped_st<tuning, T, UseMask, Bits, LoBit,
                                    SamplesPerPixel, SampleIndices...>);

    template <bool UseMask, unsigned Bits, unsigned LoBit,
              std::size_t SamplesPerPixel, std::size_t... SampleIndices>
    static constexpr histxy_func_type *histxy_func =
        MT ? Stripes == 0
                 ? histxy_unoptimized_mt<T, UseMask, Bits, LoBit,
                                         SamplesPerPixel, SampleIndices...>
                 : histxy_striped_mt<tuning, T, UseMask, Bits, LoBit,
                                     SamplesPerPixel, SampleIndices...>
        : Stripes == 0
            ? histxy_unoptimized_st<T, UseMask, Bits, LoBit, SamplesPerPixel,
                                    SampleIndices...>
            : histxy_striped_st<tuning, T, UseMask, Bits, LoBit,
                                SamplesPerPixel, SampleIndices...>;
};

// For use with TEMPLATE_LIST_TEST_CASE().
using test_traits_list =
    std::tuple<hist_function_traits<std::uint8_t, 0, 1, false>,
               hist_function_traits<std::uint8_t, 0, 1, true>,
               hist_function_traits<std::uint16_t, 0, 1, false>,
               hist_function_traits<std::uint16_t, 0, 1, true>,
               hist_function_traits<std::uint8_t, 1, 1, false>,
               hist_function_traits<std::uint8_t, 1, 1, true>,
               hist_function_traits<std::uint16_t, 1, 1, false>,
               hist_function_traits<std::uint16_t, 1, 1, true>,
               hist_function_traits<std::uint8_t, 1, 3, false>,
               hist_function_traits<std::uint8_t, 1, 3, true>,
               hist_function_traits<std::uint16_t, 1, 3, false>,
               hist_function_traits<std::uint16_t, 1, 3, true>,
               hist_function_traits<std::uint8_t, 3, 1, false>,
               hist_function_traits<std::uint8_t, 3, 1, true>,
               hist_function_traits<std::uint16_t, 3, 1, false>,
               hist_function_traits<std::uint16_t, 3, 1, true>,
               hist_function_traits<std::uint8_t, 3, 3, false>,
               hist_function_traits<std::uint8_t, 3, 3, true>,
               hist_function_traits<std::uint16_t, 3, 3, false>,
               hist_function_traits<std::uint16_t, 3, 3, true>>;

// Traits for dynamic histogram functions.
template <typename T, bool MT> struct dynamic_function_traits {
    using value_type = T;
    static constexpr bool is_mt = MT;

    template <bool UseMask, unsigned Bits, unsigned LoBit>
    static void histxy_dynamic(T const *data, std::uint8_t const *mask,
                               std::size_t height, std::size_t width,
                               std::size_t stride, std::size_t n_components,
                               std::size_t n_hist_components,
                               std::size_t const *component_indices,
                               std::uint32_t *histogram) {
        if constexpr (MT) {
            histxy_dynamic_mt<T, UseMask, Bits, LoBit>(
                data, mask, height, width, stride, n_components,
                n_hist_components, component_indices, histogram);
        } else {
            histxy_dynamic_st<T, UseMask, Bits, LoBit>(
                data, mask, height, width, stride, n_components,
                n_hist_components, component_indices, histogram);
        }
    }
};

// For use with TEMPLATE_LIST_TEST_CASE() for dynamic histogram tests.
using dynamic_test_traits_list =
    std::tuple<dynamic_function_traits<std::uint8_t, false>,
               dynamic_function_traits<std::uint8_t, true>,
               dynamic_function_traits<std::uint16_t, false>,
               dynamic_function_traits<std::uint16_t, true>>;

} // namespace ihist
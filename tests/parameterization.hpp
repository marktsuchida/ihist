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
                                  std::size_t, std::size_t, std::size_t,
                                  std::size_t, std::size_t, std::uint32_t *,
                                  std::size_t);

    static constexpr tuning_parameters tuning{Stripes, Unroll};

    template <bool UseMask, unsigned Bits, unsigned LoBit, std::size_t Stride,
              std::size_t... ComponentOffsets>
    static constexpr hist_func_type *hist_func =
        MT ? (Stripes == 0 ? hist_unoptimized_mt<T, UseMask, Bits, LoBit,
                                                 Stride, ComponentOffsets...>
                           : hist_striped_mt<tuning, T, UseMask, Bits, LoBit,
                                             Stride, ComponentOffsets...>)
           : (Stripes == 0 ? hist_unoptimized_st<T, UseMask, Bits, LoBit,
                                                 Stride, ComponentOffsets...>
                           : hist_striped_st<tuning, T, UseMask, Bits, LoBit,
                                             Stride, ComponentOffsets...>);

    template <bool UseMask, unsigned Bits, unsigned LoBit, std::size_t Stride,
              std::size_t... ComponentOffsets>
    static constexpr histxy_func_type *histxy_func =
        MT ? Stripes == 0 ? histxy_unoptimized_mt<T, UseMask, Bits, LoBit,
                                                  Stride, ComponentOffsets...>
                          : histxy_striped_mt<tuning, T, UseMask, Bits, LoBit,
                                              Stride, ComponentOffsets...>
        : Stripes == 0 ? histxy_unoptimized_st<T, UseMask, Bits, LoBit, Stride,
                                               ComponentOffsets...>
                       : histxy_striped_st<tuning, T, UseMask, Bits, LoBit,
                                           Stride, ComponentOffsets...>;
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

} // namespace ihist
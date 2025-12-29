/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

// This produces portably deterministic data given the same seed.
template <typename T, unsigned Bits = 8 * sizeof(T)>
auto generate_random_data(std::size_t count, std::uint32_t seed)
    -> std::vector<T> {
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(T) <= 8);
    static_assert(Bits <= 8 * sizeof(T));

    // We cannot use std::uniform_int_distribution because it may behave
    // differently depending on the platform, and also does not support 8-bit
    // integers. Instead, we take the low bits.

    std::mt19937_64 engine(seed);
    std::vector<T> data;
    data.resize(count);
    constexpr auto MASK = (1uLL << Bits) - 1;
    std::generate(data.begin(), data.end(),
                  [&] { return static_cast<T>(engine() & MASK); });
    return data;
}

// Reproducible tests!
constexpr std::uint32_t TEST_SEED = 1343208745u;

template <typename T, unsigned Bits = 8 * sizeof(T)>
auto test_data(std::size_t count) -> std::vector<T> {
    return generate_random_data<T, Bits>(count, TEST_SEED);
}
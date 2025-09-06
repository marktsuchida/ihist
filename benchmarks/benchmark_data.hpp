/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist::bench {

template <typename T>
auto generate_data(std::size_t bits, std::size_t count, float spread_frac)
    -> std::vector<T> {
    static_assert(std::is_unsigned_v<T>);
    std::size_t const maximum = (1uLL << bits) - 1;
    std::size_t const mean = maximum / 2;

    if (spread_frac <= 0.0f) {
        return std::vector<T>(count, mean);
    }

    auto const half_spread = std::clamp<std::size_t>(
        std::llroundf(0.5f * spread_frac * static_cast<float>(maximum)), 0,
        mean);

    std::mt19937 engine;
    std::uniform_int_distribution<std::size_t> dist(mean - half_spread,
                                                    mean + half_spread);

    // Since this is just for a benchmark, we cheat, for speed, by repeating a
    // pattern.
    std::vector<T> population(1 << 16);

    std::generate(population.begin(), population.end(), [&] {
        for (;;) {
            auto const v = dist(engine);
            if (v <= maximum) {
                return static_cast<T>(v);
            }
        }
    });

    std::vector<T> data;
    data.reserve(count / population.size() * population.size());
    while (data.size() < count) {
        data.insert(data.end(), population.begin(), population.end());
    }
    data.resize(count);
    return data;
}

inline auto generate_circle_mask(std::intptr_t width, std::intptr_t height)
    -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> mask(width * height);
    auto const center_x = width / 2;
    auto const center_y = height / 2;
    for (std::intptr_t y = 0; y < height; ++y) {
        for (std::intptr_t x = 0; x < width; ++x) {
            auto const xx =
                (x - center_x) * (x - center_x) * center_y * center_y;
            auto const yy =
                (y - center_y) * (y - center_y) * center_x * center_x;
            bool is_inside =
                xx + yy < center_x * center_x * center_y * center_y;
            mask[x + y * width] = static_cast<std::uint8_t>(is_inside);
        }
    }
    return mask;
}

} // namespace ihist::bench
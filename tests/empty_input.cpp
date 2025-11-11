/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "parameterization.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace ihist {

TEMPLATE_LIST_TEST_CASE("empty-input", "", test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;
    std::vector<std::uint32_t> hist(NBINS);
    std::vector<std::uint32_t> hist3(3 * NBINS);
    std::vector<std::uint32_t> const expected(NBINS);
    std::vector<std::uint32_t> const expected3(3 * NBINS);

    SECTION("1d") {
        SECTION("mono") {
            constexpr auto *hist_func =
                traits::template hist_func<false, BITS, 0, 1, 0>;
            hist_func(nullptr, nullptr, 0, hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *hist_func =
                traits::template hist_func<false, BITS, 0, 4, 3, 0, 1>;
            hist_func(nullptr, nullptr, 0, hist3.data(), 1);
            CHECK(hist3 == expected3);
        }
    }

    SECTION("1d-mask") {
        SECTION("empty-data") {
            SECTION("mono") {
                constexpr auto *hist_func =
                    traits::template hist_func<true, BITS, 0, 1, 0>;
                hist_func(nullptr, nullptr, 0, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *hist_func =
                    traits::template hist_func<true, BITS, 0, 4, 3, 0, 1>;
                hist_func(nullptr, nullptr, 0, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }

        SECTION("empty-mask") {
            SECTION("mono") {
                constexpr auto *hist_func =
                    traits::template hist_func<true, BITS, 0, 1, 0>;
                std::vector<T> data(10);
                std::vector<std::uint8_t> mask(10);
                hist_func(data.data(), mask.data(), 10, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *hist_func =
                    traits::template hist_func<true, BITS, 0, 4, 3, 0, 1>;
                std::vector<T> data(4 * 10);
                std::vector<std::uint8_t> mask(10);
                hist_func(data.data(), mask.data(), 10, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }
    }

    SECTION("2d") {
        SECTION("empty-data") {
            SECTION("mono") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, BITS, 0, 1, 0>;
                histxy_func(nullptr, nullptr, 0, 0, 42, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, BITS, 0, 4, 3, 0, 1>;
                histxy_func(nullptr, nullptr, 0, 0, 42, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }

        SECTION("empty-roi") {
            SECTION("mono") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, BITS, 0, 1, 0>;
                std::vector<T> data(6);
                histxy_func(data.data() + 4, nullptr, 0, 0, 3, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<false, BITS, 0, 4, 3, 0, 1>;
                std::vector<T> data(4 * 6);
                histxy_func(data.data() + 4 * 4, nullptr, 0, 0, 3,
                            hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }
    }

    SECTION("2d-mask") {
        SECTION("empty-data") {
            SECTION("mono") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 1, 0>;
                histxy_func(nullptr, nullptr, 0, 0, 42, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
                histxy_func(nullptr, nullptr, 0, 0, 42, hist3.data(), 1);
                CHECK(hist3 == expected3);
            }
        }

        SECTION("empty-roi") {
            SECTION("mono") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 1, 0>;
                std::vector<T> data(6);
                std::vector<std::uint8_t> mask(6, 1);
                histxy_func(data.data() + 4, mask.data() + 4, 0, 0, 3,
                            hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
                std::vector<T> data(4 * 6);
                std::vector<std::uint8_t> mask(6, 1);
                histxy_func(data.data() + 4 * 4, mask.data(), 0, 0, 3,
                            hist3.data() + 4, 1);
                CHECK(hist3 == expected3);
            }
        }

        SECTION("empty-mask") {
            SECTION("mono") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 1, 0>;
                std::vector<T> data(6);
                std::vector<std::uint8_t> mask(6);
                histxy_func(data.data(), mask.data(), 2, 3, 3, hist.data(), 1);
                CHECK(hist == expected);
            }
            SECTION("multi") {
                constexpr auto *histxy_func =
                    traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
                std::vector<T> data(4 * 6);
                std::vector<std::uint8_t> mask(6);
                histxy_func(data.data(), mask.data(), 2, 3, 3, hist3.data(),
                            1);
                CHECK(hist3 == expected3);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("dynamic-empty-input", "", dynamic_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr std::size_t indices[] = {0, 1};

    constexpr auto FULL_BITS = 8 * sizeof(T);
    constexpr auto FULL_NBINS = 1 << FULL_BITS;
    constexpr auto HALF_BITS = FULL_BITS / 2;
    constexpr auto HALF_NBINS = 1 << HALF_BITS;
    constexpr auto HALF_SHIFT = FULL_BITS / 4;

    SECTION("fullbits") {
        std::vector<std::uint32_t> hist(2 * FULL_NBINS);
        std::vector<std::uint32_t> const expected(2 * FULL_NBINS);

        SECTION("empty-data") {
            traits::template histxy_dynamic<false, FULL_BITS, 0>(
                nullptr, nullptr, 0, 0, 42, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty-roi") {
            std::vector<T> data(2 * 6);
            traits::template histxy_dynamic<false, FULL_BITS, 0>(
                data.data() + 2 * 4, nullptr, 0, 0, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty-mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template histxy_dynamic<true, FULL_BITS, 0>(
                data.data(), mask.data(), 2, 3, 3, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
    }

    SECTION("halfbits") {
        std::vector<std::uint32_t> hist(2 * HALF_NBINS);
        std::vector<std::uint32_t> const expected(2 * HALF_NBINS);

        SECTION("empty-data") {
            traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
                nullptr, nullptr, 0, 0, 42, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty-roi") {
            std::vector<T> data(2 * 6);
            traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
                data.data() + 2 * 4, nullptr, 0, 0, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty-mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template histxy_dynamic<true, HALF_BITS, HALF_SHIFT>(
                data.data(), mask.data(), 2, 3, 3, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
    }
}

} // namespace ihist
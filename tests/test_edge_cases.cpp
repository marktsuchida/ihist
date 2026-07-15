/*
 * This file is part of ihist
 * Copyright 2025 Board of Regents of the University of Wisconsin System
 * SPDX-License-Identifier: MIT
 */

#include <ihist.hpp>

#include "gen_data.hpp"
#include "parameterization.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ihist {

TEMPLATE_LIST_TEST_CASE("empty input leaves histogram unchanged", "",
                        test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;
    std::vector<std::uint32_t> hist(NBINS);
    std::vector<std::uint32_t> hist3(3 * NBINS);
    std::vector<std::uint32_t> const expected(NBINS);
    std::vector<std::uint32_t> const expected3(3 * NBINS);

    SECTION("1d zero-size data") {
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

    SECTION("1d zero-size data with mask function") {
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

    SECTION("1d all-zero mask excludes all pixels") {
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

    SECTION("2d zero-size data") {
        SECTION("mono") {
            constexpr auto *histxy_func =
                traits::template histxy_func<false, BITS, 0, 1, 0>;
            histxy_func(nullptr, nullptr, 0, 0, 42, 42, hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *histxy_func =
                traits::template histxy_func<false, BITS, 0, 4, 3, 0, 1>;
            histxy_func(nullptr, nullptr, 0, 0, 42, 42, hist3.data(), 1);
            CHECK(hist3 == expected3);
        }
    }

    SECTION("2d empty ROI (0x0 within larger image)") {
        SECTION("mono") {
            constexpr auto *histxy_func =
                traits::template histxy_func<false, BITS, 0, 1, 0>;
            std::vector<T> data(6);
            histxy_func(data.data() + 4, nullptr, 0, 0, 3, 3, hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *histxy_func =
                traits::template histxy_func<false, BITS, 0, 4, 3, 0, 1>;
            std::vector<T> data(4 * 6);
            histxy_func(data.data() + 4 * 4, nullptr, 0, 0, 3, 3, hist3.data(),
                        1);
            CHECK(hist3 == expected3);
        }
    }

    SECTION("2d zero-size with mask function") {
        SECTION("mono") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 1, 0>;
            histxy_func(nullptr, nullptr, 0, 0, 42, 42, hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
            histxy_func(nullptr, nullptr, 0, 0, 42, 42, hist3.data(), 1);
            CHECK(hist3 == expected3);
        }
    }

    SECTION("2d empty ROI with mask") {
        SECTION("mono") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 1, 0>;
            std::vector<T> data(6);
            std::vector<std::uint8_t> mask(6, 1);
            histxy_func(data.data() + 4, mask.data() + 4, 0, 0, 3, 3,
                        hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
            std::vector<T> data(4 * 6);
            std::vector<std::uint8_t> mask(6, 1);
            histxy_func(data.data() + 4 * 4, mask.data(), 0, 0, 3, 3,
                        hist3.data() + 4, 1);
            CHECK(hist3 == expected3);
        }
    }

    SECTION("2d all-zero mask excludes all pixels") {
        SECTION("mono") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 1, 0>;
            std::vector<T> data(6);
            std::vector<std::uint8_t> mask(6);
            histxy_func(data.data(), mask.data(), 2, 3, 3, 3, hist.data(), 1);
            CHECK(hist == expected);
        }
        SECTION("multi") {
            constexpr auto *histxy_func =
                traits::template histxy_func<true, BITS, 0, 4, 3, 0, 1>;
            std::vector<T> data(4 * 6);
            std::vector<std::uint8_t> mask(6);
            histxy_func(data.data(), mask.data(), 2, 3, 3, 3, hist3.data(), 1);
            CHECK(hist3 == expected3);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("dynamic histogram empty input", "",
                        dynamic_test_traits_list) {
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

        SECTION("zero-size data") {
            traits::template histxy_dynamic<false, FULL_BITS, 0>(
                nullptr, nullptr, 0, 0, 42, 42, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty ROI") {
            std::vector<T> data(2 * 6);
            traits::template histxy_dynamic<false, FULL_BITS, 0>(
                data.data() + 2 * 4, nullptr, 0, 0, 3, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("all-zero mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template histxy_dynamic<true, FULL_BITS, 0>(
                data.data(), mask.data(), 2, 3, 3, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("1d zero-size data") {
            traits::template hist_dynamic<false, FULL_BITS, 0>(
                nullptr, nullptr, 0, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("1d all-zero mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template hist_dynamic<true, FULL_BITS, 0>(
                data.data(), mask.data(), 6, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
    }

    SECTION("halfbits") {
        std::vector<std::uint32_t> hist(2 * HALF_NBINS);
        std::vector<std::uint32_t> const expected(2 * HALF_NBINS);

        SECTION("zero-size data") {
            traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
                nullptr, nullptr, 0, 0, 42, 42, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("empty ROI") {
            std::vector<T> data(2 * 6);
            traits::template histxy_dynamic<false, HALF_BITS, HALF_SHIFT>(
                data.data() + 2 * 4, nullptr, 0, 0, 3, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("all-zero mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template histxy_dynamic<true, HALF_BITS, HALF_SHIFT>(
                data.data(), mask.data(), 2, 3, 3, 3, 2, 2, indices,
                hist.data());
            CHECK(hist == expected);
        }
        SECTION("1d zero-size data") {
            traits::template hist_dynamic<false, HALF_BITS, HALF_SHIFT>(
                nullptr, nullptr, 0, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
        SECTION("1d all-zero mask") {
            std::vector<T> data(2 * 6);
            std::vector<std::uint8_t> mask(6);
            traits::template hist_dynamic<true, HALF_BITS, HALF_SHIFT>(
                data.data(), mask.data(), 6, 2, 2, indices, hist.data());
            CHECK(hist == expected);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("1d dynamic histogram matches reference", "",
                        dynamic_test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr std::size_t indices[] = {0, 1};
    constexpr std::size_t n_components = 2;
    constexpr std::size_t size = 4099;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    auto const data = test_data<T>(n_components * size);
    auto const mask = test_data<std::uint8_t, 1>(size);

    SECTION("nomask") {
        std::vector<std::uint32_t> ref(n_components * NBINS);
        constexpr auto *ref_func =
            hist_unoptimized_st<T, false, BITS, 0, n_components, 0, 1>;
        ref_func(data.data(), nullptr, size, ref.data(), 1);

        std::vector<std::uint32_t> hist(n_components * NBINS);
        traits::template hist_dynamic<false, BITS, 0>(
            data.data(), nullptr, size, n_components, n_components, indices,
            hist.data());
        CHECK(hist == ref);
    }

    SECTION("mask") {
        std::vector<std::uint32_t> ref(n_components * NBINS);
        constexpr auto *ref_func =
            hist_unoptimized_st<T, true, BITS, 0, n_components, 0, 1>;
        ref_func(data.data(), mask.data(), size, ref.data(), 1);

        std::vector<std::uint32_t> hist(n_components * NBINS);
        traits::template hist_dynamic<true, BITS, 0>(
            data.data(), mask.data(), size, n_components, n_components,
            indices, hist.data());
        CHECK(hist == ref);
    }
}

TEMPLATE_LIST_TEST_CASE("single pixel input", "", test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    T const value = 42;
    std::vector<T> data(1, value);

    SECTION("1d") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = 1;

        constexpr auto *hist_func =
            traits::template hist_func<false, BITS, 0, 1, 0>;
        hist_func(data.data(), nullptr, 1, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("2d") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = 1;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data(), nullptr, 1, 1, 1, 1, hist.data(), 1);
        CHECK(hist == expected);
    }
}

TEMPLATE_LIST_TEST_CASE("single row and single column ROIs", "",
                        test_traits_list) {
    using traits = TestType;
    using T = typename traits::value_type;

    constexpr auto BITS = 8 * sizeof(T);
    constexpr auto NBINS = 1 << BITS;

    T const value = 17;
    constexpr std::size_t image_width = 10;
    constexpr std::size_t image_height = 8;
    std::vector<T> data(image_width * image_height, value);

    SECTION("single row at top") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = image_width;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data(), nullptr, 1, image_width, image_width,
                    image_width, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("single row at bottom") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = image_width;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data() + (image_height - 1) * image_width, nullptr, 1,
                    image_width, image_width, image_width, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("single column at left") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = image_height;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data(), nullptr, image_height, 1, image_width,
                    image_width, hist.data(), 1);
        CHECK(hist == expected);
    }

    SECTION("single column at right") {
        std::vector<std::uint32_t> hist(NBINS);
        std::vector<std::uint32_t> expected(NBINS);
        expected[value] = image_height;

        constexpr auto *histxy_func =
            traits::template histxy_func<false, BITS, 0, 1, 0>;
        histxy_func(data.data() + (image_width - 1), nullptr, image_height, 1,
                    image_width, image_width, hist.data(), 1);
        CHECK(hist == expected);
    }
}

} // namespace ihist

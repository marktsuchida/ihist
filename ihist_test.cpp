#include <ihist.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist::test {
namespace {

// This produces portably deterministic data given the same seed.
template <typename T, unsigned BITS = 8 * sizeof(T)>
auto generate_random_data(std::size_t count, std::uint32_t seed)
    -> std::vector<T> {
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(T) <= 8);
    static_assert(BITS <= 8 * sizeof(T));

    // We cannot use std::uniform_int_distribution because it may behave
    // differently depending on the platform, and also does not support 8-bit
    // integers. Instead, we take the low bits.

    std::mt19937_64 engine(seed);
    std::vector<T> data;
    data.resize(count);
    constexpr auto MASK = (1uLL << BITS) - 1;
    std::generate(data.begin(), data.end(),
                  [&] { return static_cast<T>(engine() & MASK); });
    return data;
}

// Reproducible tests!
constexpr std::uint32_t TEST_SEED = 1343208745u;

template <typename T, unsigned BITS = 8 * sizeof(T)>
auto test_data(std::size_t count) -> std::vector<T> {
    return generate_random_data<T, BITS>(count, TEST_SEED);
}

} // namespace

TEST_CASE("bin_index-full-bits") {
    STATIC_CHECK(internal::bin_index<std::uint8_t>(0) == 0);
    STATIC_CHECK(internal::bin_index<std::uint8_t>(255) == 255);
    STATIC_CHECK(internal::bin_index(std::uint8_t(255)) == 255);
    STATIC_CHECK(internal::bin_index<std::uint16_t>(0) == 0);
    STATIC_CHECK(internal::bin_index<std::uint16_t>(65535) == 65535);
    STATIC_CHECK(internal::bin_index(std::uint16_t(65535)) == 65535);
}

TEST_CASE("bin_index-lo-bits") {
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12>(0x0fff) == 0x0fff);
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12>(0xffff) == 0x0fff);
}

TEST_CASE("bin_index-hi-bits") {
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12, 4>(0xfff0) == 0x0fff);
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12, 4>(0xffff) == 0x0fff);
}

TEST_CASE("bin_index_himask-lo-bits") {
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(0, 0) == 0);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(1, 0) == 1);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(4095, 0) ==
                 4095);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(4096, 0) ==
                 4096);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(4097, 0) ==
                 4096);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(65535, 0) ==
                 4096);

    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(0xaffd, 0xa) ==
                 0x0ffd);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 12>(0xbffd, 0xa) ==
                 0x1000);
}

TEST_CASE("bin_index_himask-mid-bits") {
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0x0000, 0) ==
                 0);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0x0010, 0) ==
                 1);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0x0ff0, 0) ==
                 0xff);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0x1000, 0) ==
                 256);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0x1010, 0) ==
                 256);
    STATIC_CHECK(internal::bin_index_himask<std::uint16_t, 8, 4>(0xffff, 0) ==
                 256);
}

TEMPLATE_TEST_CASE("empty-data", "", std::uint8_t, std::uint16_t) {
    auto const hist_func = GENERATE(hist_naive_unfiltered<TestType>,
                                    hist_striped_unfiltered<0, TestType>,
                                    hist_striped_unfiltered<1, TestType>,
                                    hist_striped_unfiltered<2, TestType>,
                                    hist_striped_unfiltered<3, TestType>,
                                    hist_striped_unfiltered<4, TestType>,
                                    hist_naive_mt_unfiltered<TestType>,
                                    hist_striped_mt_unfiltered<0, TestType>,
                                    hist_striped_mt_unfiltered<1, TestType>,
                                    hist_striped_mt_unfiltered<2, TestType>,
                                    hist_striped_mt_unfiltered<3, TestType>,
                                    hist_striped_mt_unfiltered<4, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(nullptr, 0, hist.data());
    for (std::size_t i = 0; i < NBINS; ++i) {
        REQUIRE(hist[i] == 0);
    }
}

TEMPLATE_TEST_CASE("const-data", "", std::uint8_t, std::uint16_t) {
    auto const hist_func = GENERATE(hist_naive_unfiltered<TestType>,
                                    hist_striped_unfiltered<0, TestType>,
                                    hist_striped_unfiltered<1, TestType>,
                                    hist_striped_unfiltered<2, TestType>,
                                    hist_striped_unfiltered<3, TestType>,
                                    hist_striped_unfiltered<4, TestType>,
                                    hist_naive_mt_unfiltered<TestType>,
                                    hist_striped_mt_unfiltered<0, TestType>,
                                    hist_striped_mt_unfiltered<1, TestType>,
                                    hist_striped_mt_unfiltered<2, TestType>,
                                    hist_striped_mt_unfiltered<3, TestType>,
                                    hist_striped_mt_unfiltered<4, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    std::size_t size = GENERATE(1, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);
    CAPTURE(size, value);

    std::vector<TestType> data(size, value);
    hist_func(data.data(), data.size(), hist.data());
    for (std::size_t i = 0; i < NBINS; ++i) {
        if (i == value) {
            CHECK(hist[i] == size);
        } else {
            CHECK(hist[i] == 0);
        }
    }
}

TEMPLATE_TEST_CASE("random-data", "", std::uint8_t, std::uint16_t) {
    auto const hist_func = GENERATE(hist_naive_unfiltered<TestType>,
                                    hist_striped_unfiltered<0, TestType>,
                                    hist_striped_unfiltered<1, TestType>,
                                    hist_striped_unfiltered<2, TestType>,
                                    hist_striped_unfiltered<3, TestType>,
                                    hist_striped_unfiltered<4, TestType>,
                                    hist_naive_mt_unfiltered<TestType>,
                                    hist_striped_mt_unfiltered<0, TestType>,
                                    hist_striped_mt_unfiltered<1, TestType>,
                                    hist_striped_mt_unfiltered<2, TestType>,
                                    hist_striped_mt_unfiltered<3, TestType>,
                                    hist_striped_mt_unfiltered<4, TestType>);
    auto const ref_func = hist_naive_unfiltered<TestType>;

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    auto const data = test_data<TestType>(1 << (22 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), data.size(), ref.data());
    assert(hist == ref);
}

TEMPLATE_TEST_CASE("low-bits-random-data-clean-safe", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func = GENERATE(hist_naive_himask<TestType, BITS>,
                              hist_striped_himask<2, TestType, BITS>,
                              hist_naive_mt_himask<TestType, BITS>,
                              hist_striped_mt_himask<2, TestType, BITS>);
    auto ref_func = hist_naive_unfiltered<TestType, BITS>;

    constexpr auto NBINS = 1 << BITS;
    auto const data = test_data<TestType, BITS>(1 << (22 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), 0, hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), data.size(), ref.data());
    assert(hist == ref);
}

TEMPLATE_TEST_CASE("low-bits-random-data-unclean-safe", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func = GENERATE(hist_naive_himask<TestType, BITS>,
                              hist_striped_himask<2, TestType, BITS>,
                              hist_naive_mt_himask<TestType, BITS>,
                              hist_striped_mt_himask<2, TestType, BITS>);
    auto ref_func = hist_naive_unfiltered<TestType, BITS>;

    constexpr auto NBINS = 1 << BITS;

    // Test with 6/14-bit data, exceeding the histogram range:
    auto const data =
        test_data<TestType, BITS + 2>(1 << (22 - sizeof(TestType)));

    // Cleaned data (limited to values in 4/12-bit range) should produce same
    // result as hi_to_match=0.
    auto const clean_data = [&] {
        std::vector<TestType> clean;
        clean.reserve(data.size() / 3);
        std::copy_if(data.begin(), data.end(), std::back_inserter(clean),
                     [&](TestType v) { return v < (1 << BITS); });
        return clean;
    }();

    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), 0, hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(clean_data.data(), clean_data.size(), ref.data());
    assert(hist == ref);
}

} // namespace ihist::test
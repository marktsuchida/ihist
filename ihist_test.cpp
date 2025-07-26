#include <ihist.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>
#include <vector>

namespace ihist::test {
namespace {

// This produces portably deterministic data given the same seed.
template <typename T>
auto generate_random_data(std::size_t count, std::uint32_t seed)
    -> std::vector<T> {
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(T) <= 8);

    // We cannot use std::uniform_int_distribution because it may behave
    // differently depending on the platform, and also does not support 8-bit
    // integers. Instead, we take the low bits.

    std::mt19937_64 engine(seed);
    std::vector<T> data;
    data.resize(count);
    std::generate(data.begin(), data.end(),
                  [&] { return static_cast<T>(engine()); });
    return data;
}

// Reproducible tests!
constexpr std::uint32_t TEST_SEED = 1343208745u;

template <typename T> auto test_data(std::size_t count) -> std::vector<T> {
    return generate_random_data<T>(count, TEST_SEED);
}

enum class hist_algo {
    NAIVE,
    STRIPED0,
    STRIPED1,
    STRIPED2,
    STRIPED3,
    STRIPED4,
    NAIVE_MT,
    STRIPED0_MT,
    STRIPED1_MT,
    STRIPED2_MT,
    STRIPED3_MT,
    STRIPED4_MT,
};

auto operator<<(std::ostream &s, hist_algo const &algo) -> std::ostream & {
    switch (algo) {
    case hist_algo::NAIVE:
        return s << "hist_naive<T>";
    case hist_algo::STRIPED0:
        return s << "hist_striped<T, 0>";
    case hist_algo::STRIPED1:
        return s << "hist_striped<T, 1>";
    case hist_algo::STRIPED2:
        return s << "hist_striped<T, 2>";
    case hist_algo::STRIPED3:
        return s << "hist_striped<T, 3>";
    case hist_algo::STRIPED4:
        return s << "hist_striped<T, 4>";
    case hist_algo::NAIVE_MT:
        return s << "hist_naive_mt<T>";
    case hist_algo::STRIPED0_MT:
        return s << "hist_striped_mt<T, 0>";
    case hist_algo::STRIPED1_MT:
        return s << "hist_striped_mt<T, 1>";
    case hist_algo::STRIPED2_MT:
        return s << "hist_striped_mt<T, 2>";
    case hist_algo::STRIPED3_MT:
        return s << "hist_striped_mt<T, 3>";
    case hist_algo::STRIPED4_MT:
        return s << "hist_striped_mt<T, 4>";
    }
}

template <typename T>
using hist_func = void (*)(T const *, std::size_t, std::uint32_t *);

template <typename T> auto hist_algo_func(hist_algo algo) -> hist_func<T> {
    switch (algo) {
    case hist_algo::NAIVE:
        return hist_naive<T>;
    case hist_algo::STRIPED0:
        return hist_striped<T, 0>;
    case hist_algo::STRIPED1:
        return hist_striped<T, 1>;
    case hist_algo::STRIPED2:
        return hist_striped<T, 2>;
    case hist_algo::STRIPED3:
        return hist_striped<T, 3>;
    case hist_algo::STRIPED4:
        return hist_striped<T, 4>;
    case hist_algo::NAIVE_MT:
        return hist_naive_mt<T>;
    case hist_algo::STRIPED0_MT:
        return hist_striped_mt<T, 0>;
    case hist_algo::STRIPED1_MT:
        return hist_striped_mt<T, 1>;
    case hist_algo::STRIPED2_MT:
        return hist_striped_mt<T, 2>;
    case hist_algo::STRIPED3_MT:
        return hist_striped_mt<T, 3>;
    case hist_algo::STRIPED4_MT:
        return hist_striped_mt<T, 4>;
    }
}

} // namespace

TEST_CASE("bin_index_full_bits") {
    STATIC_CHECK(internal::bin_index<std::uint8_t>(0) == 0);
    STATIC_CHECK(internal::bin_index<std::uint8_t>(255) == 255);
    STATIC_CHECK(internal::bin_index(std::uint8_t(255)) == 255);
    STATIC_CHECK(internal::bin_index<std::uint16_t>(0) == 0);
    STATIC_CHECK(internal::bin_index<std::uint16_t>(65535) == 65535);
    STATIC_CHECK(internal::bin_index(std::uint16_t(65535)) == 65535);
}

TEST_CASE("bin_index_lo_bits") {
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12>(0x0fff) == 0x0fff);
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12>(0xffff) == 0x0fff);
}

TEST_CASE("bin_index_hi_bits") {
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12, 4>(0xfff0) == 0x0fff);
    STATIC_CHECK(internal::bin_index<std::uint16_t, 12, 4>(0xffff) == 0x0fff);
}

TEMPLATE_TEST_CASE("empty-data", "", std::uint8_t, std::uint16_t) {
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::NAIVE_MT,
                 hist_algo::STRIPED0_MT, hist_algo::STRIPED1_MT,
                 hist_algo::STRIPED2_MT, hist_algo::STRIPED3_MT);
    auto const hist_func = hist_algo_func<TestType>(algo);
    CAPTURE(algo);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(nullptr, 0, hist.data());
    for (std::size_t i = 0; i < NBINS; ++i) {
        REQUIRE(hist[i] == 0);
    }
}

TEMPLATE_TEST_CASE("const-data", "", std::uint8_t, std::uint16_t) {
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::NAIVE_MT,
                 hist_algo::STRIPED0_MT, hist_algo::STRIPED1_MT,
                 hist_algo::STRIPED2_MT, hist_algo::STRIPED3_MT);
    auto const hist_func = hist_algo_func<TestType>(algo);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    std::size_t size = GENERATE(1, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);
    CAPTURE(algo, size, value);

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
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::STRIPED4,
                 hist_algo::NAIVE_MT, hist_algo::STRIPED0_MT,
                 hist_algo::STRIPED1_MT, hist_algo::STRIPED2_MT,
                 hist_algo::STRIPED3_MT, hist_algo::STRIPED4_MT);
    auto const hist_func = hist_algo_func<TestType>(algo);
    CAPTURE(algo);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    auto const data = test_data<TestType>(1 << (22 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    hist_naive(data.data(), data.size(), ref.data());
    assert(hist == ref);
}

} // namespace ihist::test
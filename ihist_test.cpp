#include <ihist.hpp>

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
        return s << "hist8_naive";
    case hist_algo::STRIPED0:
        return s << "hist8_striped<0>";
    case hist_algo::STRIPED1:
        return s << "hist8_striped<1>";
    case hist_algo::STRIPED2:
        return s << "hist8_striped<2>";
    case hist_algo::STRIPED3:
        return s << "hist8_striped<3>";
    case hist_algo::STRIPED4:
        return s << "hist8_striped<4>";
    case hist_algo::NAIVE_MT:
        return s << "hist8_naive_mt";
    case hist_algo::STRIPED0_MT:
        return s << "hist8_striped_mt<0>";
    case hist_algo::STRIPED1_MT:
        return s << "hist8_striped_mt<1>";
    case hist_algo::STRIPED2_MT:
        return s << "hist8_striped_mt<2>";
    case hist_algo::STRIPED3_MT:
        return s << "hist8_striped_mt<3>";
    case hist_algo::STRIPED4_MT:
        return s << "hist8_striped_mt<4>";
    }
}

using hist8_func = void (*)(std::uint8_t const *, std::size_t,
                            std::uint32_t *);

auto hist8_algo_func(hist_algo algo) -> hist8_func {
    switch (algo) {
    case hist_algo::NAIVE:
        return hist8_naive;
    case hist_algo::STRIPED0:
        return hist8_striped<0>;
    case hist_algo::STRIPED1:
        return hist8_striped<1>;
    case hist_algo::STRIPED2:
        return hist8_striped<2>;
    case hist_algo::STRIPED3:
        return hist8_striped<3>;
    case hist_algo::STRIPED4:
        return hist8_striped<4>;
    case hist_algo::NAIVE_MT:
        return hist8_naive_mt;
    case hist_algo::STRIPED0_MT:
        return hist8_striped_mt<0>;
    case hist_algo::STRIPED1_MT:
        return hist8_striped_mt<1>;
    case hist_algo::STRIPED2_MT:
        return hist8_striped_mt<2>;
    case hist_algo::STRIPED3_MT:
        return hist8_striped_mt<3>;
    case hist_algo::STRIPED4_MT:
        return hist8_striped_mt<4>;
    }
}

} // namespace

TEST_CASE("8bit-empty") {
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::NAIVE_MT,
                 hist_algo::STRIPED0_MT, hist_algo::STRIPED1_MT,
                 hist_algo::STRIPED2_MT, hist_algo::STRIPED3_MT);
    auto const hist8 = hist8_algo_func(algo);
    CAPTURE(algo);

    std::array<std::uint32_t, 256> hist{};
    hist8(nullptr, 0, hist.data());
    for (std::size_t i = 0; i < 256; ++i) {
        REQUIRE(hist[i] == 0);
    }
}

TEST_CASE("8bit-const") {
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::NAIVE_MT,
                 hist_algo::STRIPED0_MT, hist_algo::STRIPED1_MT,
                 hist_algo::STRIPED2_MT, hist_algo::STRIPED3_MT);
    auto const hist8 = hist8_algo_func(algo);

    std::array<std::uint32_t, 256> hist{};
    std::size_t size = GENERATE(1, 7, 1000);
    std::uint8_t value = GENERATE(0, 1, 255);
    CAPTURE(algo, size, value);

    std::vector<std::uint8_t> data(size, value);
    hist8(data.data(), data.size(), hist.data());
    for (std::size_t i = 0; i < 256; ++i) {
        if (i == value) {
            CHECK(hist[i] == size);
        } else {
            CHECK(hist[i] == 0);
        }
    }
}

TEST_CASE("8bit-random") {
    auto const algo =
        GENERATE(hist_algo::NAIVE, hist_algo::STRIPED0, hist_algo::STRIPED1,
                 hist_algo::STRIPED2, hist_algo::STRIPED3, hist_algo::STRIPED4,
                 hist_algo::NAIVE_MT, hist_algo::STRIPED0_MT,
                 hist_algo::STRIPED1_MT, hist_algo::STRIPED2_MT,
                 hist_algo::STRIPED3_MT, hist_algo::STRIPED4_MT);
    auto const hist8 = hist8_algo_func(algo);
    CAPTURE(algo);

    std::vector<std::uint8_t> data = test_data<std::uint8_t>(1 << 22);
    std::array<std::uint32_t, 256> hist{};
    hist8(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, 256> ref{};
    hist8_naive(data.data(), data.size(), ref.data());
    assert(hist == ref);
}

} // namespace ihist::test
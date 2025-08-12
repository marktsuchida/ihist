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

namespace ihist {

namespace {

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

} // namespace

namespace internal {

TEST_CASE("bin_index-full-bits") {
    STATIC_CHECK(bin_index<std::uint8_t>(0) == 0);
    STATIC_CHECK(bin_index<std::uint8_t>(255) == 255);
    STATIC_CHECK(bin_index(std::uint8_t(255)) == 255);
    STATIC_CHECK(bin_index<std::uint16_t>(0) == 0);
    STATIC_CHECK(bin_index<std::uint16_t>(65535) == 65535);
    STATIC_CHECK(bin_index(std::uint16_t(65535)) == 65535);
}

TEST_CASE("bin_index-lo-bits") {
    STATIC_CHECK(bin_index<std::uint16_t, 12>(0x0fff) == 0x0fff);
    STATIC_CHECK(bin_index<std::uint16_t, 12>(0xffff) == 0x0fff);
}

TEST_CASE("bin_index-hi-bits") {
    STATIC_CHECK(bin_index<std::uint16_t, 12, 4>(0xfff0) == 0x0fff);
    STATIC_CHECK(bin_index<std::uint16_t, 12, 4>(0xffff) == 0x0fff);
}

TEMPLATE_TEST_CASE_SIG("bin_index_himask-lo-bits", "",
                       ((bool Branchless), Branchless), (false), (true)) {
    static constexpr tuning_parameters TUNING{Branchless};
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(0, 0) == 0);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(1, 0) == 1);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(4095, 0) == 4095);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(4096, 0) == 4096);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(4097, 0) == 4096);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(65535, 0) ==
                 4096);

    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(0xaffd, 0xa) ==
                 0x0ffd);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 12>(0xbffd, 0xa) ==
                 0x1000);
}

TEMPLATE_TEST_CASE_SIG("bin_index_himask-mid-bits", "",
                       ((bool Branchless), Branchless), (false), (true)) {
    static constexpr tuning_parameters TUNING{Branchless};
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0x0000, 0) ==
                 0);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0x0010, 0) ==
                 1);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0x0ff0, 0) ==
                 0xff);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0x1000, 0) ==
                 256);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0x1010, 0) ==
                 256);
    STATIC_CHECK(bin_index_himask<TUNING, std::uint16_t, 8, 4>(0xffff, 0) ==
                 256);
}

} // namespace internal

TEMPLATE_TEST_CASE("empty-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func =
        GENERATE(hist_unfiltered_unoptimized_st<TestType>,
                 hist_unfiltered_striped_st<tune0, TestType>,
                 hist_unfiltered_striped_st<tune1, TestType>,
                 hist_unfiltered_striped_st<tune2, TestType>,
                 hist_unfiltered_striped_st<tune3, TestType>,
                 hist_unfiltered_striped_st<tune00, TestType>,
                 hist_unfiltered_striped_st<tune01, TestType>,
                 hist_unfiltered_striped_st<tune02, TestType>,
                 hist_unfiltered_striped_st<tune03, TestType>,
                 hist_unfiltered_unoptimized_mt<TestType>,
                 hist_unfiltered_striped_mt<tune0, TestType>,
                 hist_unfiltered_striped_mt<tune1, TestType>,
                 hist_unfiltered_striped_mt<tune2, TestType>,
                 hist_unfiltered_striped_mt<tune3, TestType>,
                 hist_unfiltered_striped_mt<tune00, TestType>,
                 hist_unfiltered_striped_mt<tune01, TestType>,
                 hist_unfiltered_striped_mt<tune02, TestType>,
                 hist_unfiltered_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(nullptr, 0, hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("const-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func =
        GENERATE(hist_unfiltered_unoptimized_st<TestType>,
                 hist_unfiltered_striped_st<tune0, TestType>,
                 hist_unfiltered_striped_st<tune1, TestType>,
                 hist_unfiltered_striped_st<tune2, TestType>,
                 hist_unfiltered_striped_st<tune3, TestType>,
                 hist_unfiltered_striped_st<tune00, TestType>,
                 hist_unfiltered_striped_st<tune01, TestType>,
                 hist_unfiltered_striped_st<tune02, TestType>,
                 hist_unfiltered_striped_st<tune03, TestType>,
                 hist_unfiltered_unoptimized_mt<TestType>,
                 hist_unfiltered_striped_mt<tune0, TestType>,
                 hist_unfiltered_striped_mt<tune1, TestType>,
                 hist_unfiltered_striped_mt<tune2, TestType>,
                 hist_unfiltered_striped_mt<tune3, TestType>,
                 hist_unfiltered_striped_mt<tune00, TestType>,
                 hist_unfiltered_striped_mt<tune01, TestType>,
                 hist_unfiltered_striped_mt<tune02, TestType>,
                 hist_unfiltered_striped_mt<tune03, TestType>);

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    std::size_t size = GENERATE(1, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);
    CAPTURE(size, value);

    std::vector<TestType> data(size, value);
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref[value] = size;
    CHECK(hist == ref);
}

namespace internal {

TEMPLATE_TEST_CASE("const-data-himask-discard-low", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func =
        GENERATE(hist_himask_unoptimized<TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune0, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune1, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune2, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune3, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune00, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune01, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune02, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_st<tune03, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_unoptimized_mt<TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune0, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune1, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune2, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune3, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune00, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune01, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune02, TestType, BITS, LO_BIT, 1, 0>,
                 hist_himask_striped_mt<tune03, TestType, BITS, LO_BIT, 1, 0>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t size = GENERATE(1, 7, 100);
    TestType lo_bits = GENERATE(0, 1, (1 << LO_BIT) - 1);
    TestType sample = GENERATE(0, 1, NBINS - 1);
    TestType hi_bits =
        GENERATE(0, 1, (1 << (8 * sizeof(TestType) - (BITS + LO_BIT))) - 1);
    TestType value = ((((hi_bits) << BITS) | sample) << LO_BIT) | lo_bits;
    CAPTURE(size, hi_bits, sample, lo_bits, value);

    std::vector<TestType> data(size, value);

    SECTION("matching-hi-mask") {
        std::array<std::uint32_t, NBINS> hist{};
        hist_func(data.data(), data.size(), hi_bits, hist.data());
        std::array<std::uint32_t, NBINS> ref{};
        ref[sample] = size;
        CHECK(hist == ref);
    }

    SECTION("non-matching-hi-mask") {
        std::array<std::uint32_t, NBINS> hist{};
        hist_func(data.data(), data.size(), hi_bits + 1, hist.data());
        std::array<std::uint32_t, NBINS> ref{};
        CHECK(hist == ref);
    }
}

} // namespace internal

TEMPLATE_TEST_CASE("const-data-multicomponent", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func = GENERATE(
        hist_unfiltered_unoptimized_st<TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune0, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune1, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune2, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune3, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune00, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune01, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune02, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune03, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_unoptimized_mt<TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune0, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune1, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune2, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune3, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune00, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune01, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune02, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune03, TestType, BITS, 0, STRIDE, 3, 0,
                                   1>);

    constexpr auto NBINS = 1 << BITS;
    std::size_t size = GENERATE(0, 1, 7, 1000);
    TestType value = GENERATE(0, 1, NBINS - 1);

    constexpr std::array components{1, 2, -1, 0}; // Inverse of (3, 0, 1)
    auto const offset = GENERATE(0, 1, 2, 3);
    auto const component = components[offset];

    CAPTURE(size, value, offset, component);

    std::vector<TestType> data(STRIDE * size);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i % STRIDE == offset) {
            data[i] = value;
        }
    }

    std::vector<std::uint32_t> hist(3 * NBINS);
    hist_func(data.data(), size, hist.data());
    std::vector<std::uint32_t> ref(3 * NBINS);
    for (int c = 0; c < 3; ++c) {
        if (c == component) {
            ref[c * NBINS + value] = size;
        } else {
            ref[c * NBINS + 0] = size;
        }
    }
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data", "", std::uint8_t, std::uint16_t) {
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func =
        GENERATE(hist_unfiltered_unoptimized_st<TestType>,
                 hist_unfiltered_striped_st<tune0, TestType>,
                 hist_unfiltered_striped_st<tune1, TestType>,
                 hist_unfiltered_striped_st<tune2, TestType>,
                 hist_unfiltered_striped_st<tune3, TestType>,
                 hist_unfiltered_striped_st<tune00, TestType>,
                 hist_unfiltered_striped_st<tune01, TestType>,
                 hist_unfiltered_striped_st<tune02, TestType>,
                 hist_unfiltered_striped_st<tune03, TestType>,
                 hist_unfiltered_unoptimized_mt<TestType>,
                 hist_unfiltered_striped_mt<tune0, TestType>,
                 hist_unfiltered_striped_mt<tune1, TestType>,
                 hist_unfiltered_striped_mt<tune2, TestType>,
                 hist_unfiltered_striped_mt<tune3, TestType>,
                 hist_unfiltered_striped_mt<tune00, TestType>,
                 hist_unfiltered_striped_mt<tune01, TestType>,
                 hist_unfiltered_striped_mt<tune02, TestType>,
                 hist_unfiltered_striped_mt<tune03, TestType>);
    auto const ref_func = hist_unfiltered_unoptimized_st<TestType>;

    constexpr auto NBINS = 1 << (8 * sizeof(TestType));
    auto const data = test_data<TestType>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), data.size(), ref.data());
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-filtered", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func =
        GENERATE(hist_filtered_unoptimized_st<TestType, BITS>,
                 hist_filtered_striped_st<untuned_parameters, TestType, BITS>,
                 hist_filtered_unoptimized_mt<TestType, BITS>,
                 hist_filtered_striped_mt<untuned_parameters, TestType, BITS>);
    auto ref_func = hist_unfiltered_unoptimized_st<TestType, BITS>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 4/12-bit data with no spurious high bits:
    auto const data = test_data<TestType, BITS>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), data.size(), ref.data());
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-clean-discard-low", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) / 2;
    constexpr auto LO_BIT = BITS / 2;
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func =
        GENERATE(hist_unfiltered_unoptimized_st<TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune0, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune1, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune2, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune3, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune00, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune01, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune02, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_st<tune03, TestType, BITS, LO_BIT>,
                 hist_unfiltered_unoptimized_mt<TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune0, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune1, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune2, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune3, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune00, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune01, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune02, TestType, BITS, LO_BIT>,
                 hist_unfiltered_striped_mt<tune03, TestType, BITS, LO_BIT>,
                 hist_filtered_unoptimized_st<TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune0, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune1, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune2, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune3, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune00, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune01, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune02, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_st<tune03, TestType, BITS, LO_BIT>,
                 hist_filtered_unoptimized_mt<TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune0, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune1, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune2, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune3, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune00, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune01, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune02, TestType, BITS, LO_BIT>,
                 hist_filtered_striped_mt<tune03, TestType, BITS, LO_BIT>);
    auto ref_func = hist_unfiltered_unoptimized_st<TestType, BITS, LO_BIT>;

    constexpr auto NBINS = 1 << BITS;
    // Test with 6/12-bit data with no spurious high bits (but random low
    // bits):
    auto const data =
        test_data<TestType, BITS + LO_BIT>(1 << (20 - sizeof(TestType)));
    std::array<std::uint32_t, NBINS> hist{};
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(data.data(), data.size(), ref.data());
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-unclean-filtered", "", std::uint8_t,
                   std::uint16_t) {
    constexpr auto BITS = 8 * sizeof(TestType) - 4;
    auto hist_func =
        GENERATE(hist_filtered_unoptimized_st<TestType, BITS>,
                 hist_filtered_striped_st<untuned_parameters, TestType, BITS>,
                 hist_filtered_unoptimized_mt<TestType, BITS>,
                 hist_filtered_striped_mt<untuned_parameters, TestType, BITS>);
    auto ref_func = hist_unfiltered_unoptimized_st<TestType, BITS>;

    constexpr auto NBINS = 1 << BITS;

    // Test with 6/14-bit data, exceeding the histogram range:
    auto const data =
        test_data<TestType, BITS + 2>(1 << (20 - sizeof(TestType)));

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
    hist_func(data.data(), data.size(), hist.data());
    std::array<std::uint32_t, NBINS> ref{};
    ref_func(clean_data.data(), clean_data.size(), ref.data());
    CHECK(hist == ref);
}

TEMPLATE_TEST_CASE("random-data-multicomponent", "", std::uint8_t,
                   std::uint16_t) {
    constexpr unsigned BITS = 8 * sizeof(TestType);
    constexpr std::size_t STRIDE = 4;
    static constexpr tuning_parameters tune0{false, 0};
    static constexpr tuning_parameters tune1{false, 1};
    static constexpr tuning_parameters tune2{false, 2};
    static constexpr tuning_parameters tune3{false, 3};
    static constexpr tuning_parameters tune00{false, 0, 0};
    static constexpr tuning_parameters tune01{false, 0, 1};
    static constexpr tuning_parameters tune02{false, 0, 2};
    static constexpr tuning_parameters tune03{false, 0, 3};
    auto const hist_func = GENERATE(
        hist_unfiltered_unoptimized_st<TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune0, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune1, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune2, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune3, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune00, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune01, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune02, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_st<tune03, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_unoptimized_mt<TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune0, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune1, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune2, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune3, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune00, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune01, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune02, TestType, BITS, 0, STRIDE, 3, 0, 1>,
        hist_unfiltered_striped_mt<tune03, TestType, BITS, 0, STRIDE, 3, 0,
                                   1>);
    auto const ref_func =
        hist_unfiltered_unoptimized_st<TestType, BITS, 0, STRIDE, 3, 0, 1>;

    constexpr auto NBINS = 1 << BITS;
    auto const data = test_data<TestType>(STRIDE << (20 - sizeof(TestType)));
    std::vector<std::uint32_t> hist(3 * NBINS);
    hist_func(data.data(), data.size() / STRIDE, hist.data());
    std::vector<std::uint32_t> ref(3 * NBINS);
    ref_func(data.data(), data.size() / STRIDE, ref.data());
    CHECK(hist == ref);
}

} // namespace ihist
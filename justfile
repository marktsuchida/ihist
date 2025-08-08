help:
    @just --list

exe_suffix := if os_family() == "windows" { ".exe" } else { "" }

#
# oneTBB static library
#

# Note that oneTBB's CMake scripts will warn about building a static library.
# However, the warning only applies if there is a chance that symbols from
# multiple copies of oneTBB may be accessible within a program:
# https://github.com/uxlfoundation/oneTBB/issues/646#issuecomment-966106176
#
# TODO -fvisibility=hidden is not enough to hide TBB API symbols. For the
# Python extension maybe we can use a version script (Linux) or export list
# (macOS) to hide all but the entry point. Failing that, patch TBB to avoid
# __attribute__((visibility("default"))).

tbb_version := '2022.2.0'
tbb_tgz := 'oneapi-tbb-' + tbb_version + '.tar.gz'
tbb_dir := 'oneTBB-' + tbb_version

build-tbb BUILD_TYPE:
    #!/usr/bin/env bash
    set -euxo pipefail
    CMAKE_CXX_FLAGS=''
    case {{BUILD_TYPE}} in
        "release") CMAKE_BUILD_TYPE=Release;;
        "debugoptimized") CMAKE_BUILD_TYPE=RelWithDebInfo;;
        "debug") CMAKE_BUILD_TYPE=Debug;;
        "sanitize")
            CMAKE_BUILD_TYPE=DEBUG
            CMAKE_CXX_FLAGS='-fsanitize=address -fsanitize=undefined'
            ;;
        *) echo "Unknown build type: {{BUILD_TYPE}}" >&2; return 1;;
    esac
    mkdir -p dependencies
    cd dependencies
    if [ ! -f "{{tbb_tgz}}" ]; then
        curl -Lo "{{tbb_tgz}}" \
            https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v{{tbb_version}}.tar.gz
    fi
    tar xf {{tbb_tgz}}
    cd {{tbb_dir}}
    mkdir -p build
    cmake -G Ninja -S . -B build \
      -DCMAKE_INSTALL_PREFIX="$(pwd)/../oneTBB-{{BUILD_TYPE}}" \
      -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
      -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" \
      -DBUILD_SHARED_LIBS=OFF \
      -DTBB_TEST=OFF \
      -DTBBMALLOC_BUILD=OFF \
      -DTBBMALLOC_PROXY_BUILD=OFF
    cmake --build build --target install

clean-tbb:
    rm -rf dependencies/oneTBB-*

configure BUILD_TYPE *FLAGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    if [ ! -d dependencies/oneTBB-{{BUILD_TYPE}} ]; then
        just build-tbb {{BUILD_TYPE}}
    fi
    BUILD_TYPE={{BUILD_TYPE}}
    SANITIZE_FLAGS="-Db_sanitize=none -Db_lundef=true"
    if [ "$BUILD_TYPE" = "sanitize" ]; then
        BUILD_TYPE=debug
        SANITIZE_FLAGS="-Db_sanitize=address,undefined -Db_lundef=false"
    fi
    UNAME=$(uname -s)
    if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* ]]; then
        # The .pc file doesn't work on Windows.
        TBB_CONFIG_OPT=--cmake-prefix-path='{{justfile_directory()}}/dependencies/oneTBB-{{BUILD_TYPE}}/lib/cmake'
    else
        # But CMake doesn't work on Linux, at least sometimes.
        TBB_CONFIG_OPT=--pkg-config-path='{{justfile_directory()}}/dependencies/oneTBB-{{BUILD_TYPE}}/lib/pkgconfig'
    fi
    meson setup --reconfigure builddir \
        --buildtype=$BUILD_TYPE $SANITIZE_FLAGS \
        $TBB_CONFIG_OPT \
        -Dcatch2:tests=false -Dgoogle-benchmark:tests=disabled \
        {{FLAGS}}

_configure_if_not_configured:
    if [ ! -d builddir ] || [ ! -d dependencies/oneTBB-release ]; then \
        just configure release; fi

build: _configure_if_not_configured
    meson compile -C builddir

clean:
    if [ -d builddir ]; then meson compile --clean -C builddir; fi

wipe:
    if [ -d builddir ]; then meson setup --wipe builddir; fi

test: _configure_if_not_configured
    meson test -C builddir

[positional-arguments]
benchmark *FLAGS: build test
    builddir/ihist_bench --benchmark_time_unit=us \
        --benchmark_counters_tabular=true \
        "$@"

benchmark-set-baseline: build test
    cp builddir/ihist_bench{{exe_suffix}} \
        builddir/ihist_bench_baseline{{exe_suffix}}

[positional-arguments]
_benchmark-compare *ARGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    GB_VERSION=$(meson introspect --dependencies builddir |jq -r \
        '.[] | select(.meson_variables[]? == "benchmark_dep") | .version')
    GB_TOOLS=subprojects/benchmark-$GB_VERSION/tools
    uv run --with=scipy "$GB_TOOLS/compare.py" "$@" \
        --benchmark_time_unit=us --benchmark_counters_tabular=true

[positional-arguments]
benchmark-compare *FLAGS: build test
    just _benchmark-compare benchmarks \
        builddir/ihist_bench_baseline{{exe_suffix}} \
        builddir/ihist_bench{{exe_suffix}} "$@"

[positional-arguments]
benchmark-compare-filters FILTER1 FILTER2 *FLAGS: build test
    just _benchmark-compare filters builddir/ihist_bench{{exe_suffix}} "$@"

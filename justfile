help:
    @just --list

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
    #!/usr/bin/env sh
    set -e
    case {{BUILD_TYPE}} in
        "release") CMAKE_BUILD_TYPE=Release;;
        "debugoptimized") CMAKE_BUILD_TYPE=RelWithDebInfo;;
        "debug") CMAKE_BUILD_TYPE=Debug;;
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
      -DBUILD_SHARED_LIBS=OFF \
      -DTBB_TEST=OFF \
      -DTBBMALLOC_BUILD=OFF \
      -DTBBMALLOC_PROXY_BUILD=OFF
    cmake --build build --target install >/dev/null

clean-tbb:
    rm -rf dependencies/oneTBB-*

configure BUILD_TYPE *FLAGS:
    if [ ! -d dependencies/oneTBB-{{BUILD_TYPE}} ]; then \
        just build-tbb {{BUILD_TYPE}}; fi
    meson setup --reconfigure builddir \
        --buildtype={{BUILD_TYPE}} \
        --pkg-config-path=dependencies/oneTBB-Release/lib/pkgconfig \
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

benchmark *FLAGS: build test
    builddir/ihist_bench --benchmark_time_unit=us \
        --benchmark_counters_tabular=true \
        {{FLAGS}}

benchmark-only REGEXP:
    just benchmark --benchmark_filter={{quote(REGEXP)}}
# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

# Show usage
help:
    @just --list
    @echo
    @echo 'Required tools: uv, pkg-config, C and C++ toolchain'
    @echo 'Required for Java bindings: mvn (>= 3.6.3, < 4)'
    @echo 'On Windows, Git Bash is required.'

exe_suffix := if os() == "windows" { ".exe" } else { "" }
mvn := if os() == "windows" { "mvn.cmd" } else { "mvn" }
cjdk_exec := 'uvx cjdk -j zulu:8 exec --'

onetbb_version := '2022.3.0'

# On Windows, put DLLs on path so that tests and benchmarks can run.
export PATH := if os() == "windows" {
    env("PWD") + '/dependencies/oneapi-tbb-' + onetbb_version +
    '/redist/intel64/vc14;' + env("PWD") + '/builddir;' + env("PATH")
} else {
    env("PATH")
}

# On Linux and macOS, oneTBB and (optionally) OpenCV should be installed by
# the system package manager or Homebrew (at least for development). On
# Windows, we automatically download oneTBB binaries; OpenCV can (optionally)
# be installed via scoop.
_download-windows-deps:
    #!/usr/bin/env bash
    set -euxo pipefail
    mkdir -p dependencies
    cd dependencies
    TBB_ZIP=oneapi-tbb-{{onetbb_version}}-win.zip
    if [ ! -f $TBB_ZIP ]; then
        curl -LO https://github.com/uxlfoundation/oneTBB/releases/download/v{{onetbb_version}}/$TBB_ZIP
        unzip -q $TBB_ZIP
    fi

# Configure for development build
configure BUILD_TYPE *FLAGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    BUILD_TYPE={{BUILD_TYPE}}
    SANITIZE_FLAGS="-Db_sanitize=none -Db_lundef=true"
    if [ "$BUILD_TYPE" = "sanitize" ]; then
        BUILD_TYPE=debug
        SANITIZE_FLAGS="-Db_sanitize=address,undefined -Db_lundef=false"
    elif [ "$BUILD_TYPE" = "sanitizeoptimized" ]; then
        BUILD_TYPE=debugoptimized
        SANITIZE_FLAGS="-Db_sanitize=address,undefined -Db_lundef=false"
    fi
    UNAME=$(uname -s)
    if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* ]]; then
        just _download-windows-deps
        DEPS_PATH_OPT="--cmake-prefix-path=$USERPROFILE/scoop/apps/opencv/current --pkg-config-path=$PWD/dependencies/oneapi-tbb-{{onetbb_version}}/lib/pkgconfig"
    else
        DEPS_PATH_OPT=
    fi
    uvx meson setup --reconfigure builddir --vsenv \
        ${IHIST_MESON_CROSS_ARGS-} \
        --buildtype=$BUILD_TYPE $SANITIZE_FLAGS \
        $DEPS_PATH_OPT \
        -Dcatch2:tests=false -Dgoogle-benchmark:tests=disabled \
        -Dgoogle-benchmark:werror=false \
        {{FLAGS}}

_configure_if_not_configured:
    #!/usr/bin/env bash
    set -euxo pipefail
    if [ ! -d builddir ]; then
        just configure release
    fi

# Run full build
build: _configure_if_not_configured
    uvx meson compile -C builddir

# Remove build products
clean:
    if [ -d builddir ]; then uvx meson compile --clean -C builddir; fi
    rm -f coverage/cpp.*
    rmdir coverage/ 2>/dev/null || true

# Wipe build directory and reconfigure using previous options
wipe:
    if [ -d builddir ]; then uvx meson setup --wipe builddir; fi

# Run the unit tests
test: _configure_if_not_configured
    uvx meson test -C builddir

# Run the unit tests with coverage (coverage/cpp.html)
coverage:
    uvx meson setup --wipe builddir-coverage \
        --buildtype=debugoptimized \
        -Dbenchmarks=disabled \
        -Dcatch2:tests=false \
        -Db_coverage=true
    uvx meson test -C builddir-coverage
    mkdir -p coverage
    uvx gcovr builddir-coverage/ --html-details coverage/cpp.html \
        -e subprojects/ -e tests/

# Run the 'ihist_test' program directly
[positional-arguments]
ihist_test *FLAGS: build
    builddir/tests/ihist_test "$@"

# Run the 'ihist_bench' program directly
[positional-arguments]
ihist_bench *FLAGS: build
    builddir/benchmarks/ihist_bench "$@"

# Run the 'api_bench' program directly
[positional-arguments]
api_bench *FLAGS: build
    builddir/benchmarks/api_bench "$@"

# Run benchmarks (specifying a filter is recommended)
[positional-arguments]
benchmark *FLAGS: build test
    builddir/benchmarks/api_bench --benchmark_time_unit=ms \
        --benchmark_counters_tabular=true \
        "$@"

# Keep a copy of the benchmark executable for later comparison
benchmark-set-baseline: build test
    cp builddir/benchmarks/api_bench{{exe_suffix}} \
        builddir/benchmarks/api_bench_baseline{{exe_suffix}}

[positional-arguments]
_benchmark-compare *ARGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    GB_VERSION=$(uvx meson introspect --projectinfo builddir |jq -r \
        '.subprojects[] | select(.name == "google-benchmark") | .version')
    GB_TOOLS=subprojects/benchmark-$GB_VERSION/tools
    uv run --no-project --with=scipy "$GB_TOOLS/compare.py" "$@" \
        --benchmark_time_unit=ms --benchmark_counters_tabular=true

# Compare benchmarks with baseline
[positional-arguments]
benchmark-compare *FLAGS: build test
    just _benchmark-compare benchmarks \
        builddir/benchmarks/api_bench_baseline{{exe_suffix}} \
        builddir/benchmarks/api_bench{{exe_suffix}} "$@"

# Compare two different set of benchmarks in the current build
[positional-arguments]
benchmark-compare-filters FILTER1 FILTER2 *FLAGS: build test
    just _benchmark-compare filters \
        builddir/benchmarks/api_bench{{exe_suffix}} "$@"

# Run a banchmarking or tuning script (in scripts/)
[positional-arguments]
run SCRIPT *FLAGS: build
    uv run --no-project "$@"

# We do NOT use editable installs for Python because they don't work with
# meson-python + uv (https://github.com/astral-sh/uv/issues/10214,
# https://github.com/mesonbuild/meson-python/issues/730). (They would work if
# we used pip, but we use uv also for C++ and Java tasks and it's a bit messy
# to be using both.)

# We need to disable uv's cache so that builds with different options are
# correctly (re)installed even when there are no changes to files.

# Build and install Python bindings
py-install:
    uv venv --allow-existing
    uv pip install --no-cache --reinstall --group dev . \
        -C setup-args=-Db_ndebug=false

# Build and install Python bindings for coverage
py-cov-install:
    uv venv --allow-existing
    uv pip install --no-cache --reinstall --group dev . \
        -C build-dir=build-coverage \
        -C setup-args=-Db_coverage=true \
        -C setup-args=-Dbuildtype=debugoptimized

# Run Python tests
py-test: py-install
    uv run --no-sync pytest

# Run Python tests with coverage (coverage/python.html)
py-coverage: py-cov-install
    find build-coverage/ -name '*.gcda' -exec rm -f {} +
    uv run --no-sync pytest
    mkdir -p coverage
    uvx gcovr build-coverage/ -f python/src/ihist/_ihist.cpp \
        --html-details coverage/python.html

# Build Python wheel (for local testing)
py-build:
    uv build

# Clean Python build artifacts
py-clean:
    rm -rf build/ dist/ wheelhouse/ *.egg-info \
        build-coverage/ coverage/python.* .venv/ .pytest_cache/
    rmdir coverage/ 2>/dev/null || true
    find python -type d -name __pycache__ -exec rm -rf {} +
    rm -f uv.lock  # For now, we don't pin dependencies

# Run cibuildwheel locally for native architecture
cibuildwheel:
    #!/usr/bin/env bash
    set -euxo pipefail
    export TBB_PREFIX="$PWD/dependencies/oneTBB-Release"
    export PKG_CONFIG_PATH="$TBB_PREFIX/lib/pkgconfig"
    UNAME=$(uname -s)
    if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* ]]; then
        export CXX=clang-cl
    fi
    uv run --no-project --with=cmake bash scripts/build_static_tbb.sh
    CIBW_ARCHS=native uvx cibuildwheel

# Print the project version in Maven format
java-version:
    @uvx meson introspect --projectinfo meson.build | jq -r '.version' | \
        sed 's/\.dev.*/-SNAPSHOT/'

# Build Java native library
java-build-jni:
    #!/usr/bin/env bash
    set -euxo pipefail
    export TBB_PREFIX="$PWD/dependencies/oneTBB-Release"
    export PKG_CONFIG_PATH="$TBB_PREFIX/lib/pkgconfig"
    UNAME=$(uname -s)
    if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* ]]; then
        export CXX=clang-cl
    fi
    uv run --no-project --with=cmake bash scripts/build_static_tbb.sh
    {{cjdk_exec}} uvx meson setup --reconfigure builddir-jni \
        ${IHIST_MESON_CROSS_ARGS-} \
        --buildtype=release --default-library=static -Djava-bindings=enabled \
        -Dtests=disabled -Dbenchmarks=disabled
    {{cjdk_exec}} uvx meson compile -C builddir-jni

# Build Java bindings
java-build: java-build-jni
    #!/usr/bin/env bash
    set -euxo pipefail
    VERSION=$(just java-version)
    {{cjdk_exec}} {{mvn}} -f java/pom.xml compile -Drevision="$VERSION"

# Test Java bindings (with Java coverage)
java-test: java-build-jni
    #!/usr/bin/env bash
    set -euxo pipefail
    VERSION=$(just java-version)
    {{cjdk_exec}} {{mvn}} -f java/pom.xml verify -Drevision="$VERSION" \
        -Dnative.library.path=../builddir-jni/java

# Package Java bindings without JNI libs
java-package-no-jni:
    #!/usr/bin/env bash
    set -euxo pipefail
    VERSION=$(just java-version)
    {{cjdk_exec}} {{mvn}} -f java/pom.xml package -Drevision="$VERSION" \
        -Dskip.natives=true -DskipTests=true

# Stage Java bindings
java-stage natives_jars_dir: java-package-no-jni
    #!/usr/bin/env bash
    set -euxo pipefail
    VERSION=$(just java-version)
    STAGEDIR=java/target/staging-deploy/io/github/marktsuchida/ihist/$VERSION
    mkdir -p $STAGEDIR
    cp {{natives_jars_dir}}/ihist-$VERSION-natives-*.jar $STAGEDIR/
    cp java/.flattened-pom.xml $STAGEDIR/ihist-$VERSION.pom
    cp java/target/ihist-$VERSION.jar $STAGEDIR/
    cp java/target/ihist-$VERSION-*.jar $STAGEDIR/

# Test Java bindings with C++ coverage
java-coverage:
    #!/usr/bin/env bash
    set -euxo pipefail
    {{cjdk_exec}} uvx meson setup --reconfigure builddir-jni-cov \
        ${IHIST_MESON_CROSS_ARGS-} \
        -Djava-bindings=enabled -Db_coverage=true --buildtype=debugoptimized \
        -Dtests=disabled -Dbenchmarks=disabled
    {{cjdk_exec}} uvx meson compile -C builddir-jni-cov
    find builddir-jni-cov/ -name '*.gcda' -exec rm -f {} +
    VERSION=$(just java-version)
    {{cjdk_exec}} {{mvn}} -f java/pom.xml package -Drevision="$VERSION" \
        -Dnative.library.path=../builddir-jni-cov/java
    mkdir -p coverage
    uvx gcovr builddir-jni-cov/ -f java/src/main/cpp/ihistj_jni.cpp \
        --html-details coverage/java.html

# Clean Java build artifacts
java-clean:
    {{cjdk_exec}} {{mvn}} -f java/pom.xml clean || true
    if [ -d builddir-jni ]; then \
        {{cjdk_exec}} uvx meson compile --clean -C builddir-jni; \
    fi
    if [ -d builddir-jni-cov ]; then \
        {{cjdk_exec}} uvx meson compile --clean -C builddir-jni-cov; \
    fi
    rm -rf coverage/java.*
    rmdir coverage/ 2>/dev/null || true

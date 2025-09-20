# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

help:
    @just --list

exe_suffix := if os_family() == "windows" { ".exe" } else { "" }

configure BUILD_TYPE *FLAGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    BUILD_TYPE={{BUILD_TYPE}}
    SANITIZE_FLAGS="-Db_sanitize=none -Db_lundef=true"
    if [ "$BUILD_TYPE" = "sanitize" ]; then
        BUILD_TYPE=debug
        SANITIZE_FLAGS="-Db_sanitize=address,undefined -Db_lundef=false"
    fi
    UNAME=$(uname -s)
    if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* ]]; then
        DEPS_PATH_OPT=--cmake-prefix-path="$USERPROFILE/scoop/apps/opencv/current"
    else
        DEPS_PATH_OPT=
    fi
    meson setup --reconfigure builddir \
        --buildtype=$BUILD_TYPE $SANITIZE_FLAGS \
        $DEPS_PATH_OPT \
        -Dcatch2:tests=false -Dgoogle-benchmark:tests=disabled \
        {{FLAGS}}

_configure_if_not_configured:
    #!/usr/bin/env bash
    set -euxo pipefail
    if [ ! -d builddir ]; then
        just configure release
    fi

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
    builddir/benchmarks/api_bench --benchmark_time_unit=ms \
        --benchmark_counters_tabular=true \
        "$@"

benchmark-set-baseline: build test
    cp builddir/benchmarks/api_bench{{exe_suffix}} \
        builddir/benchmarks/api_bench_baseline{{exe_suffix}}

[positional-arguments]
_benchmark-compare *ARGS:
    #!/usr/bin/env bash
    set -euxo pipefail
    GB_VERSION=$(meson introspect --dependencies builddir |jq -r \
        '.[] | select(.meson_variables[]? == "benchmark_dep") | .version')
    GB_TOOLS=subprojects/benchmark-$GB_VERSION/tools
    uv run --with=scipy "$GB_TOOLS/compare.py" "$@" \
        --benchmark_time_unit=ms --benchmark_counters_tabular=true

[positional-arguments]
benchmark-compare *FLAGS: build test
    just _benchmark-compare benchmarks \
        builddir/benchmarks/api_bench_baseline{{exe_suffix}} \
        builddir/benchmarks/api_bench{{exe_suffix}} "$@"

[positional-arguments]
benchmark-compare-filters FILTER1 FILTER2 *FLAGS: build test
    just _benchmark-compare filters \
        builddir/benchmarks/api_bench{{exe_suffix}} "$@"

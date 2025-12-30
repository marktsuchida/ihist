#!/usr/bin/env bash
# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

#
# Build oneTBB as static library
#

# Warning: This builds oneTBB in a minimal configuration and without any of its
# optional dependencies. It may not expose all functions. Because TBBBind
# (which requires hwloc as a dependency) is ommitted, tbb::info functions may
# return incorrect information (in particular, the physical core count may not
# be correct).

# Note that oneTBB's CMake scripts will warn about building a static library.
# However, the warning only applies if there is a chance that symbols from
# multiple copies of oneTBB may be accessible within a program:
# https://github.com/uxlfoundation/oneTBB/issues/646#issuecomment-966106176

set -euxo pipefail

tbb_version='2022.3.0'
tbb_tgz="oneapi-tbb-$tbb_version.tar.gz"
tbb_dir="oneTBB-$tbb_version"

cmake_build_type=Release

# Use TBB_PREFIX if set, otherwise use a default
if [[ -z "${TBB_PREFIX:-}" ]]; then
    # Default for local testing
    install_prefix="$(cd "$(dirname "$0")/.." && pwd)/dependencies/oneTBB-$cmake_build_type"
else
    install_prefix="$TBB_PREFIX"
fi

# Use Ninja if available, otherwise fall back to default generator
if command -v ninja &> /dev/null; then
    cmake_generator=(-G Ninja)
else
    cmake_generator=()
fi

# On macOS, extract architecture from ARCHFLAGS (set by cibuildwheel)
if [[ "$OSTYPE" == darwin* ]] && [[ -n "${ARCHFLAGS:-}" ]]; then
    osx_arch="${ARCHFLAGS#*-arch }"
    osx_arch="${osx_arch%% *}"
    cmake_osx_arch=(-DCMAKE_OSX_ARCHITECTURES="$osx_arch")
else
    cmake_osx_arch=()
fi

cd $(dirname "$0")/..
mkdir -p dependencies
cd dependencies
if [ ! -f "$tbb_tgz" ]; then
    curl -Lo "$tbb_tgz" \
        https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v$tbb_version.tar.gz
fi
tar xf "$tbb_tgz"
cd "$tbb_dir"
mkdir -p build
# Set CMAKE_INSTALL_LIBDIR in case default is lib64.
cmake "${cmake_generator[@]}" "${cmake_osx_arch[@]}" -S . -B build \
    -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE="$cmake_build_type" \
    -DCMAKE_INSTALL_MESSAGE=LAZY \
    -DBUILD_SHARED_LIBS=OFF \
    -DTBB_TEST=OFF \
    -DTBBMALLOC_BUILD=OFF \
    -DTBBMALLOC_PROXY_BUILD=OFF
cmake --build build --target install

set +x
echo
echo "Install prefix: $install_prefix"
echo "To test builds locally, set PKG_CONFIG_PATH to:"
echo "$install_prefix/lib/pkgconfig"

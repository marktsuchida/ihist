#!/usr/bin/env bash
# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

# This script is designed to run inside a manylinux_2_28 container.

set -euxo pipefail

yum install -y java-1.8.0-openjdk-devel

export PATH="/opt/python/cp314-cp314/bin:$PATH"
pip install meson ninja

export TBB_PREFIX=/tmp/tbb
export PKG_CONFIG_PATH=/tmp/tbb/lib/pkgconfig
bash scripts/build_static_tbb.sh

meson setup builddir-jni \
    --buildtype=release --default-library=static \
    -Djava-bindings=enabled -Dtests=disabled -Dbenchmarks=disabled
meson compile -C builddir-jni

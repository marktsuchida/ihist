# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

project(
    'ihist',
    'cpp',
    version: '0.1',
    meson_version: '>= 1.3.0',
    default_options: [
        'warning_level=3',
        'cpp_std=c++17',
        'werror=true',
    ],
)

# We do not build oneTBB as a wrap/subproject, because when using as a C++
# library, oneTBB should be dynamically linked (if other parts of the
# application also use it). For quickly building for development, use the
# justfile.
onetbb_dep = dependency(
    'tbb',
    allow_fallback: false,
    required: true,
    include_type: 'system',
)

catch2_with_main_dep = dependency(
    'catch2-with-main',
    allow_fallback: true,
    include_type: 'system',
    static: true,
)

benchmark_dep = dependency(
    'benchmark',
    allow_fallback: true,
    include_type: 'system',
    static: true,
)

dependencies = [
    onetbb_dep,
]

shlib_args = ['-DIHIST_BUILDING_SHARED']

lib = library(
    'ihist',
    'ihist.cpp',
    install: true,
    cpp_shared_args: shlib_args,
    gnu_symbol_visibility: 'hidden',
    dependencies: dependencies,
)

test_exe = executable(
    'ihist_test',
    'ihist_test.cpp',
    dependencies: [
        dependencies,
        catch2_with_main_dep,
    ],
    link_with: lib,
)
test('ihist', test_exe)

ihist_dep = declare_dependency(
    include_directories: include_directories('.'),
    dependencies: dependencies,
    link_with: lib,
)

install_headers('ihist.hpp', subdir: 'ihist')

subdir('benchmarks')

# Make this library usable as a Meson subproject.
meson.override_dependency('ihist', ihist_dep)

pkg_mod = import('pkgconfig')
pkg_mod.generate(
    lib,
    description: 'Meson sample project.',
    subdirs: 'ihist',
)
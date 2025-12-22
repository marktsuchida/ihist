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

cxx = meson.get_compiler('cpp')
if host_machine.system() == 'darwin' and \
    cxx.get_id() == 'clang' and cxx.version().startswith('16.')
    # Work around https://github.com/mesonbuild/meson/issues/14856
    # Remove when fixed (expected in Meson 1.8.4).
    # (For Apple Clang 17 it was fixed in Meson 1.8.3, but it was still broken
    # for Apple Clang 16.)
    add_global_arguments('-U_LIBCPP_ENABLE_ASSERTIONS', language: 'cpp')
    add_global_arguments('-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST', language: 'cpp')
endif

# We do not build oneTBB as a wrap/subproject, because when using as a C++
# library, oneTBB should be dynamically linked (if other parts of the
# application also use it). See justfile for quickly obtaining.
onetbb_dep = dependency(
    'tbb',
    allow_fallback: false,
    required: get_option('tbb'),
    include_type: 'system',
)

tbb_cpp_args = []
if onetbb_dep.found()
    tbb_cpp_args += '-DIHIST_USE_TBB=1'
endif

onetbb_dep = declare_dependency(
    dependencies: onetbb_dep,
    compile_args: tbb_cpp_args,
)

if not get_option('tests').disabled()
    catch2_with_main_dep = dependency(
        'catch2-with-main',
        allow_fallback: true,
        include_type: 'system',
        static: true,
    )
endif

if not get_option('benchmarks').disabled()
    benchmark_dep = dependency(
        'benchmark',
        allow_fallback: true,
        include_type: 'system',
        static: true,
    )

    # E.g., Homebrew has opencv4 (pkg-config), Scoop has OpenCV (cmake).
    opencv_dep = dependency('opencv4', required: false)
    if not opencv_dep.found()
        opencv_dep = dependency('OpenCV', required: false)
    endif

    opencv_cpp_args = []
    if opencv_dep.found()
        opencv_cpp_args += '-DIHIST_HAVE_OPENCV=1'
        if cxx.get_id() == 'clang'
            opencv_cpp_args += '-Wno-c11-extensions'
        endif
        if cxx.get_id() in ['clang-cl', 'msvc']
            opencv_cpp_args += '-D_CRT_SECURE_NO_WARNINGS'
        endif
    endif

    opencv_dep = declare_dependency(
        dependencies: opencv_dep,
        compile_args: opencv_cpp_args,
    )
endif

dependencies = []
if onetbb_dep.found()
    dependencies += onetbb_dep
endif

subdir('include')
subdir('src')

ihist_lib = library(
    'ihist',
    ihist_srcs,
    include_directories: ihist_public_inc,
    install: true,
    cpp_shared_args: [
        '-DIHIST_BUILDING_SHARED',
    ],
    gnu_symbol_visibility: 'hidden',
    dependencies: dependencies,
)

lib_user_args = []
if get_option('default_library') == 'shared'
    lib_user_args += '-DIHIST_SHARED'
elif get_option('default_library') == 'both'
    # Ideally we would define IHIST_SHARED, but only if linking to shared. But
    # there's no clean way to do so, and the shared library works without the
    # definition.
endif

ihist_dep = declare_dependency(
    include_directories: ihist_public_inc,
    compile_args: lib_user_args,
    dependencies: dependencies,
    link_with: ihist_lib,
)

if not get_option('tests').disabled()
    subdir('tests')
endif
if not get_option('benchmarks').disabled()
    subdir('benchmarks')
endif

if get_option('python-bindings').enabled()
    subdir('python')
endif

if get_option('java-bindings').enabled()
    subdir('java')
endif

# Make this library usable as a Meson subproject.
meson.override_dependency('ihist', ihist_dep)

pkg_mod = import('pkgconfig')
pkg_requires = []
if onetbb_dep.found()
    pkg_requires = ['tbb']
endif
pkg_mod.generate(
    ihist_lib,
    description: 'Fast image histograms',
    subdirs: 'ihist',
    requires: pkg_requires,
)

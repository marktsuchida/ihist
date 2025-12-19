# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

py = import('python').find_installation(pure: false)

# The nanobind WrapDB package uses declare_dependency(sources: ...), which
# causes nanobind's sources to be compiled as part of our extension_module
# target, inheriting our project's werror=true. To avoid third-party warnings
# breaking the build, we compile nanobind as a separate static library with
# werror=false.
nanobind_orig_dep = dependency('nanobind', include_type: 'system')
nanobind_lib = static_library(
    '_nanobind',
    dependencies: nanobind_orig_dep,
    override_options: ['werror=false'],
)
nanobind_dep = declare_dependency(
    link_with: nanobind_lib,
    dependencies: [
        # Only include compile_args and includes; link_args and links are
        # already propagated transitively through nanobind_lib.
        nanobind_orig_dep.partial_dependency(
            compile_args: true,
            includes: true,
        ).as_system(),
    ],
)

# Get numpy include path directly from Python for cross-compilation support
# (dependency('numpy') doesn't work for that).
numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
).stdout().strip()
numpy_dep = declare_dependency(
    include_directories: include_directories(numpy_inc),
)

py_package_name = 'ihist'

py.extension_module(
    '_ihist_bindings',
    'src/ihist/_ihist_bindings.cpp',
    dependencies: [
        ihist_dep,
        nanobind_dep,
        numpy_dep,
    ],
    install: true,
    subdir: py_package_name,
    install_tag: 'python-runtime',
)

py.install_sources(
    'src/ihist/__init__.py',
    subdir: py_package_name,
    install_tag: 'python-runtime',
)

# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

py = import('python').find_installation(pure: false)

nanobind_dep = dependency('nanobind', include_type: 'system')
numpy_dep = dependency('numpy', include_type: 'system')

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
)

py.install_sources(
    'src/ihist/__init__.py',
    subdir: py_package_name,
)

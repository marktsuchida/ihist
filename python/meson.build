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
# (dependency('numpy') doesn't work for that). Use compiler-specific system
# include flags to avoid Meson's rejection of absolute paths inside the
# source tree (e.g., venv).
numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
).stdout().strip()
cxx = meson.get_compiler('cpp')
if cxx.get_argument_syntax() == 'msvc'
    numpy_args = ['/external:I', numpy_inc]
else
    numpy_args = ['-isystem', numpy_inc]
endif
numpy_dep = declare_dependency(compile_args: numpy_args)

py_package_name = 'ihist'

# Hide symbols from statically linked dependencies (e.g., TBB). TBB uses
# visibility("default") on its API, which overrides -fvisibility=hidden.
symbol_hide_link_args = []
symbol_hide_link_depends = []

if host_machine.system() == 'linux'
    symbol_hide_link_args += ['-Wl,--exclude-libs,ALL']
elif host_machine.system() == 'darwin'
    symbols_file = meson.current_source_dir() / 'symbols-macos.txt'
    symbol_hide_link_args += ['-Wl,-exported_symbols_list,' + symbols_file]
    symbol_hide_link_depends += [files('symbols-macos.txt')]
endif

ext_module = py.extension_module(
    '_ihist',
    'src/ihist/_ihist.cpp',
    dependencies: [
        ihist_dep,
        nanobind_dep,
        numpy_dep,
    ],
    link_args: symbol_hide_link_args,
    link_depends: symbol_hide_link_depends,
    install: true,
    subdir: py_package_name,
    install_tag: 'python-runtime',
)

# Generate .pyi type stubs using nanobind.stubgen
stub_target = custom_target(
    'generate_stubs',
    input: [ext_module],
    output: '_ihist.pyi',
    command: [
        py.full_path(),
        meson.project_source_root() / 'scripts' / 'build_stubs.py',
        '@INPUT@',
        '@OUTPUT@',
    ],
    build_by_default: true,
    install: true,
    install_dir: py.get_install_dir() / py_package_name,
    install_tag: 'python-runtime',
    depends: ext_module,
)

# Copy stub to source tree for development (so type checkers can find it)
custom_target(
    'copy_stubs_to_src',
    input: stub_target,
    output: 'stub_copied_marker',
    command: [
        py.full_path(),
        '-c',
        'import shutil; shutil.copy(r"@INPUT@", r"@0@")'.format(
            meson.current_source_dir() / 'src' / 'ihist' / '_ihist.pyi'
        ),
    ],
    build_by_default: true,
    depends: stub_target,
)

py.install_sources(
    'src/ihist/__init__.py',
    'src/ihist/py.typed',
    subdir: py_package_name,
    install_tag: 'python-runtime',
)

# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

java_mod = import('java')
jni_dep = dependency('jni', version: '>= 1.8.0', modules: ['jvm'])

jni_header = java_mod.native_headers(
    'src/main/java/io/github/marktsuchida/ihist/IHistNative.java',
    'src/main/java/io/github/marktsuchida/ihist/NativeLibraryLoader.java',
    package: 'io.github.marktsuchida.ihist',
    classes: ['IHistNative'],
)

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

shared_library(
    'ihistj',
    'src/main/cpp/ihistj_jni.cpp',
    jni_header,
    dependencies: [
        ihist_dep,
        jni_dep,
    ],
    link_args: symbol_hide_link_args,
    link_depends: symbol_hide_link_depends,
    gnu_symbol_visibility: 'inlineshidden',
    install: false,
)

# Java build is continued by Maven.

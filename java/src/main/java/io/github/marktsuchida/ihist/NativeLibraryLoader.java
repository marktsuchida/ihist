// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package io.github.marktsuchida.ihist;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.security.SecureRandom;
import java.util.Locale;

/**
 * Handles extraction and loading of native library from JAR resources.
 */
final class NativeLibraryLoader {

    private static final String LIBRARY_NAME = "ihistj";
    private static final boolean DEBUG = Boolean.getBoolean("ihist.debug");

    private NativeLibraryLoader() {}

    static void load() {
        // Try extraction and fallback to java.library.path. But if extraction
        // succeeds yet load fails, we do _not_ catch the UnsatisfiedLinkError.
        try {
            loadPackaged();
        } catch (Exception e) {
            debug("Native library extraction failed: " + e);
            debug("Falling back to java.library.path");
            System.loadLibrary(LIBRARY_NAME);
        }
    }

    private static void loadPackaged() throws Exception {
        String os = detectOs();
        String arch = detectArch();
        if (os == null || arch == null) {
            throw new UnsupportedOperationException(
                "Unsupported platform: " + System.getProperty("os.name") +
                "/" + System.getProperty("os.arch"));
        }

        // We use a unique filename for the extracted library, so that the
        // library can be loaded more than once from multiple class loaders.
        String extractedLibName =
            System.mapLibraryName(LIBRARY_NAME + "-" + randomHex(12));

        String resourceLibName = System.mapLibraryName(LIBRARY_NAME);
        String resourcePath =
            "/natives/" + os + "/" + arch + "/" + resourceLibName;
        debug("Looking for resource: " + resourcePath);

        try (InputStream in =
                 NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Native library resource not found: " +
                                      resourcePath);
            }

            if ("windows".equals(os)) {
                loadPackagedWindows(in, extractedLibName);
            } else {
                loadPackagedUnix(in, extractedLibName);
            }
        }
    }

    private static String detectOs() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        if (os.contains("linux")) {
            return "linux";
        }
        if (os.contains("mac")) {
            return "macos";
        }
        if (os.contains("win")) {
            return "windows";
        }
        return null;
    }

    private static String detectArch() {
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            return "x86_64";
        }
        if (arch.equals("aarch64")) {
            return "arm64";
        }
        if (arch.equals("arm") || arch.startsWith("armv7")) {
            return "arm32";
        }
        return null;
    }

    private static String randomHex(int length) {
        byte[] bytes = new byte[(length + 1) / 2];
        new SecureRandom().nextBytes(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.substring(0, length);
    }

    private static void loadPackagedUnix(InputStream in, String libName)
        throws IOException {
        // On Unix-like systems, the temporary directory might be shared and
        // world-writable. We use a private temporary directory just while we
        // load the library, and remove the extracted library as soon as we're
        // done. (Usually the temporary directory is on a filesystem with POSIX
        // semantics, so we can delete an open file.)

        File nativesDir = createUnixTempDirectory();
        File libFile = new File(nativesDir, libName);
        boolean keepForDebug = false;

        try {
            debug("Extracting to: " + libFile.getAbsolutePath());
            extractLibrary(in, libFile);

            debug("Loading native library");
            System.load(libFile.getAbsolutePath());
        } catch (UnsatisfiedLinkError e) {
            keepForDebug = DEBUG;
            throw e;
        } finally {
            if (keepForDebug) {
                debug("Load failed; keeping extracted library for debugging");
            } else {
                if (libFile.delete()) {
                    debug("Deleted extracted library");
                }
                if (nativesDir.delete()) {
                    debug("Deleted temp directory");
                }
            }
        }
    }

    private static void loadPackagedWindows(InputStream in, String libName)
        throws IOException {
        // On Windows, we cannot remove the extracted library from the current
        // process (not even with deleteOnExit()), because it won't be unloaded
        // once loaded. So we extract to a known directory and defer cleanup to
        // the next time we load. This should be safe because the temporary
        // directory is per-user on Windows.

        File nativesDir = getWindowsNativesDirectory();
        File libFile = new File(nativesDir, libName);

        // A lock directory is used to prevent races with cleanup
        File lockDir = makeLockPath(libFile);
        if (!lockDir.mkdir()) {
            throw new IOException("Cannot create native library lock: " +
                                  lockDir.getName());
        }

        try {
            debug("Extracting to: " + libFile.getAbsolutePath());
            extractLibrary(in, libFile);

            debug("Loading native library");
            try {
                System.load(libFile.getAbsolutePath());
            } catch (UnsatisfiedLinkError e) {
                if (DEBUG) {
                    debug("Load failed; keeping extracted library for "
                          + "debugging");
                } else {
                    libFile.delete();
                }
                throw e;
            }
        } finally {
            lockDir.delete();
        }

        cleanupWindows(libFile, nativesDir);
    }

    private static File createUnixTempDirectory() throws IOException {
        String tmpdir = System.getProperty("ihist.tmpdir");
        if (tmpdir != null) {
            return Files
                .createTempDirectory(new File(tmpdir).toPath(), "ihist-")
                .toFile();
        }
        return Files.createTempDirectory("ihist-").toFile();
    }

    private static File getWindowsNativesDirectory() throws IOException {
        String tmpdir = System.getProperty("ihist.tmpdir");
        if (tmpdir == null) {
            tmpdir = System.getProperty("java.io.tmpdir");
        }
        File nativesDir = new File(tmpdir, "ihist-natives");
        if (!nativesDir.isDirectory() && !nativesDir.mkdirs()) {
            throw new IOException("Failed to create natives directory: " +
                                  nativesDir.getAbsolutePath());
        }
        return nativesDir;
    }

    private static File makeLockPath(File libFile) {
        String name = libFile.getName();
        int dot = name.lastIndexOf('.');
        String stem = (dot > 0 ? name.substring(0, dot) : name);
        return new File(libFile.getParentFile(), stem + ".lock");
    }

    private static void extractLibrary(InputStream in, File libFile)
        throws IOException {
        try (FileOutputStream out = new FileOutputStream(libFile)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }

    private static void cleanupWindows(File libFile, File nativesDir) {
        // If a previously-extracted DLL is still in use (by another JVM or
        // classloader), deletion will silently fail. We skip files with a
        // corresponding lock directory to avoid a race with concurrent
        // extraction.

        File[] files = nativesDir.listFiles();
        if (files == null) {
            return;
        }
        long now = System.currentTimeMillis();
        long lockAgeThreshold = 10 * 60 * 1000;
        for (File file : files) {
            if (file.equals(libFile)) {
                continue;
            }
            // It is safe to delete old locks, because locks are only held
            // while extraction and loading of the library.
            if (file.isDirectory() && file.getName().endsWith(".lock")) {
                if (now - file.lastModified() > lockAgeThreshold) {
                    if (file.delete()) {
                        debug("Deleted old lock: " + file.getName());
                    }
                }
                continue;
            }
            if (file.isFile()) {
                File lockDir = makeLockPath(file);
                if (lockDir.isDirectory()) {
                    continue;
                }
                if (file.delete()) {
                    debug("Deleted old library: " + file.getName());
                }
            }
        }
    }

    private static void debug(String msg) {
        if (DEBUG) {
            System.err.println("[ihist] " + msg);
        }
    }
}

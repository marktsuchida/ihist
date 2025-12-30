// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for native library loading from JARs.
 *
 * <p>These tests run after packaging (via maven-failsafe-plugin) to verify
 * that the native library can be loaded from the natives JAR.
 */
class NativeLibraryLoadingIT {

    @Test
    void loadFromDefaultClassLoader() {
        assertDoesNotThrow(IHistNative::loadNativeLibrary);

        // Smoke test with single-pixel image
        ByteBuffer image = ByteBuffer.allocateDirect(1);
        image.put((byte)42).flip();
        IntBuffer histogram = IntBuffer.allocate(256);
        IHistNative.histogram8(8, image, null, 1, 1, 1, 1, 1, new int[] {0},
                               histogram, false);
        assertEquals(1, histogram.get(42));
    }

    @Test
    void loadFromIsolatedClassLoader() throws Exception {
        URL[] jarUrls = findAllJars();
        try (URLClassLoader loader = new URLClassLoader(jarUrls, null)) {
            Class<?> cls = Class.forName("ihistj.IHistNative", true, loader);
            Method loadMethod = cls.getMethod("loadNativeLibrary");
            assertDoesNotThrow(() -> loadMethod.invoke(null));

            // Smoke test with single-pixel image
            ByteBuffer image = ByteBuffer.allocateDirect(1);
            image.put((byte)42).flip();
            IntBuffer histogram = IntBuffer.allocate(256);
            Method hist8 = cls.getMethod(
                "histogram8", int.class, ByteBuffer.class, ByteBuffer.class,
                int.class, int.class, int.class, int.class, int.class,
                int[].class, IntBuffer.class, boolean.class);
            hist8.invoke(null, 8, image, null, 1, 1, 1, 1, 1, new int[] {0},
                         histogram, false);
            assertEquals(1, histogram.get(42));
        }
    }

    private static URL[] findAllJars() throws Exception {
        URL codeSource = IHistNative.class.getProtectionDomain()
                             .getCodeSource()
                             .getLocation();
        File jarFile = new File(codeSource.toURI());
        File dir = jarFile.getParentFile();
        File[] jars = dir.listFiles(
            (d, name) -> name.startsWith("ihistj-") && name.endsWith(".jar"));
        if (jars == null || jars.length == 0) {
            throw new IllegalStateException("No ihistj JARs found in " + dir);
        }
        URL[] urls = new URL[jars.length];
        for (int i = 0; i < jars.length; i++) {
            urls[i] = jars[i].toURI().toURL();
        }
        return urls;
    }
}

// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

/**
 * Low-level JNI wrapper for ihist native library.
 *
 * <p>This class provides direct access to the ihist C API functions.
 * For most use cases, prefer the high-level {@link Histogram} API.
 *
 * <p><b>Important:</b> The histogram is <b>accumulated</b> into the output
 * buffer. Existing values are added to, not replaced. To obtain a fresh
 * histogram, zero-initialize the buffer before calling these methods.
 *
 * <p><b>Note on signed types:</b> Java's {@code byte} type is signed (-128 to
 * 127), but image pixels are typically unsigned (0 to 255). The native code
 * correctly interprets the bit patterns as unsigned values. Similarly for
 * {@code short} with 16-bit images.
 */
public final class IHistNative {

    private static final String NATIVE_LIBRARY_NAME = "ihistj";
    private static volatile boolean loaded = false;

    static { loadNativeLibrary(); }

    private IHistNative() {
        // Prevent instantiation
    }

    /**
     * Load the native library.
     *
     * <p>Called automatically on class load. Users can call this explicitly
     * to check for library availability or force early loading.
     *
     * @throws UnsatisfiedLinkError if the native library cannot be loaded
     */
    public static synchronized void loadNativeLibrary() {
        if (!loaded) {
            System.loadLibrary(NATIVE_LIBRARY_NAME);
            loaded = true;
        }
    }

    // ========== Array-based methods ==========

    /**
     * Compute histogram for 8-bit image data (array input).
     *
     * @param sampleBits      number of significant bits per sample (1-8);
     *                        determines histogram size (2^sampleBits bins per
     *                        component)
     * @param image           image pixel data (row-major, interleaved
     *                        components)
     * @param imageOffset     byte offset into image array where data starts
     * @param mask            per-pixel mask (null to histogram all pixels);
     *                        non-zero mask values include the pixel
     * @param maskOffset      byte offset into mask array where data starts
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel (e.g.,
     *                        3 for RGB)
     * @param componentIndices indices of components to histogram (each must be
     *                         &lt; nComponents)
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated
     * @param histogramOffset offset into histogram array where output starts
     * @param parallel        if true, allows multi-threaded execution for
     *                        large images
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static native void
    histogram8(int sampleBits, byte[] image, int imageOffset, byte[] mask,
               int maskOffset, int height, int width, int imageStride,
               int maskStride, int nComponents, int[] componentIndices,
               int[] histogram, int histogramOffset, boolean parallel);

    /**
     * Compute histogram for 16-bit image data (array input).
     *
     * @param sampleBits      number of significant bits per sample (1-16);
     *                        determines histogram size (2^sampleBits bins per
     *                        component)
     * @param image           image pixel data (row-major, interleaved
     *                        components)
     * @param imageOffset     element offset into image array where data starts
     * @param mask            per-pixel mask (null to histogram all pixels);
     *                        non-zero mask values include the pixel
     * @param maskOffset      byte offset into mask array where data starts
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel (e.g.,
     *                        3 for RGB)
     * @param componentIndices indices of components to histogram (each must be
     *                         &lt; nComponents)
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated
     * @param histogramOffset offset into histogram array where output starts
     * @param parallel        if true, allows multi-threaded execution for
     *                        large images
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static native void
    histogram16(int sampleBits, short[] image, int imageOffset, byte[] mask,
                int maskOffset, int height, int width, int imageStride,
                int maskStride, int nComponents, int[] componentIndices,
                int[] histogram, int histogramOffset, boolean parallel);

    // ========== Buffer-based methods ==========

    /**
     * Compute histogram for 8-bit image data (buffer input).
     *
     * <p>For direct buffers, this provides zero-copy access to native memory.
     * For heap buffers backed by arrays, the backing array is accessed
     * directly.
     *
     * <p>Buffer positions are used as the starting point for data access.
     *
     * @param sampleBits      number of significant bits per sample (1-8)
     * @param image           image pixel data buffer (position marks start of
     *                        data)
     * @param mask            per-pixel mask buffer (null to histogram all
     *                        pixels)
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel
     * @param componentIndices indices of components to histogram
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated
     * @param parallel        if true, allows multi-threaded execution
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static native void
    histogram8(int sampleBits, ByteBuffer image, ByteBuffer mask, int height,
               int width, int imageStride, int maskStride, int nComponents,
               int[] componentIndices, IntBuffer histogram, boolean parallel);

    /**
     * Compute histogram for 16-bit image data (buffer input).
     *
     * <p>For direct buffers, this provides zero-copy access to native memory.
     * For heap buffers backed by arrays, the backing array is accessed
     * directly.
     *
     * <p>Buffer positions are used as the starting point for data access.
     *
     * @param sampleBits      number of significant bits per sample (1-16)
     * @param image           image pixel data buffer (position marks start of
     *                        data)
     * @param mask            per-pixel mask buffer (null to histogram all
     *                        pixels)
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel
     * @param componentIndices indices of components to histogram
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated
     * @param parallel        if true, allows multi-threaded execution
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static native void
    histogram16(int sampleBits, ShortBuffer image, ByteBuffer mask, int height,
                int width, int imageStride, int maskStride, int nComponents,
                int[] componentIndices, IntBuffer histogram, boolean parallel);
}

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
 * For most use cases, prefer the high-level {@link HistogramRequest} API.
 *
 * <p><b>Important:</b> The histogram is <b>accumulated</b> into the output
 * buffer. Existing values are added to, not replaced. To obtain a fresh
 * histogram, zero-initialize the buffer before calling these methods.
 *
 * <p><b>Note on signed types:</b> Java's {@code byte} type is signed (-128 to
 * 127), but image pixels are typically unsigned (0 to 255). The native code
 * correctly interprets the bit patterns as unsigned values. Similarly for
 * {@code short} with 16-bit images.
 *
 * <p><b>Buffer requirements:</b> These methods accept both direct buffers and
 * array-backed buffers (such as those created by {@code ByteBuffer.wrap()}).
 * View buffers and other buffer types that are neither direct nor array-backed
 * are not supported and will throw {@code IllegalArgumentException}. Use
 * {@link HistogramRequest} if you need to work with such buffers.
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

    /**
     * Compute histogram for 8-bit image data.
     *
     * <p>This method supports both direct buffers (for zero-copy access) and
     * array-backed buffers (created via {@code ByteBuffer.wrap()}). For
     * array-backed buffers, the backing array is accessed directly using
     * JNI critical array access, also providing zero-copy performance.
     *
     * <p>View buffers and read-only heap buffers are not supported. Use
     * {@link HistogramRequest} for automatic handling of such buffers.
     *
     * <p>Buffer positions are used as the starting point for data access.
     *
     * @param sampleBits      number of significant bits per sample (0-8);
     *                        determines histogram size (2^sampleBits bins per
     *                        component)
     * @param image           image pixel data buffer (position marks start of
     *                        data); must be direct or array-backed
     * @param mask            per-pixel mask buffer (null to histogram all
     *                        pixels); must be direct or array-backed if
     *                        provided
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel
     * @param componentIndices indices of components to histogram (each must be
     *                         &lt; nComponents)
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated; must be direct or array-backed and
     *                        not read-only
     * @param parallel        if true, allows multi-threaded execution
     * @throws NullPointerException     if image, componentIndices, or
     *                                  histogram is null
     * @throws IllegalArgumentException if parameters are invalid, buffers are
     *                                  neither direct nor array-backed, or
     *                                  histogram buffer is read-only
     */
    public static native void
    histogram8(int sampleBits, ByteBuffer image, ByteBuffer mask, int height,
               int width, int imageStride, int maskStride, int nComponents,
               int[] componentIndices, IntBuffer histogram, boolean parallel);

    /**
     * Compute histogram for 16-bit image data.
     *
     * <p>This method supports both direct buffers (for zero-copy access) and
     * array-backed buffers (created via {@code ShortBuffer.wrap()}). For
     * array-backed buffers, the backing array is accessed directly using
     * JNI critical array access, also providing zero-copy performance.
     *
     * <p>View buffers and read-only heap buffers are not supported. Use
     * {@link HistogramRequest} for automatic handling of such buffers.
     *
     * <p>Buffer positions are used as the starting point for data access.
     *
     * @param sampleBits      number of significant bits per sample (0-16);
     *                        determines histogram size (2^sampleBits bins per
     *                        component)
     * @param image           image pixel data buffer (position marks start of
     *                        data); must be direct or array-backed
     * @param mask            per-pixel mask buffer (null to histogram all
     *                        pixels); must be direct or array-backed if
     *                        provided
     * @param height          image height in pixels
     * @param width           image width in pixels
     * @param imageStride     row stride in pixels (must be &gt;= width)
     * @param maskStride      mask row stride in pixels (must be &gt;= width)
     * @param nComponents     number of interleaved components per pixel
     * @param componentIndices indices of components to histogram (each must be
     *                         &lt; nComponents)
     * @param histogram       output buffer for histogram data; values are
     *                        accumulated; must be direct or array-backed and
     *                        not read-only
     * @param parallel        if true, allows multi-threaded execution
     * @throws NullPointerException     if image, componentIndices, or
     *                                  histogram is null
     * @throws IllegalArgumentException if parameters are invalid, buffers are
     *                                  neither direct nor array-backed, or
     *                                  histogram buffer is read-only
     */
    public static native void
    histogram16(int sampleBits, ShortBuffer image, ByteBuffer mask, int height,
                int width, int imageStride, int maskStride, int nComponents,
                int[] componentIndices, IntBuffer histogram, boolean parallel);
}

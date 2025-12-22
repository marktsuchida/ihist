// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

/**
 * Simple high-level API for common histogram operations.
 *
 * <p>This class provides static convenience methods for computing histograms
 * in common scenarios. For advanced use cases (ROI, masking, custom strides),
 * use {@link HistogramRequest} builder API instead.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Grayscale 8-bit image
 * int[] hist = Histogram.compute(grayData, width, height);
 *
 * // RGB 8-bit image
 * int[] hist = Histogram.compute(rgbData, width, height, 3);
 *
 * // 12-bit grayscale image (stored in short[])
 * int[] hist = Histogram.compute(data, width, height, 12);
 * }</pre>
 */
public final class Histogram {

    private Histogram() {
        // Prevent instantiation
    }

    /**
     * Compute histogram of 8-bit grayscale image.
     *
     * @param image  pixel data (length must be width * height)
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return histogram with 256 bins
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(byte[] image, int width, int height) {
        return HistogramRequest.forImage8(image, width, height).compute();
    }

    /**
     * Compute histogram of 8-bit grayscale image from buffer.
     *
     * @param image  pixel data buffer
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return histogram with 256 bins
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(ByteBuffer image, int width, int height) {
        return HistogramRequest.forImage8(image, width, height).compute();
    }

    /**
     * Compute histogram of 8-bit multi-component image.
     *
     * @param image       interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of components per pixel (e.g., 3 for RGB)
     * @return histogram with nComponents * 256 values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(byte[] image, int width, int height,
                                int nComponents) {
        return HistogramRequest.forImage8(image, width, height)
            .components(nComponents)
            .compute();
    }

    /**
     * Compute histogram of 8-bit multi-component image from buffer.
     *
     * @param image       interleaved pixel data buffer
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of components per pixel (e.g., 3 for RGB)
     * @return histogram with nComponents * 256 values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(ByteBuffer image, int width, int height,
                                int nComponents) {
        return HistogramRequest.forImage8(image, width, height)
            .components(nComponents)
            .compute();
    }

    /**
     * Compute histogram of 16-bit grayscale image.
     *
     * @param image pixel data
     * @param width image width in pixels
     * @param height image height in pixels
     * @param bits  significant bits per sample (1-16)
     * @return histogram with 2^bits bins
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(short[] image, int width, int height,
                                int bits) {
        return HistogramRequest.forImage16(image, width, height)
            .bits(bits)
            .compute();
    }

    /**
     * Compute histogram of 16-bit grayscale image from buffer.
     *
     * @param image pixel data buffer
     * @param width image width in pixels
     * @param height image height in pixels
     * @param bits  significant bits per sample (1-16)
     * @return histogram with 2^bits bins
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(ShortBuffer image, int width, int height,
                                int bits) {
        return HistogramRequest.forImage16(image, width, height)
            .bits(bits)
            .compute();
    }

    /**
     * Compute histogram of 16-bit multi-component image.
     *
     * @param image       interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of components per pixel
     * @param bits        significant bits per sample (1-16)
     * @return histogram with nComponents * 2^bits values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(short[] image, int width, int height,
                                int nComponents, int bits) {
        return HistogramRequest.forImage16(image, width, height)
            .components(nComponents)
            .bits(bits)
            .compute();
    }

    /**
     * Compute histogram of 16-bit multi-component image from buffer.
     *
     * @param image       interleaved pixel data buffer
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of components per pixel
     * @param bits        significant bits per sample (1-16)
     * @return histogram with nComponents * 2^bits values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(ShortBuffer image, int width, int height,
                                int nComponents, int bits) {
        return HistogramRequest.forImage16(image, width, height)
            .components(nComponents)
            .bits(bits)
            .compute();
    }

    /**
     * Compute histogram of 8-bit image, selecting specific components.
     *
     * <p>Example: For RGBA data, use {@code selectComponents(0, 1, 2)} to
     * histogram only RGB and skip alpha.
     *
     * @param image            interleaved pixel data
     * @param width            image width in pixels
     * @param height           image height in pixels
     * @param nComponents      number of components per pixel
     * @param componentIndices indices of components to histogram
     * @return histogram with componentIndices.length * 256 values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(byte[] image, int width, int height,
                                int nComponents, int... componentIndices) {
        return HistogramRequest.forImage8(image, width, height)
            .components(nComponents)
            .selectComponents(componentIndices)
            .compute();
    }

    /**
     * Compute histogram of 16-bit image, selecting specific components.
     *
     * @param image            interleaved pixel data
     * @param width            image width in pixels
     * @param height           image height in pixels
     * @param nComponents      number of components per pixel
     * @param bits             significant bits per sample (1-16)
     * @param componentIndices indices of components to histogram
     * @return histogram with componentIndices.length * 2^bits values
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static int[] compute(short[] image, int width, int height,
                                int nComponents, int bits,
                                int... componentIndices) {
        return HistogramRequest.forImage16(image, width, height)
            .components(nComponents)
            .bits(bits)
            .selectComponents(componentIndices)
            .compute();
    }
}

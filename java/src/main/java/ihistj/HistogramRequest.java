// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

/**
 * Builder for configuring and executing histogram computation.
 *
 * <p>This class provides a fluent API for specifying image data,
 * regions of interest (ROI), masking, and output options.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Simple grayscale histogram
 * int[] histogram = HistogramRequest.forImage8(imageData, width, height)
 *     .compute();
 *
 * // RGB histogram with ROI and mask
 * int[] histogram = HistogramRequest.forImage8(imageData, width, height, 3)
 *     .roi(10, 10, 100, 100)
 *     .mask(maskData)
 *     .bits(8)
 *     .parallel(true)
 *     .compute();
 * }</pre>
 */
public final class HistogramRequest {

    // Image data (one of these is set)
    private byte[] image8Array;
    private short[] image16Array;
    private ByteBuffer image8Buffer;
    private ShortBuffer image16Buffer;

    // Image dimensions (full image)
    private int imageWidth;
    private int imageHeight;
    private int imageStride; // defaults to imageWidth

    // ROI (region of interest)
    private int roiX = 0;
    private int roiY = 0;
    private int roiWidth = -1;  // -1 means use full width
    private int roiHeight = -1; // -1 means use full height

    // Mask data (optional)
    private byte[] maskArray;
    private ByteBuffer maskBuffer;
    private int maskStride = -1; // -1 means same as effective ROI width
    private int maskOffsetX = 0; // offset within mask for ROI
    private int maskOffsetY = 0;

    // Component configuration
    private int nComponents = 1;
    private int[] componentIndices; // null means all components

    // Histogram parameters
    private int sampleBits = -1; // -1 means use default (8 or 16)
    private int[] outputHistogram;
    private IntBuffer outputBuffer;
    private boolean accumulate = false;
    private boolean parallel = true;

    private final boolean is8Bit;

    private HistogramRequest(boolean is8Bit) { this.is8Bit = is8Bit; }

    // ========== Factory methods ==========

    /**
     * Create a request for 8-bit grayscale image data from a byte array.
     *
     * @param image  row-major pixel data
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage8(byte[] image, int width,
                                             int height) {
        return forImage8(image, width, height, 1);
    }

    /**
     * Create a request for 8-bit image data from a byte array.
     *
     * @param image       row-major interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of interleaved components per pixel
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage8(byte[] image, int width,
                                             int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException(
                "dimensions must be non-negative");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        HistogramRequest req = new HistogramRequest(true);
        req.image8Array = image;
        req.imageWidth = width;
        req.imageHeight = height;
        req.imageStride = width;
        req.nComponents = nComponents;
        return req;
    }

    /**
     * Create a request for 8-bit grayscale image data from a ByteBuffer.
     *
     * @param image  row-major pixel data
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage8(ByteBuffer image, int width,
                                             int height) {
        return forImage8(image, width, height, 1);
    }

    /**
     * Create a request for 8-bit image data from a ByteBuffer.
     *
     * @param image       row-major interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of interleaved components per pixel
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage8(ByteBuffer image, int width,
                                             int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException(
                "dimensions must be non-negative");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        HistogramRequest req = new HistogramRequest(true);
        req.image8Buffer = image;
        req.imageWidth = width;
        req.imageHeight = height;
        req.imageStride = width;
        req.nComponents = nComponents;
        return req;
    }

    /**
     * Create a request for 16-bit grayscale image data from a short array.
     *
     * @param image  row-major pixel data
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage16(short[] image, int width,
                                              int height) {
        return forImage16(image, width, height, 1);
    }

    /**
     * Create a request for 16-bit image data from a short array.
     *
     * @param image       row-major interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of interleaved components per pixel
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage16(short[] image, int width,
                                              int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException(
                "dimensions must be non-negative");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        HistogramRequest req = new HistogramRequest(false);
        req.image16Array = image;
        req.imageWidth = width;
        req.imageHeight = height;
        req.imageStride = width;
        req.nComponents = nComponents;
        return req;
    }

    /**
     * Create a request for 16-bit grayscale image data from a ShortBuffer.
     *
     * @param image  row-major pixel data
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage16(ShortBuffer image, int width,
                                              int height) {
        return forImage16(image, width, height, 1);
    }

    /**
     * Create a request for 16-bit image data from a ShortBuffer.
     *
     * @param image       row-major interleaved pixel data
     * @param width       image width in pixels
     * @param height      image height in pixels
     * @param nComponents number of interleaved components per pixel
     * @return new HistogramRequest builder
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static HistogramRequest forImage16(ShortBuffer image, int width,
                                              int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException(
                "dimensions must be non-negative");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be >= 1");
        }
        HistogramRequest req = new HistogramRequest(false);
        req.image16Buffer = image;
        req.imageWidth = width;
        req.imageHeight = height;
        req.imageStride = width;
        req.nComponents = nComponents;
        return req;
    }

    // ========== Builder methods ==========

    /**
     * Set the image row stride (for non-contiguous image data).
     *
     * @param stride row stride in pixels (must be &gt;= width)
     * @return this builder
     */
    public HistogramRequest stride(int stride) {
        this.imageStride = stride;
        return this;
    }

    /**
     * Set which component indices to histogram.
     *
     * <p>Example: For RGBA data, use {@code selectComponents(0, 1, 2)} to
     * histogram only RGB and skip alpha.
     *
     * @param indices component indices to histogram (each must be &lt;
     *     nComponents)
     * @return this builder
     */
    public HistogramRequest selectComponents(int... indices) {
        this.componentIndices = indices.clone();
        return this;
    }

    /**
     * Set a region of interest (ROI) within the image.
     *
     * @param x      ROI left edge in pixels
     * @param y      ROI top edge in pixels
     * @param width  ROI width in pixels
     * @param height ROI height in pixels
     * @return this builder
     */
    public HistogramRequest roi(int x, int y, int width, int height) {
        this.roiX = x;
        this.roiY = y;
        this.roiWidth = width;
        this.roiHeight = height;
        return this;
    }

    /**
     * Set a per-pixel mask (array input).
     *
     * <p>Pixels with mask value 0 are excluded from the histogram.
     * Pixels with non-zero mask values are included.
     *
     * @param mask mask data
     * @return this builder
     */
    public HistogramRequest mask(byte[] mask) {
        this.maskArray = mask;
        this.maskBuffer = null;
        return this;
    }

    /**
     * Set a per-pixel mask (buffer input).
     *
     * <p>Pixels with mask value 0 are excluded from the histogram.
     * Pixels with non-zero mask values are included.
     *
     * @param mask mask data
     * @return this builder
     */
    public HistogramRequest mask(ByteBuffer mask) {
        this.maskBuffer = mask;
        this.maskArray = null;
        return this;
    }

    /**
     * Set the mask stride and offset for ROI alignment.
     *
     * <p>This allows the mask to have a different layout than the image ROI.
     * The mask offset specifies where in the mask the ROI data begins.
     *
     * @param stride  mask row stride in pixels
     * @param offsetX mask X offset for ROI alignment
     * @param offsetY mask Y offset for ROI alignment
     * @return this builder
     */
    public HistogramRequest maskLayout(int stride, int offsetX, int offsetY) {
        this.maskStride = stride;
        this.maskOffsetX = offsetX;
        this.maskOffsetY = offsetY;
        return this;
    }

    /**
     * Set the number of significant bits per sample.
     *
     * <p>This determines the histogram size (2^bits bins per component).
     * Values outside the significant bit range are truncated.
     *
     * @param bits 1-8 for 8-bit images, 1-16 for 16-bit images
     * @return this builder
     */
    public HistogramRequest bits(int bits) {
        this.sampleBits = bits;
        return this;
    }

    /**
     * Set a pre-allocated output histogram array.
     *
     * <p>If not specified, a new array will be allocated.
     *
     * @param histogram output array (size must be &gt;= nHistComponents *
     *     2^bits)
     * @return this builder
     */
    public HistogramRequest output(int[] histogram) {
        this.outputHistogram = histogram;
        this.outputBuffer = null;
        return this;
    }

    /**
     * Set a pre-allocated output histogram buffer.
     *
     * <p>If not specified, a new array will be allocated.
     *
     * @param histogram output buffer (remaining capacity must be sufficient)
     * @return this builder
     */
    public HistogramRequest output(IntBuffer histogram) {
        this.outputBuffer = histogram;
        this.outputHistogram = null;
        return this;
    }

    /**
     * Set whether to accumulate into existing output values.
     *
     * <p>If false (default), the output buffer is zeroed before computing.
     * If true, histogram counts are added to existing values.
     *
     * @param accumulate true to add to existing values
     * @return this builder
     */
    public HistogramRequest accumulate(boolean accumulate) {
        this.accumulate = accumulate;
        return this;
    }

    /**
     * Set whether to allow parallel execution.
     *
     * <p>If true (default), multiple threads may be used for large images.
     * If false, single-threaded execution is guaranteed.
     *
     * @param parallel true to allow multi-threading
     * @return this builder
     */
    public HistogramRequest parallel(boolean parallel) {
        this.parallel = parallel;
        return this;
    }

    // ========== Execution ==========

    /**
     * Compute the histogram.
     *
     * @return histogram data (allocated if output was not specified)
     * @throws IllegalArgumentException if parameters are invalid
     */
    public int[] compute() {
        validate();

        int effectiveWidth = (roiWidth < 0) ? imageWidth : roiWidth;
        int effectiveHeight = (roiHeight < 0) ? imageHeight : roiHeight;
        int effectiveMaskStride =
            (maskStride < 0) ? effectiveWidth : maskStride;
        int effectiveBits = (sampleBits < 0) ? (is8Bit ? 8 : 16) : sampleBits;

        int[] indices = (componentIndices != null)
                            ? componentIndices
                            : defaultComponentIndices(nComponents);
        int nHistComponents = indices.length;

        int histSize = nHistComponents * (1 << effectiveBits);
        int[] histogram;
        if (outputHistogram != null) {
            histogram = outputHistogram;
            if (!accumulate) {
                Arrays.fill(histogram, 0, histSize, 0);
            }
        } else if (outputBuffer != null) {
            if (outputBuffer.hasArray()) {
                histogram = outputBuffer.array();
                int offset =
                    outputBuffer.arrayOffset() + outputBuffer.position();
                if (!accumulate) {
                    Arrays.fill(histogram, offset, offset + histSize, 0);
                }
            } else {
                throw new IllegalArgumentException(
                    "output IntBuffer must be array-backed when using "
                    + "compute()");
            }
        } else {
            histogram = new int[histSize];
        }

        int imageOffset = (roiY * imageStride + roiX) * nComponents;
        int maskOffset = maskOffsetY * effectiveMaskStride + maskOffsetX;

        if (is8Bit) {
            if (image8Array != null) {
                IHistNative.histogram8(
                    effectiveBits, image8Array, imageOffset, maskArray,
                    maskOffset, effectiveHeight, effectiveWidth, imageStride,
                    effectiveMaskStride, nComponents, indices, histogram,
                    outputBuffer != null
                        ? outputBuffer.arrayOffset() + outputBuffer.position()
                        : 0,
                    parallel);
            } else {
                ByteBuffer sliced = image8Buffer.duplicate();
                sliced.position(sliced.position() + imageOffset);

                ByteBuffer maskSliced = null;
                if (maskBuffer != null) {
                    maskSliced = maskBuffer.duplicate();
                    maskSliced.position(maskSliced.position() + maskOffset);
                }

                IntBuffer histBuf = outputBuffer != null
                                        ? outputBuffer
                                        : IntBuffer.wrap(histogram);

                IHistNative.histogram8(
                    effectiveBits, sliced, maskSliced, effectiveHeight,
                    effectiveWidth, imageStride, effectiveMaskStride,
                    nComponents, indices, histBuf, parallel);
            }
        } else {
            if (image16Array != null) {
                IHistNative.histogram16(
                    effectiveBits, image16Array, imageOffset, maskArray,
                    maskOffset, effectiveHeight, effectiveWidth, imageStride,
                    effectiveMaskStride, nComponents, indices, histogram,
                    outputBuffer != null
                        ? outputBuffer.arrayOffset() + outputBuffer.position()
                        : 0,
                    parallel);
            } else {
                ShortBuffer sliced = image16Buffer.duplicate();
                sliced.position(sliced.position() + imageOffset);

                ByteBuffer maskSliced = null;
                if (maskBuffer != null) {
                    maskSliced = maskBuffer.duplicate();
                    maskSliced.position(maskSliced.position() + maskOffset);
                }

                IntBuffer histBuf = outputBuffer != null
                                        ? outputBuffer
                                        : IntBuffer.wrap(histogram);

                IHistNative.histogram16(
                    effectiveBits, sliced, maskSliced, effectiveHeight,
                    effectiveWidth, imageStride, effectiveMaskStride,
                    nComponents, indices, histBuf, parallel);
            }
        }

        return histogram;
    }

    private void validate() {
        if (imageWidth < 0 || imageHeight < 0) {
            throw new IllegalArgumentException(
                "Image dimensions must be non-negative");
        }
        if (imageStride < imageWidth) {
            throw new IllegalArgumentException(
                "Image stride must be >= width");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException(
                "Number of components must be >= 1");
        }

        int effectiveWidth = (roiWidth < 0) ? imageWidth : roiWidth;
        int effectiveHeight = (roiHeight < 0) ? imageHeight : roiHeight;

        if (roiX < 0 || roiY < 0) {
            throw new IllegalArgumentException(
                "ROI offset cannot be negative");
        }
        if (roiX + effectiveWidth > imageWidth ||
            roiY + effectiveHeight > imageHeight) {
            throw new IllegalArgumentException("ROI exceeds image bounds");
        }

        int effectiveBits = (sampleBits < 0) ? (is8Bit ? 8 : 16) : sampleBits;
        int maxBits = is8Bit ? 8 : 16;
        if (effectiveBits < 1 || effectiveBits > maxBits) {
            throw new IllegalArgumentException(
                "sampleBits must be in range [1, " + maxBits + "]");
        }

        if (componentIndices != null) {
            for (int idx : componentIndices) {
                if (idx < 0 || idx >= nComponents) {
                    throw new IllegalArgumentException(
                        "Component index out of range [0, " + nComponents +
                        ")");
                }
            }
        }
    }

    private static int[] defaultComponentIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        return indices;
    }
}

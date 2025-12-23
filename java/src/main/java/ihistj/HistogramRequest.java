// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

/**
 * Builder for configuring and executing histogram computation.
 *
 * <p>
 * This class provides a fluent API for specifying image data,
 * regions of interest (ROI), masking, and output options.
 *
 * <p>
 * Example usage:
 *
 * <pre>{@code
 * // Simple grayscale histogram
 * IntBuffer histogram = HistogramRequest.forImage(imageData, width, height)
 *         .compute();
 * // Access counts: histogram.get(i) for bin i
 *
 * // RGB histogram with ROI and mask
 * IntBuffer histogram = HistogramRequest.forImage(imageData, width, height, 3)
 *         .roi(10, 10, 100, 100)
 *         .mask(maskData, 100, 100)
 *         .bits(8)
 *         .parallel(true)
 *         .compute();
 * }</pre>
 *
 * <p><b>Buffer type support:</b> This class accepts arrays, direct buffers,
 * array-backed buffers, and view buffers. Arrays and array-backed buffers are
 * handled with zero-copy via JNI. Direct buffers are also zero-copy. View
 * buffers and other non-standard buffer types are automatically copied to
 * temporary direct buffers (this incurs a copy overhead but ensures all buffer
 * types work).
 */
public final class HistogramRequest {

    // Image data (one of these is set)
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
    private ByteBuffer maskBuffer;
    private int maskWidth = -1; // -1 means no mask
    private int maskHeight = -1;
    private int maskOffsetX = 0; // offset within mask for ROI
    private int maskOffsetY = 0;

    // Component configuration
    private int nComponents = 1;
    private int[] componentIndices; // null means all components

    // Histogram parameters
    private int sampleBits = -1; // -1 means use default (8 or 16)
    private IntBuffer outputBuffer;
    private boolean accumulate = false;
    private boolean parallel = true;

    private final boolean is8Bit;

    private HistogramRequest(boolean is8Bit) { this.is8Bit = is8Bit; }

    private static void validateImageParams(Buffer image, int width,
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
    }

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
    public static HistogramRequest forImage(byte[] image, int width,
                                            int height) {
        return forImage(image, width, height, 1);
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
    public static HistogramRequest forImage(byte[] image, int width,
                                            int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        return forImage(ByteBuffer.wrap(image), width, height, nComponents);
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
    public static HistogramRequest forImage(ByteBuffer image, int width,
                                            int height) {
        return forImage(image, width, height, 1);
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
    public static HistogramRequest forImage(ByteBuffer image, int width,
                                            int height, int nComponents) {
        validateImageParams(image, width, height, nComponents);
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
    public static HistogramRequest forImage(short[] image, int width,
                                            int height) {
        return forImage(image, width, height, 1);
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
    public static HistogramRequest forImage(short[] image, int width,
                                            int height, int nComponents) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        return forImage(ShortBuffer.wrap(image), width, height, nComponents);
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
    public static HistogramRequest forImage(ShortBuffer image, int width,
                                            int height) {
        return forImage(image, width, height, 1);
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
    public static HistogramRequest forImage(ShortBuffer image, int width,
                                            int height, int nComponents) {
        validateImageParams(image, width, height, nComponents);
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
     * Set which component indices to histogram.
     *
     * <p>
     * Example: For RGBA data, use {@code selectComponents(0, 1, 2)} to
     * histogram only RGB and skip alpha.
     *
     * @param indices component indices to histogram (each must be &lt;
     *                nComponents)
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
     * <p>
     * Pixels with mask value 0 are excluded from the histogram.
     * Pixels with non-zero mask values are included.
     *
     * @param mask   mask data in row-major order
     * @param width  mask width in pixels
     * @param height mask height in pixels
     * @return this builder
     */
    public HistogramRequest mask(byte[] mask, int width, int height) {
        if (mask == null) {
            throw new IllegalArgumentException("mask cannot be null");
        }
        return mask(ByteBuffer.wrap(mask), width, height);
    }

    /**
     * Set a per-pixel mask (buffer input).
     *
     * <p>
     * Pixels with mask value 0 are excluded from the histogram.
     * Pixels with non-zero mask values are included.
     *
     * @param mask   mask data in row-major order
     * @param width  mask width in pixels
     * @param height mask height in pixels
     * @return this builder
     */
    public HistogramRequest mask(ByteBuffer mask, int width, int height) {
        this.maskBuffer = mask;
        this.maskWidth = width;
        this.maskHeight = height;
        return this;
    }

    /**
     * Set the mask offset for ROI alignment.
     *
     * <p>
     * The mask offset specifies where in the mask the ROI data begins.
     * The ROI pixels will be read from
     * mask[(offsetY + y) * maskWidth + (offsetX + x)].
     *
     * @param x mask X offset for ROI alignment
     * @param y mask Y offset for ROI alignment
     * @return this builder
     */
    public HistogramRequest maskOffset(int x, int y) {
        this.maskOffsetX = x;
        this.maskOffsetY = y;
        return this;
    }

    /**
     * Set the number of significant bits per sample.
     *
     * <p>
     * This determines the histogram size (2^bits bins per component).
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
     * <p>
     * If not specified, a new array will be allocated.
     *
     * @param histogram output array (size must be &gt;= nHistComponents *
     *                  2^bits)
     * @return this builder
     */
    public HistogramRequest output(int[] histogram) {
        if (histogram == null) {
            throw new IllegalArgumentException("histogram cannot be null");
        }
        return output(IntBuffer.wrap(histogram));
    }

    /**
     * Set a pre-allocated output histogram buffer.
     *
     * <p>
     * If not specified, a new array will be allocated.
     *
     * @param histogram output buffer (remaining capacity must be sufficient);
     *                  must not be read-only
     * @return this builder
     */
    public HistogramRequest output(IntBuffer histogram) {
        this.outputBuffer = histogram;
        return this;
    }

    /**
     * Set whether to accumulate into existing output values.
     *
     * <p>
     * If false (default), the output buffer is zeroed before computing.
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
     * <p>
     * If true (default), multiple threads may be used for large images.
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
     * <p>
     * The returned buffer's position and limit are set to cover exactly the
     * histogram data: position at the start, limit at position + histogram
     * size. This means {@code remaining()} equals the histogram size.
     *
     * <p>
     * If an output buffer was provided via {@link #output(IntBuffer)}, the
     * returned buffer is a duplicate that shares the same backing storage.
     * The original buffer's position and limit are not modified.
     *
     * @return histogram buffer (allocated if output was not specified)
     * @throws IllegalArgumentException if parameters are invalid
     */
    public IntBuffer compute() {
        validate();

        int effectiveWidth = (roiWidth < 0) ? imageWidth : roiWidth;
        int effectiveHeight = (roiHeight < 0) ? imageHeight : roiHeight;
        int effectiveMaskStride = (maskWidth > 0) ? maskWidth : effectiveWidth;
        int effectiveBits = (sampleBits < 0) ? (is8Bit ? 8 : 16) : sampleBits;

        int[] indices = (componentIndices != null)
                            ? componentIndices
                            : defaultComponentIndices(nComponents);
        int nHistComponents = indices.length;

        int histSize = nHistComponents * (1 << effectiveBits);

        // Create working buffer: duplicate of user's buffer or newly allocated
        IntBuffer workBuffer;
        int originalPos;
        if (outputBuffer != null) {
            // Validate capacity before duplicating
            if (outputBuffer.remaining() < histSize) {
                throw new IllegalArgumentException(
                    "output IntBuffer has insufficient capacity: " +
                    outputBuffer.remaining() + " < " + histSize);
            }
            // Use duplicate so we don't modify user's buffer position/limit
            workBuffer = outputBuffer.duplicate();
            originalPos = workBuffer.position();
        } else {
            workBuffer = IntBuffer.allocate(histSize);
            originalPos = 0;
        }

        // Clear if not accumulating
        if (!accumulate) {
            clearBuffer(workBuffer, histSize);
        }

        // Prepare histogram buffer for JNI call
        IntBuffer histBuf;
        IntBuffer tempHistBuf = null;
        boolean isViewBuffer =
            !workBuffer.isDirect() && !workBuffer.hasArray();

        if (isViewBuffer) {
            // View buffer: need temp direct buffer for JNI
            tempHistBuf = allocateDirectIntBuffer(histSize);
            if (accumulate) {
                // Copy existing values to temp buffer for accumulation
                IntBuffer dup = workBuffer.duplicate();
                for (int i = 0; i < histSize; i++) {
                    tempHistBuf.put(dup.get());
                }
                tempHistBuf.flip();
            }
            histBuf = tempHistBuf;
        } else {
            histBuf = workBuffer;
        }

        int imageOffset = (roiY * imageStride + roiX) * nComponents;
        int maskOffset = maskOffsetY * effectiveMaskStride + maskOffsetX;

        // Prepare mask buffer
        ByteBuffer maskBuf = prepareMaskBuffer(maskOffset);

        if (is8Bit) {
            ByteBuffer imageBuf = prepareImage8Buffer(imageOffset);
            IHistNative.histogram8(effectiveBits, imageBuf, maskBuf,
                                   effectiveHeight, effectiveWidth,
                                   imageStride, effectiveMaskStride,
                                   nComponents, indices, histBuf, parallel);
        } else {
            ShortBuffer imageBuf = prepareImage16Buffer(imageOffset);
            IHistNative.histogram16(effectiveBits, imageBuf, maskBuf,
                                    effectiveHeight, effectiveWidth,
                                    imageStride, effectiveMaskStride,
                                    nComponents, indices, histBuf, parallel);
        }

        // Copy back from temp buffer if used (view buffer case)
        if (tempHistBuf != null) {
            tempHistBuf.rewind();
            IntBuffer dup = workBuffer.duplicate();
            for (int i = 0; i < histSize; i++) {
                dup.put(tempHistBuf.get());
            }
        }

        // Set position and limit for the returned buffer
        workBuffer.position(originalPos).limit(originalPos + histSize);
        return workBuffer;
    }

    // Clear buffer contents (for non-accumulate mode)
    private static void clearBuffer(IntBuffer buf, int size) {
        if (buf.hasArray()) {
            int offset = buf.arrayOffset() + buf.position();
            Arrays.fill(buf.array(), offset, offset + size, 0);
        } else {
            IntBuffer dup = buf.duplicate();
            for (int i = 0; i < size; i++) {
                dup.put(0);
            }
        }
    }

    private void validate() {
        if (imageWidth < 0 || imageHeight < 0) {
            throw new IllegalArgumentException(
                "Image dimensions must be non-negative");
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

        // Validate mask dimensions if mask is set
        if (maskBuffer != null) {
            if (maskWidth <= 0 || maskHeight <= 0) {
                throw new IllegalArgumentException(
                    "Mask dimensions must be positive");
            }
            if (maskOffsetX < 0 || maskOffsetY < 0) {
                throw new IllegalArgumentException(
                    "Mask offset cannot be negative");
            }
            if (maskOffsetX + effectiveWidth > maskWidth ||
                maskOffsetY + effectiveHeight > maskHeight) {
                throw new IllegalArgumentException("ROI exceeds mask bounds");
            }
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

        // Validate output buffer is not read-only
        if (outputBuffer != null && outputBuffer.isReadOnly()) {
            throw new IllegalArgumentException(
                "output histogram buffer cannot be read-only");
        }
    }

    private static int[] defaultComponentIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        return indices;
    }

    // Prepare 8-bit image buffer for JNI call
    private ByteBuffer prepareImage8Buffer(int imageOffset) {
        ByteBuffer buf = image8Buffer;
        if (buf.isDirect() || buf.hasArray()) {
            // Supported by JNI; apply offset
            ByteBuffer sliced = buf.duplicate();
            sliced.position(sliced.position() + imageOffset);
            return sliced;
        }
        // Unsupported buffer type; copy to temp direct buffer
        return copyToDirectByteBuffer(buf, imageOffset);
    }

    // Prepare 16-bit image buffer for JNI call
    private ShortBuffer prepareImage16Buffer(int imageOffset) {
        ShortBuffer buf = image16Buffer;
        if (buf.isDirect() || buf.hasArray()) {
            // Supported by JNI; apply offset
            ShortBuffer sliced = buf.duplicate();
            sliced.position(sliced.position() + imageOffset);
            return sliced;
        }
        // Unsupported buffer type; copy to temp direct buffer
        return copyToDirectShortBuffer(buf, imageOffset);
    }

    // Prepare mask buffer for JNI call
    private ByteBuffer prepareMaskBuffer(int maskOffset) {
        if (maskBuffer == null) {
            return null;
        }
        if (maskBuffer.isDirect() || maskBuffer.hasArray()) {
            // Supported by JNI; apply offset
            ByteBuffer sliced = maskBuffer.duplicate();
            sliced.position(sliced.position() + maskOffset);
            return sliced;
        }
        // Unsupported buffer type; copy to temp direct buffer
        return copyToDirectByteBuffer(maskBuffer, maskOffset);
    }

    // Copy ByteBuffer to direct buffer (for unsupported buffer types)
    private static ByteBuffer copyToDirectByteBuffer(ByteBuffer src,
                                                     int offset) {
        ByteBuffer srcDup = src.duplicate();
        srcDup.position(srcDup.position() + offset);
        int remaining = srcDup.remaining();
        ByteBuffer direct = ByteBuffer.allocateDirect(remaining);
        direct.put(srcDup);
        direct.flip();
        return direct;
    }

    // Copy ShortBuffer to direct buffer (for unsupported buffer types)
    private static ShortBuffer copyToDirectShortBuffer(ShortBuffer src,
                                                       int offset) {
        ShortBuffer srcDup = src.duplicate();
        srcDup.position(srcDup.position() + offset);
        int remaining = srcDup.remaining();
        ByteBuffer bb = ByteBuffer.allocateDirect(remaining * 2)
                            .order(ByteOrder.nativeOrder());
        ShortBuffer direct = bb.asShortBuffer();
        direct.put(srcDup);
        direct.flip();
        return direct;
    }

    // Allocate direct IntBuffer
    private static IntBuffer allocateDirectIntBuffer(int size) {
        ByteBuffer bb =
            ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder());
        return bb.asIntBuffer();
    }
}

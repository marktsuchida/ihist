// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import org.junit.jupiter.api.*;

/**
 * Tests for the high-level APIs {@link Histogram} and {@link
 * HistogramRequest}.
 */
class HistogramTest {

    @BeforeAll
    static void loadLibrary() {
        IHistNative.loadNativeLibrary();
    }

    @Nested
    class CoreTests {

        @Test
        void basicUsage() {
            byte[] image = {0, 1, 2, 3};
            IntBuffer hist = HistogramRequest.forImage(image, 4, 1).compute();

            assertEquals(256, hist.remaining());
            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void withBits() {
            byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
            IntBuffer hist =
                HistogramRequest.forImage(image, 8, 1).bits(3).compute();

            assertEquals(8, hist.remaining()); // 2^3 = 8 bins
            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void withBits0() {
            byte[] image = {0, 1, 0, 3, 0, 5, 6, 7};
            IntBuffer hist =
                HistogramRequest.forImage(image, 8, 1).bits(0).compute();

            assertEquals(1, hist.remaining()); // 2^0 = 1 bin
            assertEquals(3, hist.get(0));      // count of zero-valued pixels
        }
    }

    @Nested
    class RoiTests {

        @Test
        void withRoi() {
            // 4x2 image, ROI is middle 2x1
            byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
            IntBuffer hist = HistogramRequest.forImage(image, 4, 2)
                                 .roi(1, 0, 2, 1)
                                 .compute();

            assertEquals(1, hist.get(1)); // Only values 1, 2 from ROI
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(0));
            assertEquals(0, hist.get(3));
        }

        @Test
        void roiAtCorner() {
            byte[] image = new byte[16]; // 4x4
            for (int i = 0; i < 16; i++)
                image[i] = (byte)i;

            // ROI at bottom-right corner
            IntBuffer hist = HistogramRequest.forImage(image, 4, 4)
                                 .roi(2, 2, 2, 2)
                                 .compute();

            // Pixels: 10, 11, 14, 15
            assertEquals(1, hist.get(10));
            assertEquals(1, hist.get(11));
            assertEquals(1, hist.get(14));
            assertEquals(1, hist.get(15));
            assertEquals(0, hist.get(0));
        }

        @Test
        void fullImageRoi() {
            byte[] image = {0, 1, 2, 3};

            IntBuffer hist1 = HistogramRequest.forImage(image, 4, 1).compute();
            IntBuffer hist2 = HistogramRequest.forImage(image, 4, 1)
                                  .roi(0, 0, 4, 1)
                                  .compute();

            // Results should be identical
            for (int i = 0; i < 256; i++) {
                assertEquals(hist1.get(i), hist2.get(i));
            }
        }
    }

    @Nested
    class MaskTests {

        @Test
        void withMask() {
            byte[] image = {0, 1, 2, 3};
            byte[] mask = {1, 0, 1, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void heapByteBufferWithMask() {
            ByteBuffer image = ByteBuffer.allocate(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocate(4);
            mask.put(new byte[] {1, 0, 1, 0});
            mask.flip();

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void directByteBufferWithMask() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocateDirect(4);
            mask.put(new byte[] {1, 0, 1, 0});
            mask.flip();

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void viewByteBufferWithMask() {
            ByteBuffer original = ByteBuffer.allocate(4);
            original.put(new byte[] {0, 1, 2, 3});
            original.flip();
            ByteBuffer imageView = original.asReadOnlyBuffer();

            ByteBuffer maskOrig = ByteBuffer.allocate(4);
            maskOrig.put(new byte[] {1, 0, 1, 0});
            maskOrig.flip();
            ByteBuffer maskView = maskOrig.asReadOnlyBuffer();

            IntBuffer hist = HistogramRequest.forImage(imageView, 4, 1)
                                 .mask(maskView, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void maskOffset() {
            byte[] image = {0, 1, 2, 3}; // 4x1
            // Larger mask, offset into it
            byte[] mask = {0, 0, 1, 0, 1, 0, 0, 0}; // 8x1

            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1)
                    .mask(mask, 8, 1)
                    .maskOffset(2, 0) // Start reading mask at position 2
                    .compute();

            // Mask values at offset 2: 1, 0, 1, 0 -> include pixels 0, 2
            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void allPixelsMasked() {
            byte[] image = {0, 1, 2, 3};
            byte[] mask = {0, 0, 0, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .compute();

            // All bins should be zero
            for (int i = 0; i < 256; i++) {
                assertEquals(0, hist.get(i));
            }
        }
    }

    @Nested
    class ComponentTests {

        @Test
        void withComponents() {
            byte[] image = {10, 20, 11, 21}; // 2-pixel, 2-component image
            IntBuffer hist =
                HistogramRequest.forImage(image, 2, 1, 2).compute();

            assertEquals(2 * 256, hist.remaining());
            assertEquals(1, hist.get(10));
            assertEquals(1, hist.get(11));
            assertEquals(1, hist.get(256 + 20));
            assertEquals(1, hist.get(256 + 21));
        }

        @Test
        void selectComponents() {
            // RGBA, select only G and A
            byte[] image = {10, 20, 30, 40, 11, 21, 31, 41};
            IntBuffer hist = HistogramRequest.forImage(image, 2, 1, 4)
                                 .selectComponents(1, 3) // G and A
                                 .compute();

            assertEquals(2 * 256, hist.remaining());
            // First histogram is G channel
            assertEquals(1, hist.get(20));
            assertEquals(1, hist.get(21));
            // Second histogram is A channel
            assertEquals(1, hist.get(256 + 40));
            assertEquals(1, hist.get(256 + 41));
        }

        @Test
        void byteBufferWithComponents() {
            ByteBuffer image = ByteBuffer.allocate(8);
            image.put(new byte[] {10, 20, 11, 21, 12, 22, 13,
                                  23}); // 4 pixels, 2 components
            image.flip();

            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1, 2).compute();

            assertEquals(2 * 256, hist.remaining());
            assertEquals(1, hist.get(10));
            assertEquals(1, hist.get(11));
            assertEquals(1, hist.get(12));
            assertEquals(1, hist.get(13));
            assertEquals(1, hist.get(256 + 20));
            assertEquals(1, hist.get(256 + 21));
            assertEquals(1, hist.get(256 + 22));
            assertEquals(1, hist.get(256 + 23));
        }

        @Test
        void emptySelectComponents() {
            byte[] image = new byte[12];
            IntBuffer hist = HistogramRequest.forImage(image, 4, 1, 3)
                                 .selectComponents()
                                 .compute();
            assertEquals(0, hist.remaining());
        }
    }

    @Nested
    class Image16Tests {

        @Test
        void image16() {
            short[] image = {0, 1000, 2000, 3000};
            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1).bits(12).compute();

            assertEquals(4096, hist.remaining());
            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1000));
            assertEquals(1, hist.get(2000));
            assertEquals(1, hist.get(3000));
        }

        @Test
        void heapShortBuffer() {
            ShortBuffer image = ShortBuffer.allocate(4);
            image.put(new short[] {0, 1, 2, 3});
            image.flip();

            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1).bits(8).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void directShortBuffer() {
            ByteBuffer bb = ByteBuffer.allocateDirect(4 * 2).order(
                java.nio.ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            image.put(new short[] {0, 1, 2, 3});
            image.flip();

            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1).bits(8).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void viewShortBuffer() {
            // Create a view buffer that is neither direct nor array-backed
            ShortBuffer original = ShortBuffer.allocate(4);
            original.put(new short[] {0, 1, 2, 3});
            original.flip();
            ShortBuffer view = original.asReadOnlyBuffer();

            IntBuffer hist =
                HistogramRequest.forImage(view, 4, 1).bits(8).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void buffer16() {
            ShortBuffer image = ShortBuffer.allocate(4);
            image.put((short)0);
            image.put((short)100);
            image.put((short)200);
            image.put((short)300);
            image.flip();

            IntBuffer hist =
                HistogramRequest.forImage(image, 4, 1).bits(9).compute();

            assertEquals(512, hist.remaining());
            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(100));
            assertEquals(1, hist.get(200));
            assertEquals(1, hist.get(300));
        }

        @Test
        void multicomponent16() {
            short[] image = {100, 200, 101, 201}; // 2 pixels, 2 components
            IntBuffer hist =
                HistogramRequest.forImage(image, 2, 1, 2).bits(9).compute();

            assertEquals(2 * 512, hist.remaining());
            assertEquals(1, hist.get(100));
            assertEquals(1, hist.get(101));
            assertEquals(1, hist.get(512 + 200));
            assertEquals(1, hist.get(512 + 201));
        }

        @Test
        void selectComponents16() {
            short[] image = {100, 200, 300,
                             101, 201, 301}; // 2 pixels, 3 components
            IntBuffer hist = HistogramRequest.forImage(image, 2, 1, 3)
                                 .selectComponents(0, 2) // Skip component 1
                                 .bits(9)
                                 .compute();

            assertEquals(2 * 512, hist.remaining());
            assertEquals(1, hist.get(100));
            assertEquals(1, hist.get(101));
            assertEquals(1, hist.get(512 + 300));
            assertEquals(1, hist.get(512 + 301));
        }

        @Test
        void roi16() {
            short[] image = {0, 1, 2, 3, 4, 5, 6, 7}; // 4x2
            IntBuffer hist = HistogramRequest.forImage(image, 4, 2)
                                 .roi(1, 0, 2, 1)
                                 .bits(8)
                                 .compute();

            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(0));
            assertEquals(0, hist.get(3));
        }

        @Test
        void mask16() {
            short[] image = {0, 1, 2, 3};
            byte[] mask = {1, 0, 1, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .bits(8)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }
    }

    @Nested
    class BufferTypeTests {

        @Test
        void heapByteBuffer() {
            ByteBuffer image = ByteBuffer.allocate(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void directByteBuffer() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }

        @Test
        void viewByteBuffer() {
            // Create a view buffer via asReadOnlyBuffer() - this is neither
            // direct nor array-backed, so HistogramRequest copies to temp
            // buffer
            ByteBuffer original = ByteBuffer.allocate(4);
            original.put(new byte[] {0, 1, 2, 3});
            original.flip();
            ByteBuffer view = original.asReadOnlyBuffer();

            IntBuffer hist = HistogramRequest.forImage(view, 4, 1).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
        }
    }

    @Nested
    class OutputBufferTests {

        @Test
        void withPreallocatedOutput() {
            byte[] image = {0, 1, 2, 3};
            int[] hist = new int[256];
            hist[0] = 100; // Pre-existing value

            IntBuffer result = HistogramRequest.forImage(image, 4, 1)
                                   .output(hist)
                                   .accumulate(false) // Should zero first
                                   .compute();

            assertSame(hist, result.array());
            assertEquals(1, hist[0]); // Was zeroed, then 1 added
        }

        @Test
        void arrayImageDirectHistogram() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .compute();

            // Result shares storage but is a different buffer object
            assertNotSame(histogram, result);
            // Original buffer's position/limit unchanged
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Data is accessible via original buffer
            assertEquals(1, histogram.get(0));
            assertEquals(1, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(1, histogram.get(3));
        }

        @Test
        void directImageArrayHistogram() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            int[] histogram = new int[256];

            IntBuffer result = HistogramRequest.forImage(image, 4, 1)
                                   .output(histogram)
                                   .compute();

            assertSame(histogram, result.array());
            assertEquals(1, histogram[0]);
            assertEquals(1, histogram[1]);
            assertEquals(1, histogram[2]);
            assertEquals(1, histogram[3]);
        }

        @Test
        void viewImageDirectHistogram() {
            // View buffer for image (will be copied), direct for histogram
            ByteBuffer original = ByteBuffer.allocate(4);
            original.put(new byte[] {0, 1, 2, 3});
            original.flip();
            ByteBuffer view = original.asReadOnlyBuffer();

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();

            IntBuffer result = HistogramRequest.forImage(view, 4, 1)
                                   .output(histogram)
                                   .compute();

            // Result shares storage but is a different buffer object
            assertNotSame(histogram, result);
            // Original buffer's position/limit unchanged
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Data is accessible via original buffer
            assertEquals(1, histogram.get(0));
            assertEquals(1, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(1, histogram.get(3));
        }

        @Test
        void directImageArrayMask() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            byte[] maskData = {1, 0, 1, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(maskData, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }

        @Test
        void directHistogramBufferNoAccumulate() {
            byte[] imageData = {0, 1, 2, 3};

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();
            // Pre-fill with values that should be cleared
            for (int i = 0; i < 256; i++) {
                histogram.put(i, 100);
            }

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .accumulate(false)
                                   .compute();

            assertNotSame(histogram, result);
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Should have been zeroed first, then histogram computed
            assertEquals(1, result.get(0));
            assertEquals(1, result.get(1));
            assertEquals(1, result.get(2));
            assertEquals(1, result.get(3));
            assertEquals(0, result.get(4)); // Other bins should be zero
            assertEquals(0, result.get(100));
        }

        @Test
        void directHistogramBufferAccumulate() {
            byte[] imageData = {0, 1, 2, 3};

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();
            // Pre-fill with values that should be accumulated
            histogram.put(0, 100);
            histogram.put(1, 200);

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .accumulate(true)
                                   .compute();

            assertNotSame(histogram, result);
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Should have accumulated
            assertEquals(101, result.get(0)); // 100 + 1
            assertEquals(201, result.get(1)); // 200 + 1
            assertEquals(1, result.get(2));
            assertEquals(1, result.get(3));
        }

        @Test
        void viewHistogramBufferNoAccumulate() {
            byte[] imageData = {0, 1, 2, 3};

            // Create a view buffer (asIntBuffer on a heap ByteBuffer)
            ByteBuffer heapBuf =
                ByteBuffer.allocate(256 * 4).order(ByteOrder.nativeOrder());
            IntBuffer histogram = heapBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();
            // Pre-fill with values that should be zeroed
            for (int i = 0; i < 10; i++) {
                histogram.put(i, 100);
            }

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .accumulate(false)
                                   .compute();

            assertNotSame(histogram, result);
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Should have been zeroed first, then histogram computed
            assertEquals(1, result.get(0));
            assertEquals(1, result.get(1));
            assertEquals(1, result.get(2));
            assertEquals(1, result.get(3));
            for (int i = 4; i < 10; i++) {
                assertEquals(0, result.get(i));
            }
        }

        @Test
        void viewHistogramBufferAccumulate() {
            byte[] imageData = {0, 1, 2, 3};

            // Create a view buffer (asIntBuffer on a heap ByteBuffer)
            ByteBuffer heapBuf =
                ByteBuffer.allocate(256 * 4).order(ByteOrder.nativeOrder());
            IntBuffer histogram = heapBuf.asIntBuffer();
            int origPos = histogram.position();
            int origLimit = histogram.limit();
            // Pre-fill with values that should be accumulated
            histogram.put(0, 100);
            histogram.put(1, 200);

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .accumulate(true)
                                   .compute();

            assertNotSame(histogram, result);
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());
            // Should have accumulated
            assertEquals(101, result.get(0)); // 100 + 1
            assertEquals(201, result.get(1)); // 200 + 1
            assertEquals(1, result.get(2));
            assertEquals(1, result.get(3));
        }

        @Test
        void outputBufferPositionPreserved() {
            byte[] imageData = {0, 1, 2, 3};

            // Create a buffer with position at 100 and remaining = 256
            ByteBuffer histBuf = ByteBuffer.allocateDirect(512 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            histogram.position(100).limit(356); // remaining = 256
            int origPos = histogram.position();
            int origLimit = histogram.limit();

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .compute();

            // Original buffer's position/limit must be unchanged
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());

            // Result buffer should cover the histogram at offset 100
            assertEquals(100, result.position());
            assertEquals(356, result.limit()); // 100 + 256
            assertEquals(256, result.remaining());

            // Data written at the correct offset
            assertEquals(1, result.get(100));
            assertEquals(1, histogram.get(100));
        }

        @Test
        void outputBufferWithOffsetPreserved() {
            byte[] imageData = {0, 1, 2, 3};

            // Create a buffer positioned at 50 with remaining = 256
            ByteBuffer histBuf = ByteBuffer.allocateDirect(512 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            histogram.position(50).limit(306); // remaining = 256
            int origPos = histogram.position();
            int origLimit = histogram.limit();

            IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                                   .output(histogram)
                                   .compute();

            // Original buffer's position/limit must be unchanged
            assertEquals(origPos, histogram.position());
            assertEquals(origLimit, histogram.limit());

            // Result should start at 50 and end at 50 + 256 = 306
            assertEquals(50, result.position());
            assertEquals(306, result.limit());
            assertEquals(256, result.remaining());

            // Data written correctly
            assertEquals(1, result.get(50));
            assertEquals(1, histogram.get(50));
        }
    }

    @Nested
    class AccumulationTests {

        @Test
        void accumulate() {
            byte[] image = {0, 1};
            int[] hist = new int[256];
            hist[0] = 100;

            HistogramRequest.forImage(image, 2, 1)
                .output(hist)
                .accumulate(true)
                .compute();

            assertEquals(101, hist[0]);
            assertEquals(1, hist[1]);
        }

        @Test
        void accumulateMultipleImages() {
            byte[] image1 = {0, 1};
            byte[] image2 = {0, 2};
            byte[] image3 = {0, 0};
            int[] hist = new int[256];

            HistogramRequest.forImage(image1, 2, 1)
                .output(hist)
                .accumulate(true)
                .compute();
            HistogramRequest.forImage(image2, 2, 1)
                .output(hist)
                .accumulate(true)
                .compute();
            HistogramRequest.forImage(image3, 2, 1)
                .output(hist)
                .accumulate(true)
                .compute();

            assertEquals(4, hist[0]); // 0 appears 4 times total
            assertEquals(1, hist[1]);
            assertEquals(1, hist[2]);
        }
    }

    @Nested
    class ParallelTests {

        @Test
        void parallel() {
            byte[] image = new byte[1000 * 1000];
            for (int i = 0; i < image.length; i++) {
                image[i] = (byte)(i % 256);
            }

            IntBuffer hist1 = HistogramRequest.forImage(image, 1000, 1000)
                                  .parallel(true)
                                  .compute();

            IntBuffer hist2 = HistogramRequest.forImage(image, 1000, 1000)
                                  .parallel(false)
                                  .compute();

            assertArrayEquals(hist1.array(), hist2.array());
        }
    }

    @Nested
    class FeatureInteractionTests {

        @Test
        void roiWithMask() {
            // 4x2 image, ROI is 2x2, mask excludes some pixels
            byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
            byte[] mask = {1, 0, 1,
                           0}; // Only include pixels 0,2 of the 2x2 ROI

            IntBuffer hist = HistogramRequest.forImage(image, 4, 2)
                                 .roi(1, 0, 2, 2) // Select columns 1-2
                                 .mask(mask, 2, 2)
                                 .compute();

            // ROI pixels: 1,2,5,6. With mask: only 1,5 (positions 0,2 in ROI)
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(5));
            assertEquals(0, hist.get(2));
            assertEquals(0, hist.get(6));
        }

        @Test
        void roiWithComponents() {
            // 4x1 RGB image, ROI selects middle 2 pixels
            byte[] image = {10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1, 3)
                                 .roi(1, 0, 2, 1)
                                 .compute();

            assertEquals(3 * 256, hist.remaining());
            // R: 11, 12
            assertEquals(1, hist.get(11));
            assertEquals(1, hist.get(12));
            assertEquals(0, hist.get(10));
            assertEquals(0, hist.get(13));
        }

        @Test
        void roiWithBits() {
            // 4x4 image with values 0-15
            byte[] image = new byte[16];
            for (int i = 0; i < 16; i++)
                image[i] = (byte)i;

            IntBuffer hist = HistogramRequest.forImage(image, 4, 4)
                                 .roi(1, 1, 2, 2)
                                 .bits(4)
                                 .compute();

            assertEquals(16, hist.remaining());
            // ROI pixels at positions (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10
            // Values: 5, 6, 9, 10 -> with bits(4), values 0-15 map to bins
            // 0-15
            assertEquals(1, hist.get(5));
            assertEquals(1, hist.get(6));
            assertEquals(1, hist.get(9));
            assertEquals(1, hist.get(10));
        }

        @Test
        void maskWithComponents() {
            byte[] image = {10, 20, 11, 21,
                            12, 22, 13, 23}; // 4 pixels, 2 components
            byte[] mask = {1, 0, 1, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1, 2)
                                 .mask(mask, 4, 1)
                                 .compute();

            assertEquals(2 * 256, hist.remaining());
            assertEquals(1, hist.get(10));
            assertEquals(1, hist.get(12));
            assertEquals(0, hist.get(11));
            assertEquals(0, hist.get(13));
        }

        @Test
        void maskWithBits() {
            // Values 0-3 with bits(2) gives 4 bins, each value maps to its own
            // bin
            byte[] image = {0, 1, 2, 3};
            byte[] mask = {1, 0, 1, 0};

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .bits(2)
                                 .compute();

            assertEquals(4, hist.remaining());
            assertEquals(1, hist.get(0)); // value 0, included by mask
            assertEquals(0, hist.get(1)); // value 1, excluded by mask
            assertEquals(1, hist.get(2)); // value 2, included by mask
            assertEquals(0, hist.get(3)); // value 3, excluded by mask
        }

        @Test
        void selectComponentsWithBits() {
            // 2 pixels, 4 components each. Values 0-7 fit in 3 bits.
            byte[] image = {0, 1, 2, 3,
                            4, 5, 6, 7}; // 2 pixels, 4 components (RGBA)

            IntBuffer hist = HistogramRequest.forImage(image, 2, 1, 4)
                                 .selectComponents(0, 2) // R and B only
                                 .bits(3)
                                 .compute();

            assertEquals(2 * 8, hist.remaining());
            // First component (R): pixel0=0, pixel1=4 -> bins 0, 4
            // Third component (B): pixel0=2, pixel1=6 -> bins 2, 6 (offset by
            // 8)
            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(4));
            assertEquals(1, hist.get(8 + 2));
            assertEquals(1, hist.get(8 + 6));
        }
    }

    @Nested
    class EdgeCaseTests {

        @Test
        void singlePixelImage() {
            byte[] image = {42};
            IntBuffer hist = HistogramRequest.forImage(image, 1, 1).compute();

            assertEquals(256, hist.remaining());
            assertEquals(1, hist.get(42));
        }

        @Test
        void singleRowImage() {
            byte[] image = {0, 1, 2, 3, 4};
            IntBuffer hist = HistogramRequest.forImage(image, 5, 1).compute();

            for (int i = 0; i < 5; i++) {
                assertEquals(1, hist.get(i));
            }
        }

        @Test
        void singleColumnImage() {
            byte[] image = {0, 1, 2, 3, 4};
            IntBuffer hist = HistogramRequest.forImage(image, 1, 5).compute();

            for (int i = 0; i < 5; i++) {
                assertEquals(1, hist.get(i));
            }
        }

        @Test
        void uniformImage() {
            byte[] image = new byte[100];
            Arrays.fill(image, (byte)42);

            IntBuffer hist =
                HistogramRequest.forImage(image, 10, 10).compute();

            assertEquals(100, hist.get(42));
            assertEquals(0, hist.get(0));
            assertEquals(0, hist.get(255));
        }

        @Test
        void maximumPixelValues8() {
            byte[] image = {(byte)127, (byte)128, (byte)255};
            IntBuffer hist = HistogramRequest.forImage(image, 3, 1).compute();

            assertEquals(1, hist.get(127));
            assertEquals(1, hist.get(128));
            assertEquals(1, hist.get(255));
        }

        @Test
        void maximumPixelValues16() {
            short[] image = {32767, (short)32768, (short)65535};
            IntBuffer hist = HistogramRequest.forImage(image, 3, 1).compute();

            assertEquals(1, hist.get(32767));
            assertEquals(1, hist.get(32768));
            assertEquals(1, hist.get(65535));
        }

        @Test
        void imageBufferWithPosition() {
            ByteBuffer image = ByteBuffer.allocate(8);
            image.put(new byte[] {99, 99, 0, 1, 2, 3, 99, 99});
            image.position(2).limit(6); // Position at start of actual data

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1).compute();

            assertEquals(1, hist.get(0));
            assertEquals(1, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(1, hist.get(3));
            assertEquals(0, hist.get(99));
        }

        @Test
        void maskBufferWithPosition() {
            byte[] image = {0, 1, 2, 3};
            ByteBuffer mask = ByteBuffer.allocate(8);
            mask.put(new byte[] {99, 99, 1, 0, 1, 0, 99, 99});
            mask.position(2).limit(6);

            IntBuffer hist = HistogramRequest.forImage(image, 4, 1)
                                 .mask(mask, 4, 1)
                                 .compute();

            assertEquals(1, hist.get(0));
            assertEquals(0, hist.get(1));
            assertEquals(1, hist.get(2));
            assertEquals(0, hist.get(3));
        }
    }
}

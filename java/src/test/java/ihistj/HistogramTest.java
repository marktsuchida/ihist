// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
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

    @Test
    void basicUsage() {
        byte[] image = {0, 1, 2, 3};
        int[] hist = HistogramRequest.forImage(image, 4, 1).compute();

        assertEquals(256, hist.length);
        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void withRoi() {
        // 4x2 image, ROI is middle 2x1
        byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] hist =
            HistogramRequest.forImage(image, 4, 2).roi(1, 0, 2, 1).compute();

        assertEquals(1, hist[1]); // Only values 1, 2 from ROI
        assertEquals(1, hist[2]);
        assertEquals(0, hist[0]);
        assertEquals(0, hist[3]);
    }

    @Test
    void withMask() {
        byte[] image = {0, 1, 2, 3};
        byte[] mask = {1, 0, 1, 0};

        int[] hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(0, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(0, hist[3]);
    }

    @Test
    void withComponents() {
        byte[] image = {10, 20, 11, 21}; // 2-pixel, 2-component image
        int[] hist = HistogramRequest.forImage(image, 2, 1, 2).compute();

        assertEquals(2 * 256, hist.length);
        assertEquals(1, hist[10]);
        assertEquals(1, hist[11]);
        assertEquals(1, hist[256 + 20]);
        assertEquals(1, hist[256 + 21]);
    }

    @Test
    void selectComponents() {
        // RGBA, select only G and A
        byte[] image = {10, 20, 30, 40, 11, 21, 31, 41};
        int[] hist = HistogramRequest.forImage(image, 2, 1, 4)
                         .selectComponents(1, 3) // G and A
                         .compute();

        assertEquals(2 * 256, hist.length);
        // First histogram is G channel
        assertEquals(1, hist[20]);
        assertEquals(1, hist[21]);
        // Second histogram is A channel
        assertEquals(1, hist[256 + 40]);
        assertEquals(1, hist[256 + 41]);
    }

    @Test
    void withBits() {
        byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] hist = HistogramRequest.forImage(image, 8, 1).bits(3).compute();

        assertEquals(8, hist.length); // 2^3 = 8 bins
        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void withPreallocatedOutput() {
        byte[] image = {0, 1, 2, 3};
        int[] hist = new int[256];
        hist[0] = 100; // Pre-existing value

        int[] result = HistogramRequest.forImage(image, 4, 1)
                           .output(hist)
                           .accumulate(false) // Should zero first
                           .compute();

        assertSame(hist, result);
        assertEquals(1, hist[0]); // Was zeroed, then 1 added
    }

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
    void parallel() {
        byte[] image = new byte[1000 * 1000];
        for (int i = 0; i < image.length; i++) {
            image[i] = (byte)(i % 256);
        }

        int[] hist1 = HistogramRequest.forImage(image, 1000, 1000)
                          .parallel(true)
                          .compute();

        int[] hist2 = HistogramRequest.forImage(image, 1000, 1000)
                          .parallel(false)
                          .compute();

        assertArrayEquals(hist1, hist2);
    }

    @Test
    void image16() {
        short[] image = {0, 1000, 2000, 3000};
        int[] hist = HistogramRequest.forImage(image, 4, 1).bits(12).compute();

        assertEquals(4096, hist.length);
        assertEquals(1, hist[0]);
        assertEquals(1, hist[1000]);
        assertEquals(1, hist[2000]);
        assertEquals(1, hist[3000]);
    }

    @Test
    void buffer16() {
        ShortBuffer image = ShortBuffer.allocate(4);
        image.put((short)0);
        image.put((short)100);
        image.put((short)200);
        image.put((short)300);
        image.flip();

        int[] hist = HistogramRequest.forImage(image, 4, 1).bits(9).compute();

        assertEquals(512, hist.length);
        assertEquals(1, hist[0]);
        assertEquals(1, hist[100]);
        assertEquals(1, hist[200]);
        assertEquals(1, hist[300]);
    }

    // Tests for heap vs direct buffer handling

    @Test
    void heapByteBuffer() {
        ByteBuffer image = ByteBuffer.allocate(4);
        image.put(new byte[] {0, 1, 2, 3});
        image.flip();

        int[] hist = HistogramRequest.forImage(image, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void directByteBuffer() {
        ByteBuffer image = ByteBuffer.allocateDirect(4);
        image.put(new byte[] {0, 1, 2, 3});
        image.flip();

        int[] hist = HistogramRequest.forImage(image, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void heapShortBuffer() {
        ShortBuffer image = ShortBuffer.allocate(4);
        image.put(new short[] {0, 1, 2, 3});
        image.flip();

        int[] hist = HistogramRequest.forImage(image, 4, 1).bits(8).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void directShortBuffer() {
        ByteBuffer bb = ByteBuffer.allocateDirect(4 * 2).order(
            java.nio.ByteOrder.nativeOrder());
        ShortBuffer image = bb.asShortBuffer();
        image.put(new short[] {0, 1, 2, 3});
        image.flip();

        int[] hist = HistogramRequest.forImage(image, 4, 1).bits(8).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void heapByteBufferWithMask() {
        ByteBuffer image = ByteBuffer.allocate(4);
        image.put(new byte[] {0, 1, 2, 3});
        image.flip();

        ByteBuffer mask = ByteBuffer.allocate(4);
        mask.put(new byte[] {1, 0, 1, 0});
        mask.flip();

        int[] hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(0, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(0, hist[3]);
    }

    @Test
    void directByteBufferWithMask() {
        ByteBuffer image = ByteBuffer.allocateDirect(4);
        image.put(new byte[] {0, 1, 2, 3});
        image.flip();

        ByteBuffer mask = ByteBuffer.allocateDirect(4);
        mask.put(new byte[] {1, 0, 1, 0});
        mask.flip();

        int[] hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(0, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(0, hist[3]);
    }

    // Tests for view buffers (automatically copied to temp direct buffer)

    @Test
    void viewByteBuffer() {
        // Create a view buffer via asReadOnlyBuffer() - this is neither
        // direct nor array-backed, so HistogramRequest copies to temp buffer
        ByteBuffer original = ByteBuffer.allocate(4);
        original.put(new byte[] {0, 1, 2, 3});
        original.flip();
        ByteBuffer view = original.asReadOnlyBuffer();

        int[] hist = HistogramRequest.forImage(view, 4, 1).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
    }

    @Test
    void viewShortBuffer() {
        // Create a view buffer that is neither direct nor array-backed
        ShortBuffer original = ShortBuffer.allocate(4);
        original.put(new short[] {0, 1, 2, 3});
        original.flip();
        ShortBuffer view = original.asReadOnlyBuffer();

        int[] hist = HistogramRequest.forImage(view, 4, 1).bits(8).compute();

        assertEquals(1, hist[0]);
        assertEquals(1, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(1, hist[3]);
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

        int[] hist = HistogramRequest.forImage(imageView, 4, 1)
                         .mask(maskView, 4, 1)
                         .compute();

        assertEquals(1, hist[0]);
        assertEquals(0, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(0, hist[3]);
    }

    // Tests for mixed buffer types

    @Test
    void arrayImageDirectHistogram() {
        byte[] imageData = {0, 1, 2, 3};
        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();

        int[] result = HistogramRequest.forImage(imageData, 4, 1)
                           .output(histogram)
                           .compute();

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

        int[] result =
            HistogramRequest.forImage(image, 4, 1).output(histogram).compute();

        assertSame(histogram, result);
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

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();

        HistogramRequest.forImage(view, 4, 1).output(histogram).compute();

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

        int[] hist = HistogramRequest.forImage(image, 4, 1)
                         .mask(maskData, 4, 1)
                         .compute();

        assertEquals(1, hist[0]);
        assertEquals(0, hist[1]);
        assertEquals(1, hist[2]);
        assertEquals(0, hist[3]);
    }

    // Test read-only histogram buffer rejection

    @Test
    void readOnlyHistogramBufferRejected() {
        byte[] imageData = {0, 1, 2, 3};
        IntBuffer histogram = IntBuffer.allocate(256).asReadOnlyBuffer();

        assertThrows(IllegalArgumentException.class, () -> {
            HistogramRequest.forImage(imageData, 4, 1)
                .output(histogram)
                .compute();
        });
    }

    // Tests for direct output buffer accumulation

    @Test
    void directHistogramBufferNoAccumulate() {
        byte[] imageData = {0, 1, 2, 3};

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        // Pre-fill with values that should be cleared
        for (int i = 0; i < 256; i++) {
            histogram.put(i, 100);
        }

        int[] result = HistogramRequest.forImage(imageData, 4, 1)
                           .output(histogram)
                           .accumulate(false)
                           .compute();

        // Should have been zeroed first, then histogram computed
        assertEquals(1, result[0]);
        assertEquals(1, result[1]);
        assertEquals(1, result[2]);
        assertEquals(1, result[3]);
        assertEquals(0, result[4]); // Other bins should be zero
        assertEquals(0, result[100]);

        // Direct buffer should also have the results
        assertEquals(1, histogram.get(0));
        assertEquals(1, histogram.get(1));
    }

    @Test
    void directHistogramBufferAccumulate() {
        byte[] imageData = {0, 1, 2, 3};

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        // Pre-fill with values that should be accumulated
        histogram.put(0, 100);
        histogram.put(1, 200);

        int[] result = HistogramRequest.forImage(imageData, 4, 1)
                           .output(histogram)
                           .accumulate(true)
                           .compute();

        // Should have accumulated
        assertEquals(101, result[0]); // 100 + 1
        assertEquals(201, result[1]); // 200 + 1
        assertEquals(1, result[2]);
        assertEquals(1, result[3]);

        // Direct buffer should also have the accumulated results
        assertEquals(101, histogram.get(0));
        assertEquals(201, histogram.get(1));
    }

    // Tests for view output buffer accumulation

    @Test
    void viewHistogramBufferNoAccumulate() {
        byte[] imageData = {0, 1, 2, 3};

        // Create a view buffer (asIntBuffer on a heap ByteBuffer)
        ByteBuffer heapBuf =
            ByteBuffer.allocate(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = heapBuf.asIntBuffer();
        // Pre-fill with values that should be zeroed
        for (int i = 0; i < 10; i++) {
            histogram.put(i, 100);
        }

        int[] result = HistogramRequest.forImage(imageData, 4, 1)
                           .output(histogram)
                           .accumulate(false)
                           .compute();

        // Should have been zeroed first, then histogram computed
        assertEquals(1, result[0]);
        assertEquals(1, result[1]);
        assertEquals(1, result[2]);
        assertEquals(1, result[3]);
        for (int i = 4; i < 10; i++) {
            assertEquals(0, result[i]);
        }

        // View buffer should also have the results
        assertEquals(1, histogram.get(0));
        assertEquals(1, histogram.get(1));
    }

    @Test
    void viewHistogramBufferAccumulate() {
        byte[] imageData = {0, 1, 2, 3};

        // Create a view buffer (asIntBuffer on a heap ByteBuffer)
        ByteBuffer heapBuf =
            ByteBuffer.allocate(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = heapBuf.asIntBuffer();
        // Pre-fill with values that should be accumulated
        histogram.put(0, 100);
        histogram.put(1, 200);

        int[] result = HistogramRequest.forImage(imageData, 4, 1)
                           .output(histogram)
                           .accumulate(true)
                           .compute();

        // Should have accumulated
        assertEquals(101, result[0]); // 100 + 1
        assertEquals(201, result[1]); // 200 + 1
        assertEquals(1, result[2]);
        assertEquals(1, result[3]);

        // View buffer should also have the accumulated results
        assertEquals(101, histogram.get(0));
        assertEquals(201, histogram.get(1));
    }
}

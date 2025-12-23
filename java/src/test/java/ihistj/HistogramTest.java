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
        IntBuffer hist = HistogramRequest.forImage(image, 4, 1).compute();

        assertEquals(256, hist.remaining());
        assertEquals(1, hist.get(0));
        assertEquals(1, hist.get(1));
        assertEquals(1, hist.get(2));
        assertEquals(1, hist.get(3));
    }

    @Test
    void withRoi() {
        // 4x2 image, ROI is middle 2x1
        byte[] image = {0, 1, 2, 3, 4, 5, 6, 7};
        IntBuffer hist =
            HistogramRequest.forImage(image, 4, 2).roi(1, 0, 2, 1).compute();

        assertEquals(1, hist.get(1)); // Only values 1, 2 from ROI
        assertEquals(1, hist.get(2));
        assertEquals(0, hist.get(0));
        assertEquals(0, hist.get(3));
    }

    @Test
    void withMask() {
        byte[] image = {0, 1, 2, 3};
        byte[] mask = {1, 0, 1, 0};

        IntBuffer hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

        assertEquals(1, hist.get(0));
        assertEquals(0, hist.get(1));
        assertEquals(1, hist.get(2));
        assertEquals(0, hist.get(3));
    }

    @Test
    void withComponents() {
        byte[] image = {10, 20, 11, 21}; // 2-pixel, 2-component image
        IntBuffer hist = HistogramRequest.forImage(image, 2, 1, 2).compute();

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

        IntBuffer hist1 = HistogramRequest.forImage(image, 1000, 1000)
                              .parallel(true)
                              .compute();

        IntBuffer hist2 = HistogramRequest.forImage(image, 1000, 1000)
                              .parallel(false)
                              .compute();

        assertArrayEquals(hist1.array(), hist2.array());
    }

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

    // Tests for heap vs direct buffer handling

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
    void heapByteBufferWithMask() {
        ByteBuffer image = ByteBuffer.allocate(4);
        image.put(new byte[] {0, 1, 2, 3});
        image.flip();

        ByteBuffer mask = ByteBuffer.allocate(4);
        mask.put(new byte[] {1, 0, 1, 0});
        mask.flip();

        IntBuffer hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

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

        IntBuffer hist =
            HistogramRequest.forImage(image, 4, 1).mask(mask, 4, 1).compute();

        assertEquals(1, hist.get(0));
        assertEquals(0, hist.get(1));
        assertEquals(1, hist.get(2));
        assertEquals(0, hist.get(3));
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

        IntBuffer hist = HistogramRequest.forImage(view, 4, 1).compute();

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

    // Tests for mixed buffer types

    @Test
    void arrayImageDirectHistogram() {
        byte[] imageData = {0, 1, 2, 3};
        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
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

        IntBuffer result =
            HistogramRequest.forImage(image, 4, 1).output(histogram).compute();

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

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int origPos = histogram.position();
        int origLimit = histogram.limit();

        IntBuffer result =
            HistogramRequest.forImage(view, 4, 1).output(histogram).compute();

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

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
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

    // Tests for view output buffer accumulation

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

    // Tests for buffer position/limit preservation

    @Test
    void outputBufferPositionPreserved() {
        byte[] imageData = {0, 1, 2, 3};

        // Create a buffer and position it at offset 100
        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(512 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        histogram.position(100);
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
    void outputBufferLimitBeyondHistogramPreserved() {
        byte[] imageData = {0, 1, 2, 3};

        // Create a buffer with limit well beyond histogram size
        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(1024 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        // Set limit to 1024 (well beyond 256-bin histogram)
        histogram.limit(1024);
        int origPos = histogram.position();
        int origLimit = histogram.limit();

        IntBuffer result = HistogramRequest.forImage(imageData, 4, 1)
                               .output(histogram)
                               .compute();

        // Original buffer's position/limit must be unchanged
        assertEquals(origPos, histogram.position());
        assertEquals(origLimit, histogram.limit());

        // Result buffer's limit should be at histogram end, not original limit
        assertEquals(0, result.position());
        assertEquals(256, result.limit());
        assertEquals(256, result.remaining());
    }

    @Test
    void outputBufferWithOffsetAndLargeLimitPreserved() {
        byte[] imageData = {0, 1, 2, 3};

        // Create a buffer positioned at 50 with limit at 800
        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(1024 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        histogram.position(50).limit(800);
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

// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
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

    @Nested
    class HistogramConvenienceTests {

        @Test
        void grayscale8() {
            byte[] image = {0, 0, 1, 2, 2, 2};
            int[] hist = Histogram.compute(image, 6, 1);

            assertEquals(256, hist.length);
            assertEquals(2, hist[0]);
            assertEquals(1, hist[1]);
            assertEquals(3, hist[2]);
        }

        @Test
        void rgb8() {
            byte[] image = {10, 20, 30, 10, 21, 31};
            int[] hist = Histogram.compute(image, 2, 1, 3);

            assertEquals(3 * 256, hist.length);
            assertEquals(2, hist[10]);       // R
            assertEquals(1, hist[256 + 20]); // G
            assertEquals(1, hist[256 + 21]); // G
            assertEquals(1, hist[512 + 30]); // B
            assertEquals(1, hist[512 + 31]); // B
        }

        @Test
        void grayscale16() {
            short[] image = {0, 0, 1000, 2000};
            int[] hist = Histogram.compute(image, 4, 1, 12);

            assertEquals(4096, hist.length);
            assertEquals(2, hist[0]);
            assertEquals(1, hist[1000]);
            assertEquals(1, hist[2000]);
        }

        @Test
        void withBuffer() {
            ByteBuffer image = ByteBuffer.allocate(4);
            image.put((byte)0);
            image.put((byte)1);
            image.put((byte)1);
            image.put((byte)2);
            image.flip();

            int[] hist = Histogram.compute(image, 4, 1);

            assertEquals(1, hist[0]);
            assertEquals(2, hist[1]);
            assertEquals(1, hist[2]);
        }

        @Test
        void selectComponentsConvenience() {
            // RGBA image, skip alpha
            byte[] image = {10, 20, 30, (byte)255, 11, 21, 31, (byte)255};
            int[] hist = Histogram.compute(image, 2, 1, 4, 0, 1, 2);

            assertEquals(3 * 256, hist.length); // Only 3 components
        }
    }

    @Nested
    class HistogramRequestTests {

        @Test
        void basicUsage() {
            byte[] image = {0, 1, 2, 3};
            int[] hist = HistogramRequest.forImage8(image, 4, 1).compute();

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
            int[] hist = HistogramRequest.forImage8(image, 4, 2)
                             .roi(1, 0, 2, 1)
                             .compute();

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
                HistogramRequest.forImage8(image, 4, 1).mask(mask).compute();

            assertEquals(1, hist[0]);
            assertEquals(0, hist[1]);
            assertEquals(1, hist[2]);
            assertEquals(0, hist[3]);
        }

        @Test
        void withStride() {
            // 2x2 image with stride=4
            byte[] image = {1, 2, 99, 99, 3, 4, 99, 99};
            int[] hist =
                HistogramRequest.forImage8(image, 2, 2).stride(4).compute();

            assertEquals(1, hist[1]);
            assertEquals(1, hist[2]);
            assertEquals(1, hist[3]);
            assertEquals(1, hist[4]);
            assertEquals(0, hist[99]);
        }

        @Test
        void withComponents() {
            byte[] image = {10, 20, 11, 21}; // 2-pixel, 2-component image
            int[] hist = HistogramRequest.forImage8(image, 2, 1)
                             .components(2)
                             .compute();

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
            int[] hist = HistogramRequest.forImage8(image, 2, 1)
                             .components(4)
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
            int[] hist =
                HistogramRequest.forImage8(image, 8, 1).bits(3).compute();

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

            int[] result = HistogramRequest.forImage8(image, 4, 1)
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

            HistogramRequest.forImage8(image, 2, 1)
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

            int[] hist1 = HistogramRequest.forImage8(image, 1000, 1000)
                              .parallel(true)
                              .compute();

            int[] hist2 = HistogramRequest.forImage8(image, 1000, 1000)
                              .parallel(false)
                              .compute();

            assertArrayEquals(hist1, hist2);
        }

        @Test
        void image16() {
            short[] image = {0, 1000, 2000, 3000};
            int[] hist =
                HistogramRequest.forImage16(image, 4, 1).bits(12).compute();

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

            int[] hist =
                HistogramRequest.forImage16(image, 4, 1).bits(9).compute();

            assertEquals(512, hist.length);
            assertEquals(1, hist[0]);
            assertEquals(1, hist[100]);
            assertEquals(1, hist[200]);
            assertEquals(1, hist[300]);
        }
    }
}

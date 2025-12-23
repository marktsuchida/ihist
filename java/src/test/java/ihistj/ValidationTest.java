// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import org.junit.jupiter.api.*;

/**
 * Tests for parameter validation in both JNI and high-level APIs.
 */
class ValidationTest {

    @BeforeAll
    static void loadLibrary() {
        IHistNative.loadNativeLibrary();
    }

    @Nested
    class IHistNativeValidationTests {

        @Test
        void invalidSampleBits8() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(0, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(9, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void invalidSampleBits16() {
            short[] image = {0, 1, 2};
            int[] histogram = new int[65536];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram16(0, image, 0, null, 0,
                                                        1, 3, 3, 3, 1, indices,
                                                        histogram, 0, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram16(17, image, 0, null, 0,
                                                        1, 3, 3, 3, 1, indices,
                                                        histogram, 0, false));
        }

        @Test
        void invalidStride() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            // imageStride < width
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 2, 3, 1, indices,
                                                       histogram, 0, false));

            // maskStride < width
            byte[] mask = {1, 1, 1};
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, mask, 0, 1,
                                                       3, 3, 2, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void invalidNComponents() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 0, indices,
                                                       histogram, 0, false));
        }

        @Test
        void nullComponentIndices() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, null,
                                                       histogram, 0, false));
        }

        @Test
        void emptyComponentIndices() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void componentIndexOutOfRange() {
            byte[] image = {0, 1, 2, 3, 4, 5}; // 2 pixels, 3 components each
            int[] histogram = new int[256];
            int[] indices = {0,
                             3}; // Index 3 is out of range for nComponents=3

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       2, 2, 2, 3, indices,
                                                       histogram, 0, false));
        }

        @Test
        void negativeComponentIndex() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {-1};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void negativeOffset() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, -1, null, 0,
                                                       1, 3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void imageTooSmall() {
            byte[] image = {0, 1}; // Only 2 elements
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void histogramTooSmall() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[128]; // Too small for 8 bits
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void nullImage() {
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, null, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }

        @Test
        void nullHistogram() {
            byte[] image = {0, 1, 2};
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       3, 3, 3, 1, indices,
                                                       null, 0, false));
        }

        @Test
        void negativeDimensions() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0,
                                                       -1, 3, 3, 3, 1, indices,
                                                       histogram, 0, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, 0, null, 0, 1,
                                                       -3, 3, 3, 1, indices,
                                                       histogram, 0, false));
        }
    }

    @Nested
    class HistogramRequestValidationTests {

        @Test
        void nullImage() {
            assertThrows(
                IllegalArgumentException.class,
                () -> HistogramRequest.forImage((byte[])null, 10, 10));
        }

        @Test
        void negativeDimensions() {
            byte[] image = {0, 1, 2};

            assertThrows(IllegalArgumentException.class,
                         () -> HistogramRequest.forImage(image, -1, 1));

            assertThrows(IllegalArgumentException.class,
                         () -> HistogramRequest.forImage(image, 1, -1));
        }

        @Test
        void invalidComponents() {
            byte[] image = {0, 1, 2, 3};

            assertThrows(IllegalArgumentException.class,
                         () -> HistogramRequest.forImage(image, 4, 1, 0));
        }

        @Test
        void invalidBits8() {
            byte[] image = {0, 1, 2, 3};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .bits(0)
                                    .compute());

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .bits(9)
                                    .compute());
        }

        @Test
        void invalidBits16() {
            short[] image = {0, 1, 2, 3};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .bits(0)
                                    .compute());

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .bits(17)
                                    .compute());
        }

        @Test
        void roiExceedsBounds() {
            byte[] image = new byte[100];

            // ROI extends beyond image width
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .roi(8, 0, 5, 5)
                                    .compute());

            // ROI extends beyond image height
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .roi(0, 8, 5, 5)
                                    .compute());
        }

        @Test
        void negativeRoiOffset() {
            byte[] image = new byte[100];

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .roi(-1, 0, 5, 5)
                                    .compute());

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .roi(0, -1, 5, 5)
                                    .compute());
        }

        @Test
        void componentIndexOutOfRange() {
            byte[] image = new byte[12]; // 4 pixels, 3 components

            assertThrows(
                IllegalArgumentException.class,
                ()
                    -> HistogramRequest.forImage(image, 4, 1, 3)
                           .selectComponents(0, 1, 3) // 3 is out of range
                           .compute());
        }

        @Test
        void negativeComponentIndex() {
            byte[] image = new byte[12];

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1, 3)
                                    .selectComponents(-1, 0)
                                    .compute());
        }

        @Test
        void directOutputBufferTooSmall() {
            // Direct image buffer with direct output buffer that's too small
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            // Create a direct IntBuffer that's too small (need 256 for 8 bits)
            ByteBuffer bb = ByteBuffer.allocateDirect(128 * 4).order(
                java.nio.ByteOrder.nativeOrder());
            IntBuffer tooSmall = bb.asIntBuffer();

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .output(tooSmall)
                                    .compute());
        }
    }
}

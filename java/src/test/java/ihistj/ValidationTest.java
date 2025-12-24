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
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(0, image, null, 1, 3, 3,
                                                       3, 1, indices,
                                                       histogram, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(9, image, null, 1, 3, 3,
                                                       3, 1, indices,
                                                       histogram, false));
        }

        @Test
        void invalidSampleBits16() {
            short[] imageData = {0, 1, 2};
            ShortBuffer image = ShortBuffer.wrap(imageData);
            int[] histData = new int[65536];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram16(0, image, null, 1, 3,
                                                        3, 3, 1, indices,
                                                        histogram, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram16(17, image, null, 1, 3,
                                                        3, 3, 1, indices,
                                                        histogram, false));
        }

        @Test
        void invalidStride() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            // imageStride < width
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 2,
                                                       3, 1, indices,
                                                       histogram, false));

            // maskStride < width
            byte[] maskData = {1, 1, 1};
            ByteBuffer mask = ByteBuffer.wrap(maskData);
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, mask, 1, 3, 3,
                                                       2, 1, indices,
                                                       histogram, false));
        }

        @Test
        void invalidNComponents() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 3,
                                                       3, 0, indices,
                                                       histogram, false));
        }

        @Test
        void nullComponentIndices() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 3,
                                                       3, 1, null, histogram,
                                                       false));
        }

        @Test
        void emptyComponentIndices() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 3,
                                                       3, 1, indices,
                                                       histogram, false));
        }

        @Test
        void componentIndexOutOfRange() {
            byte[] imageData = {0, 1, 2,
                                3, 4, 5}; // 2 pixels, 3 components each
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0,
                             3}; // Index 3 is out of range for nComponents=3

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 2, 2,
                                                       2, 3, indices,
                                                       histogram, false));
        }

        @Test
        void negativeComponentIndex() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {-1};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 3,
                                                       3, 1, indices,
                                                       histogram, false));
        }

        @Test
        void nullImageBuffer() {
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, null, null, 1, 3, 3,
                                                       3, 1, indices,
                                                       histogram, false));
        }

        @Test
        void nullHistogramBuffer() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 3, 3,
                                                       3, 1, indices, null,
                                                       false));
        }

        @Test
        void negativeDimensions() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, -1, 3,
                                                       3, 3, 1, indices,
                                                       histogram, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, -3,
                                                       3, 3, 1, indices,
                                                       histogram, false));
        }

        @Test
        void readOnlyHistogramBufferRejected() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            IntBuffer histogram = IntBuffer.allocate(256).asReadOnlyBuffer();
            int[] indices = {0};

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 4, 4,
                                                       4, 1, indices,
                                                       histogram, false));
        }

        @Test
        void viewBufferImageRejected() {
            // Create a view buffer that is neither direct nor array-backed
            ByteBuffer original = ByteBuffer.allocate(4);
            original.put(new byte[] {0, 1, 2, 3});
            original.flip();
            ByteBuffer view = original.asReadOnlyBuffer();

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            // View buffer should be rejected at JNI level
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, view, null, 1, 4, 4,
                                                       4, 1, indices,
                                                       histogram, false));
        }

        @Test
        void viewBufferMaskRejected() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);

            // Create a view buffer mask
            ByteBuffer original = ByteBuffer.allocate(4);
            original.put(new byte[] {1, 0, 1, 0});
            original.flip();
            ByteBuffer maskView = original.asReadOnlyBuffer();

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            // View buffer mask should be rejected at JNI level
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, maskView, 1,
                                                       4, 4, 4, 1, indices,
                                                       histogram, false));
        }

        @Test
        void viewBufferHistogramRejected() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);

            // Create a view buffer histogram (neither direct nor array-backed)
            // Note: IntBuffer.allocate().asReadOnlyBuffer() creates a
            // read-only buffer which should also be rejected
            ByteBuffer bb = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer directHist = bb.asIntBuffer();
            // asReadOnlyBuffer creates a view that's read-only
            IntBuffer viewHist = directHist.asReadOnlyBuffer();

            int[] indices = {0};

            // This should be rejected (read-only)
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram8(8, image, null, 1, 4, 4,
                                                       4, 1, indices, viewHist,
                                                       false));
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
                ByteOrder.nativeOrder());
            IntBuffer tooSmall = bb.asIntBuffer();

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .output(tooSmall)
                                    .compute());
        }

        @Test
        void readOnlyHistogramBufferRejected() {
            byte[] image = {0, 1, 2, 3};
            IntBuffer histogram = IntBuffer.allocate(256).asReadOnlyBuffer();

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 1)
                                    .output(histogram)
                                    .compute());
        }

        @Test
        void maskTooSmallForRoi() {
            byte[] image = new byte[100]; // 10x10 image
            byte[] mask = new byte[25];   // 5x5 mask (too small for 10x10 ROI)

            // Without explicit ROI, ROI defaults to full image (10x10)
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .mask(mask, 5, 5)
                                    .compute());
        }

        @Test
        void maskTooSmallForRoiWithOffset() {
            byte[] image = new byte[100]; // 10x10 image
            byte[] mask = new byte[100];  // 10x10 mask

            // ROI is 5x5, maskOffset is (6,6), so 6+5=11 > 10
            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 10, 10)
                                    .roi(0, 0, 5, 5)
                                    .mask(mask, 10, 10)
                                    .maskOffset(6, 6)
                                    .compute());
        }

        @Test
        void negativeMaskOffset() {
            byte[] image = new byte[16]; // 4x4 image
            byte[] mask = new byte[16];  // 4x4 mask

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 4)
                                    .mask(mask, 4, 4)
                                    .maskOffset(-1, 0)
                                    .compute());

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> HistogramRequest.forImage(image, 4, 4)
                                    .mask(mask, 4, 4)
                                    .maskOffset(0, -1)
                                    .compute());
        }

        @Test
        void maskWithValidOffset() {
            byte[] image = new byte[100]; // 10x10 image
            byte[] mask = new byte[100];  // 10x10 mask
            java.util.Arrays.fill(mask, (byte)1);

            // ROI is 5x5 at (2,2), maskOffset is (3,3)
            // 3+5=8 <= 10, so this should succeed
            IntBuffer result = HistogramRequest.forImage(image, 10, 10)
                                   .roi(2, 2, 5, 5)
                                   .mask(mask, 10, 10)
                                   .maskOffset(3, 3)
                                   .compute();

            assertNotNull(result);
        }
    }
}

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
 * Tests for the low-level JNI wrapper {@link IHistNative}.
 */
class IHistNativeTest {

    @BeforeAll
    static void loadLibrary() {
        IHistNative.loadNativeLibrary();
    }

    @Nested
    class Histogram8ArrayBackedTests {

        // Tests using array-backed buffers (via ByteBuffer.wrap)

        @Test
        void simpleGrayscale() {
            byte[] imageData = {0, 1, 1, 2, 2, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 6, 6, 6, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(2, histData[1]);
            assertEquals(3, histData[2]);
        }

        @Test
        void withOffset() {
            byte[] imageData = {99, 99, 0, 1, 2}; // Start at offset 2
            ByteBuffer image = ByteBuffer.wrap(imageData);
            image.position(2); // Set position to skip first 2 bytes
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
        }

        @Test
        void withHistogramOffset() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[512]; // Extra space at beginning
            IntBuffer histogram = IntBuffer.wrap(histData);
            histogram.position(256); // Set position to skip first 256 ints
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);

            assertEquals(0, histData[0]); // Not at offset 0
            assertEquals(1, histData[256 + 0]);
            assertEquals(1, histData[256 + 1]);
            assertEquals(1, histData[256 + 2]);
        }

        @Test
        void maskWithOffset() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            byte[] maskData = {99, 99, 1, 0, 1, 0};
            ByteBuffer mask = ByteBuffer.wrap(maskData);
            mask.position(2);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, mask, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(0, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(0, histData[3]);
        }

        @Test
        void emptyComponentIndices() {
            byte[] imageData = {0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {};

            IHistNative.histogram8(8, image, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);
        }

        @Test
        void emptyImage() {
            byte[] imageData = {};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            // Should not throw for empty image
            IHistNative.histogram8(8, image, null, 0, 0, 0, 0, 1, indices,
                                   histogram, false);

            // Histogram should be unchanged (all zeros)
            for (int i = 0; i < 256; i++) {
                assertEquals(0, histData[i]);
            }
        }

        @Test
        void unsignedByteInterpretation() {
            // Java bytes 127 to -128 correspond to unsigned 127 to 128
            byte[] imageData = {127, (byte)128, (byte)255};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[127]);
            assertEquals(1, histData[128]); // -128 in Java = 128 unsigned
            assertEquals(1, histData[255]); // -1 in Java = 255 unsigned
        }

        @Test
        void strideHandlingMultiRow() {
            // 2x2 image with stride=4; gaps contain sentinel value 99
            byte[] imageData = {0, 1, 99, 99, 2, 3, 99, 99};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 2, 2, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(1, histData[3]);
            assertEquals(0, histData[99]); // Stride gaps not counted
        }
    }

    @Nested
    class Histogram8DirectBufferTests {

        @Test
        void directBuffer() {
            ByteBuffer image = ByteBuffer.allocateDirect(256);
            for (int i = 0; i < 256; i++) {
                image.put((byte)i);
            }
            image.flip();

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 256, 256, 256, 1,
                                   indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void bufferPosition() {
            ByteBuffer image = ByteBuffer.allocateDirect(260);
            image.position(4); // Skip first 4 bytes
            for (int i = 0; i < 256; i++) {
                image.put((byte)i);
            }
            image.position(4); // Reset to start of data

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 256, 256, 256, 1,
                                   indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void directBufferWithMask() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocateDirect(4);
            mask.put(new byte[] {1, 0, 1, 0}); // Include pixels 0 and 2
            mask.flip();

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram8(8, image, mask, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(0, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(0, histogram.get(3));
        }

        @Test
        void directBufferMaskWithPosition() {
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocateDirect(8);
            mask.put(new byte[] {99, 99, 99, 99, 1, 0, 1, 0});
            mask.position(4);

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram8(8, image, mask, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(0, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(0, histogram.get(3));
        }
    }

    @Nested
    class Histogram8MixedBufferTests {

        @Test
        void directImageArrayHistogram() {
            // Direct buffer for image, array-backed for histogram
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(1, histData[3]);
        }

        @Test
        void arrayImageDirectHistogram() {
            // Array-backed buffer for image, direct buffer for histogram
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(1, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(1, histogram.get(3));
        }

        @Test
        void directImageArrayMaskArrayHistogram() {
            // Direct image, array-backed mask, array-backed histogram
            ByteBuffer image = ByteBuffer.allocateDirect(4);
            image.put(new byte[] {0, 1, 2, 3});
            image.flip();

            byte[] maskData = {1, 0, 1, 0};
            ByteBuffer mask = ByteBuffer.wrap(maskData);

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, mask, 1, 4, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(0, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(0, histData[3]);
        }
    }

    @Nested
    class Histogram16ArrayBackedTests {

        @Test
        void simpleGrayscale16() {
            short[] imageData = {0, 1, 1, 2, 2, 2};
            ShortBuffer image = ShortBuffer.wrap(imageData);
            int[] histData = new int[65536];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram16(16, image, null, 1, 6, 6, 6, 1, indices,
                                    histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(2, histData[1]);
            assertEquals(3, histData[2]);
        }

        @Test
        void unsignedShortInterpretation() {
            // Java shorts are signed; test high values
            short[] imageData = {32767, (short)32768, (short)65535};
            ShortBuffer image = ShortBuffer.wrap(imageData);
            int[] histData = new int[65536];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram16(16, image, null, 1, 3, 3, 3, 1, indices,
                                    histogram, false);

            assertEquals(1, histData[32767]);
            assertEquals(1,
                         histData[32768]); // -32768 in Java = 32768 unsigned
            assertEquals(1, histData[65535]); // -1 in Java = 65535 unsigned
        }
    }

    @Nested
    class Histogram16DirectBufferTests {

        @Test
        void directShortBuffer() {
            ByteBuffer bb =
                ByteBuffer.allocateDirect(512).order(ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            for (int i = 0; i < 256; i++) {
                image.put((short)i);
            }
            image.flip();

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram16(8, image, null, 1, 256, 256, 256, 1,
                                    indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void directShortBufferWithMask() {
            ByteBuffer bb = ByteBuffer.allocateDirect(4 * 2).order(
                ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            image.put(new short[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocateDirect(4);
            mask.put(new byte[] {1, 0, 1, 0}); // Include pixels 0 and 2
            mask.flip();

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram16(8, image, mask, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(0, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(0, histogram.get(3));
        }

        @Test
        void directMaskWithPosition() {
            ByteBuffer bb = ByteBuffer.allocateDirect(4 * 2).order(
                ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            image.put(new short[] {0, 1, 2, 3});
            image.flip();

            ByteBuffer mask = ByteBuffer.allocateDirect(8);
            mask.put(new byte[] {99, 99, 99, 99, 1, 0, 1, 0});
            mask.position(4);

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram16(8, image, mask, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(0, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(0, histogram.get(3));
        }

        @Test
        void exactBoundaryCapacity() {
            // 2 rows, width=2, stride=3: need (2-1)*3+2 = 5 elements
            ByteBuffer exactBb = ByteBuffer.allocateDirect(5 * 2).order(
                ByteOrder.nativeOrder());
            ShortBuffer exact = exactBb.asShortBuffer();
            exact.put(new short[] {0, 1, 99, 2, 3});
            exact.flip();

            ByteBuffer tooSmallBb = ByteBuffer.allocateDirect(4 * 2).order(
                ByteOrder.nativeOrder());
            ShortBuffer tooSmall = tooSmallBb.asShortBuffer();
            tooSmall.put(new short[] {0, 1, 99, 2});
            tooSmall.flip();

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            assertDoesNotThrow(
                ()
                    -> IHistNative.histogram16(8, exact, null, 2, 2, 3, 3, 1,
                                               indices, histogram, false));

            assertThrows(IllegalArgumentException.class,
                         ()
                             -> IHistNative.histogram16(8, tooSmall, null, 2,
                                                        2, 3, 3, 1, indices,
                                                        histogram, false));
        }
    }

    @Nested
    class Histogram16MixedBufferTests {

        @Test
        void directImageArrayHistogram() {
            // Direct buffer for image, array-backed for histogram
            ByteBuffer bb =
                ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            image.put(new short[] {0, 1, 2, 3});
            image.flip();

            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram16(8, image, null, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(1, histData[0]);
            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(1, histData[3]);
        }

        @Test
        void arrayImageDirectHistogram() {
            // Array-backed buffer for image, direct buffer for histogram
            short[] imageData = {0, 1, 2, 3};
            ShortBuffer image = ShortBuffer.wrap(imageData);

            ByteBuffer histBuf = ByteBuffer.allocateDirect(256 * 4).order(
                ByteOrder.nativeOrder());
            IntBuffer histogram = histBuf.asIntBuffer();
            int[] indices = {0};

            IHistNative.histogram16(8, image, null, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(1, histogram.get(0));
            assertEquals(1, histogram.get(1));
            assertEquals(1, histogram.get(2));
            assertEquals(1, histogram.get(3));
        }
    }

    @Nested
    class ParallelTests {

        @Test
        void parallelParameter() {
            byte[] imageData = new byte[1000];
            for (int i = 0; i < 1000; i++) {
                imageData[i] = (byte)(i % 256);
            }
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 10, 100, 100, 100, 1,
                                   indices, histogram, true);

            int total = 0;
            for (int i = 0; i < 256; i++) {
                total += histData[i];
            }
            assertEquals(1000, total);
        }
    }

    @Nested
    class AccumulationTests {

        @Test
        void histogramAccumulation() {
            byte[] imageData1 = {0, 1, 2};
            ByteBuffer image1 = ByteBuffer.wrap(imageData1);
            byte[] imageData2 = {0, 0, 3};
            ByteBuffer image2 = ByteBuffer.wrap(imageData2);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image1, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);
            IHistNative.histogram8(8, image2, null, 1, 3, 3, 3, 1, indices,
                                   histogram, false);

            assertEquals(3, histData[0]); // 1 from image1 + 2 from image2
            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(1, histData[3]);
        }
    }

    @Nested
    class EdgeCaseTests {

        @Test
        void sampleBitsZero() {
            // With 0 bits, only value 0 maps to bin 0; higher values are out
            // of range
            byte[] imageData = {0, 0, 0, 1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[1];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(0, image, null, 1, 5, 5, 5, 1, indices,
                                   histogram, false);

            assertEquals(3, histData[0]); // Only the 3 zero-valued pixels
        }

        @Test
        void sampleBitsZero16() {
            // With 0 bits, only value 0 maps to bin 0; higher values are out
            // of range
            short[] imageData = {0, 0, 1000, (short)65535};
            ShortBuffer image = ShortBuffer.wrap(imageData);
            int[] histData = new int[1];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram16(0, image, null, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(2, histData[0]); // Only the 2 zero-valued pixels
        }
    }
}

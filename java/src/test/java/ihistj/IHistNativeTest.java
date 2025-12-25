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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

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
        void rgbImage() {
            // 2x2 RGB image
            byte[] imageData = {10, 20, 30, 11, 21, 31,
                                12, 22, 32, 13, 23, 33};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[3 * 256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0, 1, 2};

            IHistNative.histogram8(8, image, null, 2, 2, 2, 2, 3, indices,
                                   histogram, false);

            // Check R channel
            assertEquals(1, histData[10]);
            assertEquals(1, histData[11]);
            assertEquals(1, histData[12]);
            assertEquals(1, histData[13]);

            // Check G channel (offset by 256)
            assertEquals(1, histData[256 + 20]);
            assertEquals(1, histData[256 + 21]);
            assertEquals(1, histData[256 + 22]);
            assertEquals(1, histData[256 + 23]);

            // Check B channel (offset by 512)
            assertEquals(1, histData[512 + 30]);
            assertEquals(1, histData[512 + 31]);
        }

        @Test
        void withMask() {
            byte[] imageData = {0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            byte[] maskData = {1, 0, 1, 0}; // Include only pixels 0 and 2
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

        @ParameterizedTest
        @ValueSource(booleans = {true, false})
        void parallelFlag(boolean parallel) {
            byte[] imageData = new byte[1000 * 1000];
            Arrays.fill(imageData, (byte)42);
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1000, 1000, 1000, 1000, 1,
                                   indices, histogram, parallel);

            assertEquals(1000000, histData[42]);
        }

        @Test
        void accumulate() {
            byte[] imageData = {1, 2};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            histData[1] = 10;
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 2, 2, 2, 1, indices,
                                   histogram, false);

            // Should accumulate
            assertEquals(11, histData[1]);
            assertEquals(1, histData[2]);
        }

        @Test
        void reducedBits() {
            // With sample_bits=2, only values 0-3 are valid (4 bins).
            byte[] imageData = {0, 1, 2, 3, 0, 1, 2, 3};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[4]; // 2 bits = 4 bins
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(2, image, null, 1, 8, 8, 8, 1, indices,
                                   histogram, false);

            assertEquals(2, histData[0]); // Two 0s
            assertEquals(2, histData[1]); // Two 1s
            assertEquals(2, histData[2]); // Two 2s
            assertEquals(2, histData[3]); // Two 3s
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
        void selectComponents() {
            // 2-pixel RGBA image, only histogram R and B (skip G and A)
            byte[] imageData = {10, 20, 30, (byte)255, 11, 21, 31, (byte)255};
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[2 * 256]; // 2 components
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0, 2}; // R and B only

            IHistNative.histogram8(8, image, null, 1, 2, 2, 2, 4, indices,
                                   histogram, false);

            // Check R channel (first histogram)
            assertEquals(1, histData[10]);
            assertEquals(1, histData[11]);

            // Check B channel (second histogram, offset 256)
            assertEquals(1, histData[256 + 30]);
            assertEquals(1, histData[256 + 31]);
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
        void stride() {
            // Image with padding (stride > width)
            // 2x2 image with stride=4 (2 padding bytes per row)
            byte[] imageData = {
                1, 2, 99, 99, // Row 0: data, padding
                3, 4, 99, 99  // Row 1: data, padding
            };
            ByteBuffer image = ByteBuffer.wrap(imageData);
            int[] histData = new int[256];
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 2, 2, 4, 4, 1, indices,
                                   histogram, false);

            assertEquals(1, histData[1]);
            assertEquals(1, histData[2]);
            assertEquals(1, histData[3]);
            assertEquals(1, histData[4]);
            assertEquals(0, histData[99]); // Padding not counted
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
        void reducedBits16() {
            // With sample_bits=9, values 0-511 are valid (512 bins).
            short[] imageData = {0, 255, 256, 511};
            ShortBuffer image = ShortBuffer.wrap(imageData);
            int[] histData = new int[512]; // 9 bits = 512 bins
            IntBuffer histogram = IntBuffer.wrap(histData);
            int[] indices = {0};

            IHistNative.histogram16(9, image, null, 1, 4, 4, 4, 1, indices,
                                    histogram, false);

            assertEquals(1, histData[0]);   // 0 -> bin 0
            assertEquals(1, histData[255]); // 255 -> bin 255
            assertEquals(1, histData[256]); // 256 -> bin 256
            assertEquals(1, histData[511]); // 511 -> bin 511
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
}

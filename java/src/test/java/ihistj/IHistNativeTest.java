// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
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
    class Histogram8ArrayTests {

        @Test
        void simpleGrayscale() {
            byte[] image = {0, 1, 1, 2, 2, 2};
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 1, 6, 6, 6, 1,
                                   indices, histogram, 0, false);

            assertEquals(1, histogram[0]);
            assertEquals(2, histogram[1]);
            assertEquals(3, histogram[2]);
        }

        @Test
        void rgbImage() {
            // 2x2 RGB image
            byte[] image = {10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33};
            int[] histogram = new int[3 * 256];
            int[] indices = {0, 1, 2};

            IHistNative.histogram8(8, image, 0, null, 0, 2, 2, 2, 2, 3,
                                   indices, histogram, 0, false);

            // Check R channel
            assertEquals(1, histogram[10]);
            assertEquals(1, histogram[11]);
            assertEquals(1, histogram[12]);
            assertEquals(1, histogram[13]);

            // Check G channel (offset by 256)
            assertEquals(1, histogram[256 + 20]);
            assertEquals(1, histogram[256 + 21]);
            assertEquals(1, histogram[256 + 22]);
            assertEquals(1, histogram[256 + 23]);

            // Check B channel (offset by 512)
            assertEquals(1, histogram[512 + 30]);
            assertEquals(1, histogram[512 + 31]);
        }

        @Test
        void withMask() {
            byte[] image = {0, 1, 2, 3};
            byte[] mask = {1, 0, 1, 0}; // Include only pixels 0 and 2
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, mask, 0, 1, 4, 4, 4, 1,
                                   indices, histogram, 0, false);

            assertEquals(1, histogram[0]);
            assertEquals(0, histogram[1]);
            assertEquals(1, histogram[2]);
            assertEquals(0, histogram[3]);
        }

        @ParameterizedTest
        @ValueSource(booleans = {true, false})
        void parallelFlag(boolean parallel) {
            byte[] image = new byte[1000 * 1000];
            Arrays.fill(image, (byte)42);
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 1000, 1000, 1000,
                                   1000, 1, indices, histogram, 0, parallel);

            assertEquals(1000000, histogram[42]);
        }

        @Test
        void accumulate() {
            byte[] image = {1, 2};
            int[] histogram = new int[256];
            histogram[1] = 10;
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 1, 2, 2, 2, 1,
                                   indices, histogram, 0, false);

            // Should accumulate
            assertEquals(11, histogram[1]);
            assertEquals(1, histogram[2]);
        }

        @Test
        void reducedBits() {
            // With sample_bits=2, only values 0-3 are valid (4 bins).
            // Values with bits beyond sample_bits are discarded.
            byte[] image = {0, 1, 2, 3, 0, 1, 2, 3};
            int[] histogram = new int[4]; // 2 bits = 4 bins
            int[] indices = {0};

            IHistNative.histogram8(2, image, 0, null, 0, 1, 8, 8, 8, 1,
                                   indices, histogram, 0, false);

            assertEquals(2, histogram[0]); // Two 0s
            assertEquals(2, histogram[1]); // Two 1s
            assertEquals(2, histogram[2]); // Two 2s
            assertEquals(2, histogram[3]); // Two 3s
        }

        @Test
        void withOffset() {
            byte[] image = {99, 99, 0, 1, 2}; // Start at offset 2
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 2, null, 0, 1, 3, 3, 3, 1,
                                   indices, histogram, 0, false);

            assertEquals(1, histogram[0]);
            assertEquals(1, histogram[1]);
            assertEquals(1, histogram[2]);
        }

        @Test
        void withHistogramOffset() {
            byte[] image = {0, 1, 2};
            int[] histogram = new int[512]; // Extra space at beginning
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 1, 3, 3, 3, 1,
                                   indices, histogram, 256, false);

            assertEquals(0, histogram[0]); // Not at offset 0
            assertEquals(1, histogram[256 + 0]);
            assertEquals(1, histogram[256 + 1]);
            assertEquals(1, histogram[256 + 2]);
        }

        @Test
        void selectComponents() {
            // 2-pixel RGBA image, only histogram R and B (skip G and A)
            byte[] image = {10, 20, 30, (byte)255, 11, 21, 31, (byte)255};
            int[] histogram = new int[2 * 256]; // 2 components
            int[] indices = {0, 2};             // R and B only

            IHistNative.histogram8(8, image, 0, null, 0, 1, 2, 2, 2, 4,
                                   indices, histogram, 0, false);

            // Check R channel (first histogram)
            assertEquals(1, histogram[10]);
            assertEquals(1, histogram[11]);

            // Check B channel (second histogram, offset 256)
            assertEquals(1, histogram[256 + 30]);
            assertEquals(1, histogram[256 + 31]);
        }

        @Test
        void emptyImage() {
            byte[] image = {};
            int[] histogram = new int[256];
            int[] indices = {0};

            // Should not throw for empty image
            IHistNative.histogram8(8, image, 0, null, 0, 0, 0, 0, 0, 1,
                                   indices, histogram, 0, false);

            // Histogram should be unchanged (all zeros)
            for (int i = 0; i < 256; i++) {
                assertEquals(0, histogram[i]);
            }
        }

        @Test
        void stride() {
            // Image with padding (stride > width)
            // 2x2 image with stride=4 (2 padding bytes per row)
            byte[] image = {
                1, 2, 99, 99, // Row 0: data, padding
                3, 4, 99, 99  // Row 1: data, padding
            };
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 2, 2, 4, 4, 1,
                                   indices, histogram, 0, false);

            assertEquals(1, histogram[1]);
            assertEquals(1, histogram[2]);
            assertEquals(1, histogram[3]);
            assertEquals(1, histogram[4]);
            assertEquals(0, histogram[99]); // Padding not counted
        }

        @Test
        void unsignedByteInterpretation() {
            // Java bytes 127 to -128 correspond to unsigned 127 to 128
            byte[] image = {127, (byte)128, (byte)255};
            int[] histogram = new int[256];
            int[] indices = {0};

            IHistNative.histogram8(8, image, 0, null, 0, 1, 3, 3, 3, 1,
                                   indices, histogram, 0, false);

            assertEquals(1, histogram[127]);
            assertEquals(1, histogram[128]); // -128 in Java = 128 unsigned
            assertEquals(1, histogram[255]); // -1 in Java = 255 unsigned
        }
    }

    @Nested
    class Histogram8BufferTests {

        @Test
        void directBuffer() {
            ByteBuffer image = ByteBuffer.allocateDirect(256);
            for (int i = 0; i < 256; i++) {
                image.put((byte)i);
            }
            image.flip();

            IntBuffer histogram = IntBuffer.allocate(256);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 256, 256, 256, 1,
                                   indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void heapBuffer() {
            ByteBuffer image = ByteBuffer.allocate(256);
            for (int i = 0; i < 256; i++) {
                image.put((byte)i);
            }
            image.flip();

            IntBuffer histogram = IntBuffer.allocate(256);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 256, 256, 256, 1,
                                   indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void bufferPosition() {
            ByteBuffer image = ByteBuffer.allocate(260);
            image.position(4); // Skip first 4 bytes
            for (int i = 0; i < 256; i++) {
                image.put((byte)i);
            }
            image.position(4); // Reset to start of data

            IntBuffer histogram = IntBuffer.allocate(256);
            int[] indices = {0};

            IHistNative.histogram8(8, image, null, 1, 256, 256, 256, 1,
                                   indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }
    }

    @Nested
    class Histogram16ArrayTests {

        @Test
        void simpleGrayscale16() {
            short[] image = {0, 1, 1, 2, 2, 2};
            int[] histogram = new int[65536];
            int[] indices = {0};

            IHistNative.histogram16(16, image, 0, null, 0, 1, 6, 6, 6, 1,
                                    indices, histogram, 0, false);

            assertEquals(1, histogram[0]);
            assertEquals(2, histogram[1]);
            assertEquals(3, histogram[2]);
        }

        @Test
        void reducedBits16() {
            // With sample_bits=9, values 0-511 are valid (512 bins).
            // Each value maps to its own bin. Values >= 512 are discarded.
            short[] image = {0, 255, 256, 511};
            int[] histogram = new int[512]; // 9 bits = 512 bins
            int[] indices = {0};

            IHistNative.histogram16(9, image, 0, null, 0, 1, 4, 4, 4, 1,
                                    indices, histogram, 0, false);

            assertEquals(1, histogram[0]);   // 0 -> bin 0
            assertEquals(1, histogram[255]); // 255 -> bin 255
            assertEquals(1, histogram[256]); // 256 -> bin 256
            assertEquals(1, histogram[511]); // 511 -> bin 511
        }

        @Test
        void unsignedShortInterpretation() {
            // Java shorts are signed; test high values
            short[] image = {32767, (short)32768, (short)65535};
            int[] histogram = new int[65536];
            int[] indices = {0};

            IHistNative.histogram16(16, image, 0, null, 0, 1, 3, 3, 3, 1,
                                    indices, histogram, 0, false);

            assertEquals(1, histogram[32767]);
            assertEquals(1,
                         histogram[32768]); // -32768 in Java = 32768 unsigned
            assertEquals(1, histogram[65535]); // -1 in Java = 65535 unsigned
        }
    }

    @Nested
    class Histogram16BufferTests {

        @Test
        void directShortBuffer() {
            // ShortBuffer.allocateDirect doesn't exist; we must wrap a direct
            // ByteBuffer. Use native byte order so shorts are read correctly
            // by native code.
            ByteBuffer bb = ByteBuffer.allocateDirect(512).order(
                java.nio.ByteOrder.nativeOrder());
            ShortBuffer image = bb.asShortBuffer();
            for (int i = 0; i < 256; i++) {
                image.put((short)i);
            }
            image.flip();

            IntBuffer histogram = IntBuffer.allocate(256);
            int[] indices = {0};

            IHistNative.histogram16(8, image, null, 1, 256, 256, 256, 1,
                                    indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }

        @Test
        void heapShortBuffer() {
            ShortBuffer image = ShortBuffer.allocate(256);
            for (int i = 0; i < 256; i++) {
                image.put((short)i);
            }
            image.flip();

            IntBuffer histogram = IntBuffer.allocate(256);
            int[] indices = {0};

            IHistNative.histogram16(8, image, null, 1, 256, 256, 256, 1,
                                    indices, histogram, false);

            for (int i = 0; i < 256; i++) {
                assertEquals(1, histogram.get(i));
            }
        }
    }
}

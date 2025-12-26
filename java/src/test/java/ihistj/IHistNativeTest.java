// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

package ihistj;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import org.junit.jupiter.api.*;

abstract class HistogramTestBase {

    @BeforeAll
    static void loadLibrary() {
        IHistNative.loadNativeLibrary();
    }

    abstract Buffer wrapImage(int... values);

    abstract Buffer wrapImageWithPrefix(int prefixLength, int... values);

    abstract Buffer allocateDirectImage(int size);

    abstract void putToDirectImage(Buffer buf, int... values);

    abstract int histogramSize();

    abstract int[] unsignedBoundaryValues();

    abstract void invokeHistogram(int sampleBits, Buffer image,
                                  ByteBuffer mask, int rows, int width,
                                  int stride, int maskStride, int components,
                                  int[] indices, IntBuffer histogram,
                                  boolean parallel);

    // Array-backed buffer tests

    @Test
    void simpleGrayscale() {
        Buffer image = wrapImage(0, 1, 1, 2, 2, 2);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 6, 6, 6, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(2, histData[1]);
        assertEquals(3, histData[2]);
    }

    @Test
    void withOffset() {
        Buffer image = wrapImageWithPrefix(2, 0, 1, 2);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 3, 3, 3, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(1, histData[1]);
        assertEquals(1, histData[2]);
    }

    @Test
    void withHistogramOffset() {
        Buffer image = wrapImage(0, 1, 2);
        int[] histData = new int[512];
        IntBuffer histogram = IntBuffer.wrap(histData);
        histogram.position(256);
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 3, 3, 3, 1, indices, histogram,
                        false);

        assertEquals(0, histData[0]);
        assertEquals(1, histData[256 + 0]);
        assertEquals(1, histData[256 + 1]);
        assertEquals(1, histData[256 + 2]);
    }

    @Test
    void maskWithOffset() {
        Buffer image = wrapImage(0, 1, 2, 3);
        byte[] maskData = {99, 99, 1, 0, 1, 0};
        ByteBuffer mask = ByteBuffer.wrap(maskData);
        mask.position(2);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, mask, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(0, histData[1]);
        assertEquals(1, histData[2]);
        assertEquals(0, histData[3]);
    }

    @Test
    void emptyComponentIndices() {
        Buffer image = wrapImage(0, 1, 2);
        int[] histData = new int[0];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {};

        invokeHistogram(8, image, null, 1, 3, 3, 3, 1, indices, histogram,
                        false);
    }

    @Test
    void emptyImage() {
        Buffer image = wrapImage();
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 0, 0, 0, 0, 1, indices, histogram,
                        false);

        for (int i = 0; i < 256; i++) {
            assertEquals(0, histData[i]);
        }
    }

    @Test
    void unsignedInterpretation() {
        int[] boundaries = unsignedBoundaryValues();
        Buffer image = wrapImage(boundaries[0], boundaries[1], boundaries[2]);
        int[] histData = new int[histogramSize()];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(histogramSize() == 256 ? 8 : 16, image, null, 1, 3, 3,
                        3, 1, indices, histogram, false);

        assertEquals(1, histData[boundaries[0]]);
        assertEquals(1, histData[boundaries[1]]);
        assertEquals(1, histData[boundaries[2]]);
    }

    @Test
    void strideHandlingMultiRow() {
        // height=2, width=2, stride=4: requires (2-1)*4+2 = 6 elements
        Buffer image = wrapImage(0, 1, 99, 99, 2, 3);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 2, 2, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(1, histData[1]);
        assertEquals(1, histData[2]);
        assertEquals(1, histData[3]);
        assertEquals(0, histData[99]);
    }

    // Direct buffer tests

    @Test
    void directBuffer() {
        Buffer image = allocateDirectImage(256);
        int[] vals = new int[256];
        for (int i = 0; i < 256; i++) {
            vals[i] = i;
        }
        putToDirectImage(image, vals);
        ((Buffer)image).flip();

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 256, 256, 256, 1, indices,
                        histogram, false);

        for (int i = 0; i < 256; i++) {
            assertEquals(1, histogram.get(i));
        }
    }

    @Test
    void bufferPosition() {
        Buffer image = allocateDirectImage(260);
        ((Buffer)image).position(4);
        int[] vals = new int[256];
        for (int i = 0; i < 256; i++) {
            vals[i] = i;
        }
        putToDirectImage(image, vals);
        ((Buffer)image).position(4);

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 256, 256, 256, 1, indices,
                        histogram, false);

        for (int i = 0; i < 256; i++) {
            assertEquals(1, histogram.get(i));
        }
    }

    @Test
    void directBufferWithMask() {
        Buffer image = allocateDirectImage(4);
        putToDirectImage(image, 0, 1, 2, 3);
        ((Buffer)image).flip();

        ByteBuffer mask = ByteBuffer.allocateDirect(4);
        mask.put(new byte[] {1, 0, 1, 0});
        mask.flip();

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int[] indices = {0};

        invokeHistogram(8, image, mask, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histogram.get(0));
        assertEquals(0, histogram.get(1));
        assertEquals(1, histogram.get(2));
        assertEquals(0, histogram.get(3));
    }

    @Test
    void directBufferMaskWithPosition() {
        Buffer image = allocateDirectImage(4);
        putToDirectImage(image, 0, 1, 2, 3);
        ((Buffer)image).flip();

        ByteBuffer mask = ByteBuffer.allocateDirect(8);
        mask.put(new byte[] {99, 99, 99, 99, 1, 0, 1, 0});
        mask.position(4);

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int[] indices = {0};

        invokeHistogram(8, image, mask, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histogram.get(0));
        assertEquals(0, histogram.get(1));
        assertEquals(1, histogram.get(2));
        assertEquals(0, histogram.get(3));
    }

    @Test
    void exactBoundaryCapacity() {
        // 2 rows, width=2, stride=3: need (2-1)*3+2 = 5 elements
        Buffer exact = allocateDirectImage(5);
        putToDirectImage(exact, 0, 1, 99, 2, 3);
        ((Buffer)exact).flip();

        Buffer tooSmall = allocateDirectImage(4);
        putToDirectImage(tooSmall, 0, 1, 99, 2);
        ((Buffer)tooSmall).flip();

        Buffer tooLarge = allocateDirectImage(6);
        putToDirectImage(tooLarge, 0, 1, 99, 2, 3, 99);
        ((Buffer)tooLarge).flip();

        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        assertDoesNotThrow(()
                               -> invokeHistogram(8, exact, null, 2, 2, 3, 3,
                                                  1, indices, histogram,
                                                  false));

        assertThrows(IllegalArgumentException.class,
                     ()
                         -> invokeHistogram(8, tooSmall, null, 2, 2, 3, 3, 1,
                                            indices, histogram, false));

        assertThrows(IllegalArgumentException.class,
                     ()
                         -> invokeHistogram(8, tooLarge, null, 2, 2, 3, 3, 1,
                                            indices, histogram, false));
    }

    // Mixed buffer tests

    @Test
    void directImageArrayHistogram() {
        Buffer image = allocateDirectImage(4);
        putToDirectImage(image, 0, 1, 2, 3);
        ((Buffer)image).flip();

        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(1, histData[1]);
        assertEquals(1, histData[2]);
        assertEquals(1, histData[3]);
    }

    @Test
    void arrayImageDirectHistogram() {
        Buffer image = wrapImage(0, 1, 2, 3);

        ByteBuffer histBuf =
            ByteBuffer.allocateDirect(256 * 4).order(ByteOrder.nativeOrder());
        IntBuffer histogram = histBuf.asIntBuffer();
        int[] indices = {0};

        invokeHistogram(8, image, null, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histogram.get(0));
        assertEquals(1, histogram.get(1));
        assertEquals(1, histogram.get(2));
        assertEquals(1, histogram.get(3));
    }

    @Test
    void directImageArrayMaskArrayHistogram() {
        Buffer image = allocateDirectImage(4);
        putToDirectImage(image, 0, 1, 2, 3);
        ((Buffer)image).flip();

        byte[] maskData = {1, 0, 1, 0};
        ByteBuffer mask = ByteBuffer.wrap(maskData);

        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, mask, 1, 4, 4, 4, 1, indices, histogram,
                        false);

        assertEquals(1, histData[0]);
        assertEquals(0, histData[1]);
        assertEquals(1, histData[2]);
        assertEquals(0, histData[3]);
    }

    // Behavioral tests

    @Test
    void parallelParameter() {
        int[] vals = new int[1000];
        for (int i = 0; i < 1000; i++) {
            vals[i] = i % 256;
        }
        Buffer image = wrapImage(vals);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image, null, 10, 100, 100, 100, 1, indices,
                        histogram, true);

        int total = 0;
        for (int i = 0; i < 256; i++) {
            total += histData[i];
        }
        assertEquals(1000, total);
    }

    @Test
    void histogramAccumulation() {
        Buffer image1 = wrapImage(0, 1, 2);
        Buffer image2 = wrapImage(0, 0, 3);
        int[] histData = new int[256];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(8, image1, null, 1, 3, 3, 3, 1, indices, histogram,
                        false);
        invokeHistogram(8, image2, null, 1, 3, 3, 3, 1, indices, histogram,
                        false);

        assertEquals(3, histData[0]);
        assertEquals(1, histData[1]);
        assertEquals(1, histData[2]);
        assertEquals(1, histData[3]);
    }

    // Edge case tests

    @Test
    void sampleBitsZero() {
        Buffer image = wrapImage(0, 0, 0, 100, 200);
        int[] histData = new int[1];
        IntBuffer histogram = IntBuffer.wrap(histData);
        int[] indices = {0};

        invokeHistogram(0, image, null, 1, 5, 5, 5, 1, indices, histogram,
                        false);

        assertEquals(3, histData[0]);
    }
}

class Histogram8Tests extends HistogramTestBase {

    @Override
    Buffer wrapImage(int... values) {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte)values[i];
        }
        return ByteBuffer.wrap(bytes);
    }

    @Override
    Buffer wrapImageWithPrefix(int prefixLength, int... values) {
        byte[] bytes = new byte[prefixLength + values.length];
        for (int i = 0; i < prefixLength; i++) {
            bytes[i] = 99;
        }
        for (int i = 0; i < values.length; i++) {
            bytes[prefixLength + i] = (byte)values[i];
        }
        ByteBuffer buf = ByteBuffer.wrap(bytes);
        buf.position(prefixLength);
        return buf;
    }

    @Override
    Buffer allocateDirectImage(int size) {
        return ByteBuffer.allocateDirect(size);
    }

    @Override
    void putToDirectImage(Buffer buf, int... values) {
        ByteBuffer bb = (ByteBuffer)buf;
        for (int v : values) {
            bb.put((byte)v);
        }
    }

    @Override
    int histogramSize() {
        return 256;
    }

    @Override
    int[] unsignedBoundaryValues() {
        return new int[] {127, 128, 255};
    }

    @Override
    void invokeHistogram(int sampleBits, Buffer image, ByteBuffer mask,
                         int rows, int width, int stride, int maskStride,
                         int components, int[] indices, IntBuffer histogram,
                         boolean parallel) {
        IHistNative.histogram8(sampleBits, (ByteBuffer)image, mask, rows,
                               width, stride, maskStride, components, indices,
                               histogram, parallel);
    }
}

class Histogram16Tests extends HistogramTestBase {

    @Override
    Buffer wrapImage(int... values) {
        short[] shorts = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            shorts[i] = (short)values[i];
        }
        return ShortBuffer.wrap(shorts);
    }

    @Override
    Buffer wrapImageWithPrefix(int prefixLength, int... values) {
        short[] shorts = new short[prefixLength + values.length];
        for (int i = 0; i < prefixLength; i++) {
            shorts[i] = 99;
        }
        for (int i = 0; i < values.length; i++) {
            shorts[prefixLength + i] = (short)values[i];
        }
        ShortBuffer buf = ShortBuffer.wrap(shorts);
        buf.position(prefixLength);
        return buf;
    }

    @Override
    Buffer allocateDirectImage(int size) {
        ByteBuffer bb =
            ByteBuffer.allocateDirect(size * 2).order(ByteOrder.nativeOrder());
        return bb.asShortBuffer();
    }

    @Override
    void putToDirectImage(Buffer buf, int... values) {
        ShortBuffer sb = (ShortBuffer)buf;
        for (int v : values) {
            sb.put((short)v);
        }
    }

    @Override
    int histogramSize() {
        return 65536;
    }

    @Override
    int[] unsignedBoundaryValues() {
        return new int[] {32767, 32768, 65535};
    }

    @Override
    void invokeHistogram(int sampleBits, Buffer image, ByteBuffer mask,
                         int rows, int width, int stride, int maskStride,
                         int components, int[] indices, IntBuffer histogram,
                         boolean parallel) {
        IHistNative.histogram16(sampleBits, (ShortBuffer)image, mask, rows,
                                width, stride, maskStride, components, indices,
                                histogram, parallel);
    }
}

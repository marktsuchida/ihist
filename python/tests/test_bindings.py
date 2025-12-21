# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

"""Tests for Python binding layer functionality.

Tests how Python objects are converted to/from C, including:
- Basic smoke tests
- Buffer protocol compatibility
- Memory layout handling
- Return types and ownership
- Dimension interpretation
- Error messages
- Module attributes
"""

import array

import numpy as np

import ihist


class TestBasicSmokeTests:
    """Minimal smoke tests to verify function is callable and returns correct types."""

    def test_simple_grayscale_8bit(self):
        """Test simple grayscale 8-bit histogram."""
        # Basic smoke test: verify function works, returns correct shape/dtype
        image = np.array([0, 1, 1, 2, 2, 2], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist.dtype == np.uint32
        assert hist[0] == 1
        assert hist[1] == 2
        assert hist[2] == 3

    def test_simple_rgb(self):
        """Test simple RGB histogram."""
        # Verify multi-component works
        image = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (3, 256)
        assert hist.dtype == np.uint32

    def test_simple_16bit(self):
        """Test 16-bit histogram."""
        # Verify uint16 works
        image = np.array([0, 100, 1000, 10000, 65535], dtype=np.uint16)
        hist = ihist.histogram(image, bits=16)

        assert hist.shape == (65536,)
        assert hist.dtype == np.uint32


class TestBufferProtocol:
    """Test that various buffer protocol objects work."""

    def test_numpy_array_input(self):
        """Test that NumPy arrays work as input."""
        image = np.array([0, 1, 2], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1

    def test_memoryview_input(self):
        """Test that memoryview works as input."""
        arr = array.array("B", [0, 1, 2])  # 'B' is unsigned char
        image = memoryview(arr).cast("B")

        hist = ihist.histogram(image)

        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1

    def test_numpy_array_mask(self):
        """Test that NumPy arrays work as mask."""
        image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        hist = ihist.histogram(image, mask=mask)

        assert hist[0] == 1
        assert hist[2] == 1
        assert hist[1] == 0
        assert hist[3] == 0

    def test_numpy_array_output(self):
        """Test that NumPy arrays work as output."""
        image = np.array([0, 1], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)

        result = ihist.histogram(image, out=out)

        assert result is out
        assert result[0] == 1
        assert result[1] == 1


class TestReturnTypes:
    """Test return type behavior."""

    def test_returns_numpy_array_when_no_out(self):
        """Test that function returns NumPy array when out is not provided."""
        image = np.array([0, 1], dtype=np.uint8)
        hist = ihist.histogram(image)

        # Check that it's a NumPy array
        assert isinstance(hist, np.ndarray)
        assert hist.dtype == np.uint32

    def test_returns_provided_out(self):
        """Test that function returns the provided out buffer."""
        image = np.array([0, 1], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)

        result = ihist.histogram(image, out=out)

        # Should return the exact same object
        assert result is out

    def test_single_component_returns_1d(self):
        """Test that single component returns 1D array."""
        image = np.array([0], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.ndim == 1
        assert hist.shape == (256,)

    def test_multi_component_returns_2d(self):
        """Test that multiple components return 2D array."""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.ndim == 2
        assert hist.shape == (3, 256)

    def test_single_selected_component_returns_2d(self):
        """Test that selecting single component explicitly returns 2D."""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        hist = ihist.histogram(image, components=[0])

        # Explicit components= always returns 2D for generic code compatibility
        assert hist.ndim == 2
        assert hist.shape == (1, 256)


class TestMemoryLayout:
    """Test memory layout and stride handling."""

    def test_c_contiguous_array(self):
        """Test C-contiguous array (default NumPy layout)."""
        image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        assert image.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image)

        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1
        assert hist[3] == 1

    def test_non_contiguous_accepted(self):
        """Test that non-contiguous array is automatically converted."""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[::2, ::2] = 5
        # Create non-contiguous view
        image_nc = image[::2, ::2]

        # nanobind automatically converts to contiguous
        hist = ihist.histogram(image_nc)
        assert hist[5] == 25
        assert hist.sum() == 25

    def test_fortran_order_accepted(self):
        """Test that Fortran-ordered array is automatically converted."""
        image = np.array([[0, 1], [2, 3]], dtype=np.uint8, order="F")

        # nanobind automatically converts to C-contiguous
        hist = ihist.histogram(image)
        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1
        assert hist[3] == 1
        assert hist.sum() == 4

    def test_contiguous_copy_works(self):
        """Test that both non-contiguous and explicit contiguous copies work."""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[::2, ::2] = 5

        # Create non-contiguous view
        image_nc = image[::2, ::2]

        # Non-contiguous works due to automatic conversion
        hist_nc = ihist.histogram(image_nc)
        assert hist_nc[0] == 0  # No zero values in the view
        assert hist_nc[5] == 25  # 5x5 grid of 5s

        # Explicit contiguous copy also works
        image_c = np.ascontiguousarray(image_nc)
        hist = ihist.histogram(image_c)

        assert hist[0] == 0  # No zero values in the view
        assert hist[5] == 25  # 5x5 grid of 5s


class TestArrayOwnership:
    """Test that arrays are properly owned and not freed prematurely."""

    def test_output_array_persists(self):
        """Test that output array persists after function returns."""
        image = np.array([0, 1, 2], dtype=np.uint8)
        hist = ihist.histogram(image)

        # Access after function return
        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1

    def test_output_array_modifiable(self):
        """Test that output array is modifiable."""
        image = np.array([0, 1], dtype=np.uint8)
        hist = ihist.histogram(image)

        # Modify the array
        hist[0] = 999
        assert hist[0] == 999

    def test_provided_out_modified(self):
        """Test that provided out buffer is properly modified."""
        image = np.array([0, 1], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)
        out[100] = 777  # Marker value

        result = ihist.histogram(image, out=out)

        # Check that it's the same object
        assert result is out
        # Check that it was modified
        assert result[0] == 1
        assert result[1] == 1
        # Check that other values were zeroed (accumulate=False by default)
        assert result[100] == 0


class TestModuleAttributes:
    """Test module-level attributes."""

    def test_module_has_histogram(self):
        """Test that module exports histogram function."""
        assert hasattr(ihist, "histogram")
        assert callable(ihist.histogram)

    def test_module_all(self):
        """Test that module __all__ is defined correctly."""
        assert hasattr(ihist, "__all__")
        assert "histogram" in ihist.__all__

    def test_histogram_has_docstring(self):
        """Test that histogram function has docstring."""
        assert ihist.histogram.__doc__ is not None
        assert len(ihist.histogram.__doc__) > 0
        assert "histogram" in ihist.histogram.__doc__.lower()


class TestDataTypes:
    """Test handling of different data types."""

    def test_uint8_input(self):
        """Test uint8 input."""
        image = np.array([0, 255], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.dtype == np.uint32
        assert hist[0] == 1
        assert hist[255] == 1

    def test_uint16_input(self):
        """Test uint16 input."""
        image = np.array([0, 65535], dtype=np.uint16)
        hist = ihist.histogram(image, bits=16)

        assert hist.dtype == np.uint32
        assert hist[0] == 1
        assert hist[65535] == 1

    def test_uint32_output(self):
        """Test that output is always uint32."""
        image = np.array([0], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.dtype == np.uint32

    def test_explicit_uint32_out(self):
        """Test that explicit uint32 out is accepted."""
        image = np.array([0], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)

        result = ihist.histogram(image, out=out)

        assert result.dtype == np.uint32


class TestComponentsSequenceTypes:
    """Test that components accepts various sequence types."""

    def test_components_list(self):
        """Test components as list."""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        hist = ihist.histogram(image, components=[0, 1])

        assert hist.shape == (2, 256)

    def test_components_tuple(self):
        """Test components as tuple."""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        hist = ihist.histogram(image, components=(0, 1))

        assert hist.shape == (2, 256)

    def test_components_numpy_array(self):
        """Test components as NumPy array."""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        components = np.array([0, 1])
        hist = ihist.histogram(image, components=components)

        assert hist.shape == (2, 256)

    def test_components_range(self):
        """Test components as range."""
        image = np.zeros((5, 5, 4), dtype=np.uint8)
        hist = ihist.histogram(image, components=range(3))

        assert hist.shape == (3, 256)


class TestEmptyImages:
    """Test histogram computation with empty images."""

    def test_empty_1d(self):
        """Test empty 1D array."""
        image = np.array([], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_zero_height(self):
        """Test image with zero height."""
        image = np.zeros((0, 10), dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_zero_width(self):
        """Test image with zero width."""
        image = np.zeros((10, 0), dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_zero_height_width(self):
        """Test image with zero height and width."""
        image = np.zeros((0, 0), dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_empty_3d(self):
        """Test empty 3D array."""
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (3, 256)
        assert hist.sum() == 0


class TestDimensionHandling:
    """Test proper handling of 1D, 2D, 3D arrays."""

    def test_1d_interpreted_as_row(self):
        """Test that 1D array is interpreted as (1, width, 1)."""
        image = np.array([10, 20, 30], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist[10] == 1
        assert hist[20] == 1
        assert hist[30] == 1

    def test_2d_interpreted_as_single_component(self):
        """Test that 2D array is interpreted as (height, width, 1)."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist[10] == 1
        assert hist[20] == 1
        assert hist[30] == 1
        assert hist[40] == 1

    def test_3d_uses_third_dimension(self):
        """Test that 3D array uses third dimension as component."""
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (3, 256)
        assert hist[0, 10] == 1
        assert hist[1, 20] == 1
        assert hist[2, 30] == 1

    def test_single_pixel_1d(self):
        """Test single pixel as 1D array."""
        image = np.array([42], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist[42] == 1
        assert hist.sum() == 1

    def test_single_pixel_2d(self):
        """Test single pixel as 2D array."""
        image = np.array([[42]], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist[42] == 1
        assert hist.sum() == 1

    def test_single_pixel_3d(self):
        """Test single pixel as 3D array."""
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (3, 256)
        assert hist.sum() == 3


class TestSingleComponentImages:
    """Test images with single component."""

    def test_single_component_3d(self):
        """Test 3D image with single component (shape H, W, 1)."""
        image = np.array([[[10], [20]], [[30], [40]]], dtype=np.uint8)
        hist = ihist.histogram(image)

        # 3D image always returns 2D histogram for generic code compatibility
        assert hist.shape == (1, 256)
        assert hist[0, 10] == 1
        assert hist[0, 20] == 1
        assert hist[0, 30] == 1
        assert hist[0, 40] == 1

    def test_select_single_component_from_rgb(self):
        """Test selecting single component from RGB."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :, 1] = 42  # Set green component

        hist = ihist.histogram(image, components=[1])

        # Explicit components= returns 2D histogram
        assert hist.shape == (1, 256)
        assert hist[0, 42] == 100


class TestStridedArrayOptimization:
    """Test that strided arrays are handled correctly without unnecessary copies."""

    def test_row_padded_2d_no_copy(self):
        """Test 2D array with row padding (stride > width) works without copy."""
        # Create padded array: 10 rows, 8 columns, but stride of 16
        base = np.zeros((10, 16), dtype=np.uint8)
        base[:, :8] = np.arange(80, dtype=np.uint8).reshape(10, 8)
        image = base[:, :8]  # View with stride(0) = 16, stride(1) = 1

        assert not image.flags["C_CONTIGUOUS"]
        assert image.strides == (16, 1)

        hist = ihist.histogram(image)
        # Verify correct values
        assert hist.sum() == 80

    def test_row_padded_3d_no_copy(self):
        """Test 3D array with row padding works without copy."""
        # Create padded RGB array
        base = np.zeros((10, 16, 3), dtype=np.uint8)
        base[:, :8, :] = np.arange(240, dtype=np.uint8).reshape(10, 8, 3)
        image = base[:, :8, :]  # stride(0) = 48, stride(1) = 3, stride(2) = 1

        assert not image.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)
        # Each component has 80 pixels, sum should be 80 per component
        assert hist[0].sum() == 80
        assert hist[1].sum() == 80
        assert hist[2].sum() == 80

    def test_non_contiguous_columns_requires_copy(self):
        """Test that non-contiguous column stride falls back to copy."""
        # Select every other column
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, ::2] = 42
        view = image[:, ::2]  # stride(1) = 2, not 1

        assert not view.flags["C_CONTIGUOUS"]

        # Should still work (via copy)
        hist = ihist.histogram(view)
        assert hist[42] == 50

    def test_transposed_array_requires_copy(self):
        """Test that transposed (Fortran order) array falls back to copy."""
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8, order="F")

        # Should work via copy
        hist = ihist.histogram(image)
        assert hist[1] == 1
        assert hist[2] == 1
        assert hist[3] == 1
        assert hist[4] == 1

    def test_mask_row_padded_no_copy(self):
        """Test mask with row padding works without copy."""
        image = np.zeros((10, 8), dtype=np.uint8)
        image[0, 0] = 100

        # Create padded mask
        base_mask = np.zeros((10, 16), dtype=np.uint8)
        base_mask[:, :8] = 1
        mask = base_mask[:, :8]

        assert not mask.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image, mask=mask)
        assert hist[100] == 1
        assert hist[0] == 79

    def test_mask_non_contiguous_requires_copy(self):
        """Test mask with non-contiguous columns falls back to copy."""
        image = np.zeros((10, 5), dtype=np.uint8)
        image[0, 0] = 50

        # Mask with stride(1) = 2
        base_mask = np.ones((10, 10), dtype=np.uint8)
        mask = base_mask[:, ::2]

        hist = ihist.histogram(image, mask=mask)
        assert hist[50] == 1

    def test_16bit_row_padded(self):
        """Test 16-bit array with row padding."""
        base = np.zeros((10, 16), dtype=np.uint16)
        base[:, :8] = 1000
        image = base[:, :8]

        hist = ihist.histogram(image, bits=12)
        assert hist[1000] == 80

    def test_empty_strided_array(self):
        """Test empty strided array (edge case)."""
        base = np.zeros((10, 16), dtype=np.uint8)
        image = base[:0, :8]  # Empty but strided

        hist = ihist.histogram(image)
        assert hist.sum() == 0

    def test_single_row_strided(self):
        """Test 1D-like view from 2D array."""
        base = np.zeros((10, 10), dtype=np.uint8)
        base[0, :] = np.arange(10, dtype=np.uint8)
        image = base[0, :]  # 1D view

        hist = ihist.histogram(image)
        for i in range(10):
            assert hist[i] == 1

    def test_both_image_and_mask_row_padded(self):
        """Test both image and mask with row padding simultaneously."""
        # Create row-padded image
        base_img = np.zeros((5, 16), dtype=np.uint8)
        base_img[:, :8] = 42
        image = base_img[:, :8]

        # Create row-padded mask with different padding
        base_mask = np.zeros((5, 32), dtype=np.uint8)
        base_mask[:3, :8] = 1  # Mask only first 3 rows
        mask = base_mask[:, :8]

        assert not image.flags["C_CONTIGUOUS"]
        assert not mask.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image, mask=mask)
        assert hist[42] == 24  # 3 rows * 8 cols = 24 masked pixels
        assert hist.sum() == 24


class TestFOrderOptimization:
    """Test F-contiguous (Fortran order) array handling."""

    def test_f_contiguous_2d_no_copy(self):
        """Test 2D F-contiguous array works without copy."""
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8, order="F")
        assert image.flags["F_CONTIGUOUS"]
        assert not image.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image)
        # Should have one count for each value 1-6
        for i in range(1, 7):
            assert hist[i] == 1
        assert hist.sum() == 6

    def test_f_contiguous_2d_column_padded(self):
        """Test 2D F-order-like array with column padding (stride(1) > height)."""
        # Create F-order array with padding
        base = np.zeros((16, 10), dtype=np.uint8, order="F")
        base[:8, :] = np.arange(80, dtype=np.uint8).reshape(8, 10, order="F")
        image = base[:8, :]  # View with column padding

        # Numpy doesn't consider this F_CONTIGUOUS due to the slice,
        # but it has F-order compatible strides: stride(0)=1, stride(1)=16
        assert not image.flags["C_CONTIGUOUS"]
        assert not image.flags["F_CONTIGUOUS"]
        # Strides: (1, 16) - column stride (16) > height (8), but stride(0) == 1
        assert image.strides == (1, 16)

        hist = ihist.histogram(image)
        assert hist.sum() == 80

    def test_f_contiguous_2d_16bit(self):
        """Test 16-bit F-contiguous array."""
        image = np.array([[100, 200], [300, 400]], dtype=np.uint16, order="F")
        assert image.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image, bits=12)
        assert hist[100] == 1
        assert hist[200] == 1
        assert hist[300] == 1
        assert hist[400] == 1

    def test_f_contiguous_3d_requires_copy(self):
        """Test 3D F-contiguous array falls back to copy (components not contiguous)."""
        image = np.zeros((4, 5, 3), dtype=np.uint8, order="F")
        image[:, :, 0] = 10  # R
        image[:, :, 1] = 20  # G
        image[:, :, 2] = 30  # B
        assert image.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)
        assert hist[0, 10] == 20  # 4*5 = 20 pixels with R=10
        assert hist[1, 20] == 20
        assert hist[2, 30] == 20

    def test_f_image_f_mask_no_copy(self):
        """Test F-contiguous image and F-contiguous mask (no copies needed)."""
        image = np.zeros((10, 8), dtype=np.uint8, order="F")
        image[0, 0] = 100
        assert image.flags["F_CONTIGUOUS"]

        mask = np.zeros((10, 8), dtype=np.uint8, order="F")
        mask[:5, :] = 1  # Mask first 5 rows
        assert mask.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image, mask=mask)
        assert hist[100] == 1  # The 100 is at (0,0), which is masked
        assert hist[0] == 39  # 5*8 - 1 = 39 zeros
        assert hist.sum() == 40  # 5*8 = 40 masked pixels

    def test_f_image_c_mask_copies_mask(self):
        """Test F-contiguous image with C-contiguous mask (mask copied)."""
        image = np.zeros((10, 8), dtype=np.uint8, order="F")
        image[0, 0] = 50
        image[5, 5] = 60
        assert image.flags["F_CONTIGUOUS"]

        mask = np.ones((10, 8), dtype=np.uint8, order="C")
        mask[5:, :] = 0  # Mask only first 5 rows
        assert mask.flags["C_CONTIGUOUS"]

        hist = ihist.histogram(image, mask=mask)
        assert hist[50] == 1  # (0,0) is in masked region
        assert hist[60] == 0  # (5,5) is NOT in masked region
        assert hist.sum() == 40  # 5*8 = 40 masked pixels

    def test_c_image_f_mask_copies_mask(self):
        """Test C-contiguous image with F-contiguous mask (mask copied)."""
        image = np.zeros((10, 8), dtype=np.uint8, order="C")
        image[0, 0] = 70
        image[5, 5] = 80
        assert image.flags["C_CONTIGUOUS"]

        mask = np.ones((10, 8), dtype=np.uint8, order="F")
        mask[5:, :] = 0  # Mask only first 5 rows
        assert mask.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image, mask=mask)
        assert hist[70] == 1  # (0,0) is in masked region
        assert hist[80] == 0  # (5,5) is NOT in masked region
        assert hist.sum() == 40  # 5*8 = 40 masked pixels

    def test_f_order_larger_image(self):
        """Test F-order with a larger image to ensure correct pixel iteration."""
        # Create a known pattern in F-order
        height, width = 100, 80
        image = np.zeros((height, width), dtype=np.uint8, order="F")

        # Fill with a pattern: each column has a unique value
        for j in range(width):
            image[:, j] = j % 256

        assert image.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image)
        # Each column value appears 'height' times
        for j in range(width):
            expected = height  # 100 pixels per column
            assert hist[j % 256] == expected, f"Mismatch at bin {j % 256}"

    def test_f_order_with_components_selection(self):
        """Test F-order 3D with component selection still works."""
        image = np.zeros((4, 5, 4), dtype=np.uint8, order="F")
        image[:, :, 0] = 10  # R
        image[:, :, 1] = 20  # G
        image[:, :, 2] = 30  # B
        image[:, :, 3] = 40  # A

        # Select only R and B channels
        hist = ihist.histogram(image, components=[0, 2])
        assert hist.shape == (2, 256)
        assert hist[0, 10] == 20  # R channel
        assert hist[1, 30] == 20  # B channel

    def test_3d_transposed_axes_01_no_copy(self):
        """Test 3D with axes 0,1 swapped but components still contiguous."""
        # Create (W, H, C) C-order, then transpose to (H, W, C).
        # This gives stride(0) == C (pixels contiguous along axis 0),
        # stride(1) = H*C (row stride), stride(2) = 1 (components contiguous).
        h, w, c = 4, 5, 3
        base = np.zeros((w, h, c), dtype=np.uint8, order="C")
        base[:, :, 0] = 10  # R
        base[:, :, 1] = 20  # G
        base[:, :, 2] = 30  # B
        image = base.transpose(1, 0, 2)  # Now shape (H, W, C)

        assert image.shape == (h, w, c)
        # Verify strides: (C, H*C, 1) = (3, 12, 1)
        assert image.strides == (c, h * c, 1)
        assert not image.flags["C_CONTIGUOUS"]
        assert not image.flags["F_CONTIGUOUS"]

        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)
        assert hist[0, 10] == h * w  # All pixels have R=10
        assert hist[1, 20] == h * w
        assert hist[2, 30] == h * w

    def test_3d_transposed_16bit(self):
        """Test 16-bit 3D with transposed axes 0,1."""
        h, w, c = 6, 8, 2
        base = np.zeros((w, h, c), dtype=np.uint16, order="C")
        base[:, :, 0] = 100
        base[:, :, 1] = 200
        image = base.transpose(1, 0, 2)

        assert image.strides == (c * 2, h * c * 2, 2)  # elem_size = 2

        hist = ihist.histogram(image, bits=12)
        assert hist.shape == (2, 4096)
        assert hist[0, 100] == h * w
        assert hist[1, 200] == h * w

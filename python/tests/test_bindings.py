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

# This file is part of ihist
# Copyright 2025 Board of Regents of the University of Wisconsin System
# SPDX-License-Identifier: MIT

"""Parameter validation tests for histogram function.

Tests that all parameters are properly validated by the bindings:
- Image (dtype, dimensions)
- Bits (range, defaults)
- Mask (dtype, shape, dimensions)
- Components (range, empty, defaults)
- Out (dtype, shape, dimensions)
- Accumulate (buffer zeroing behavior)
"""

import numpy as np
import pytest

import ihist


class TestImageParameterValidation:
    """Test validation of image parameter."""

    def test_invalid_dtype(self):
        """Test that non-uint8/uint16 dtype raises error."""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            image = np.array([0, 1, 2], dtype=dtype)
            with pytest.raises(
                ValueError, match="must have dtype uint8 or uint16"
            ):
                ihist.histogram(image)

    def test_4d_array_raises_error(self):
        """Test that 4D array raises error."""
        image = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 1D, 2D, or 3D"):
            ihist.histogram(image)

    def test_0d_array_raises_error(self):
        """Test that 0D array raises error."""
        image = np.array(42, dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 1D, 2D, or 3D"):
            ihist.histogram(image)


class TestBitsParameterValidation:
    """Test validation of bits parameter."""

    def test_bits_default_uint8(self):
        """Test that bits defaults to 8 for uint8."""
        image = np.array([0, 255], dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (256,)

    def test_bits_default_uint16(self):
        """Test that bits defaults to 16 for uint16."""
        image = np.array([0, 65535], dtype=np.uint16)
        hist = ihist.histogram(image)
        assert hist.shape == (65536,)

    def test_bits_explicit_8(self):
        """Test explicit bits=8 for uint8."""
        image = np.array([0, 255], dtype=np.uint8)
        hist = ihist.histogram(image, bits=8)
        assert hist.shape == (256,)

    def test_bits_reduced_uint8(self):
        """Test reduced bits for uint8."""
        image = np.array([0, 1, 2, 3], dtype=np.uint8)
        hist = ihist.histogram(image, bits=2)
        assert hist.shape == (4,)

    def test_bits_invalid_too_low(self):
        """Test that bits < 1 raises error."""
        image = np.array([0], dtype=np.uint8)
        with pytest.raises(ValueError, match="bits must be in range"):
            ihist.histogram(image, bits=0)

    def test_bits_invalid_too_high_uint8(self):
        """Test that bits > 8 raises error for uint8."""
        image = np.array([0], dtype=np.uint8)
        with pytest.raises(ValueError, match="bits must be in range"):
            ihist.histogram(image, bits=9)

    def test_bits_invalid_too_high_uint16(self):
        """Test that bits > 16 raises error for uint16."""
        image = np.array([0], dtype=np.uint16)
        with pytest.raises(ValueError, match="bits must be in range"):
            ihist.histogram(image, bits=17)


class TestMaskParameterValidation:
    """Test validation of mask parameter."""

    def test_mask_wrong_dtype(self):
        """Test that non-uint8 mask raises error."""
        image = np.array([[0, 1]], dtype=np.uint8)
        mask = np.array([[1, 0]], dtype=np.uint16)
        with pytest.raises(ValueError, match="Mask must have dtype uint8"):
            ihist.histogram(image, mask=mask)

    def test_mask_wrong_ndim(self):
        """Test that non-2D mask raises error."""
        image = np.array([[0, 1]], dtype=np.uint8)
        mask = np.array([1, 0], dtype=np.uint8)  # 1D
        with pytest.raises(ValueError, match="Mask must be 2D"):
            ihist.histogram(image, mask=mask)

    def test_mask_wrong_shape(self):
        """Test that mismatched mask shape raises error."""
        image = np.array([[0, 1]], dtype=np.uint8)  # Shape (1, 2)
        mask = np.array([[1, 0, 0]], dtype=np.uint8)  # Shape (1, 3)
        with pytest.raises(ValueError, match="does not match image shape"):
            ihist.histogram(image, mask=mask)


class TestComponentsParameterValidation:
    """Test validation of components parameter."""

    def test_components_default(self):
        """Test that components defaults to all components."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)

    def test_components_subset(self):
        """Test selecting subset of components."""
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        hist = ihist.histogram(image, components=[0, 2])
        assert hist.shape == (2, 256)

    def test_components_single(self):
        """Test selecting single component."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        hist = ihist.histogram(image, components=[1])
        assert hist.shape == (1, 256)  # Explicit components= returns 2D

    def test_components_out_of_range(self):
        """Test that out-of-range component raises error."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(
            ValueError, match="Component index .* out of range"
        ):
            ihist.histogram(image, components=[3])

    def test_components_empty(self):
        """Test that empty components produces empty output."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        hist = ihist.histogram(image, components=[])
        assert hist.shape == (0, 256)
        assert hist.dtype == np.uint32

    def test_components_repeated(self):
        """Test that repeated component indices are allowed."""
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0, 0] = 10  # R
        image[0, 0, 1] = 20  # G

        hist = ihist.histogram(image, components=[0, 0, 1])

        assert hist.shape == (3, 256)
        assert hist[0, 10] == 1  # First R
        assert hist[1, 10] == 1  # Second R (same as first)
        assert hist[2, 20] == 1  # G


class TestOutParameterValidation:
    """Test validation of out parameter."""

    def test_out_wrong_dtype(self):
        """Test that non-uint32 out raises error."""
        image = np.array([0], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint64)
        with pytest.raises(ValueError, match="Output must have dtype uint32"):
            ihist.histogram(image, out=out)

    def test_out_wrong_shape_1d(self):
        """Test that wrong 1D shape raises error."""
        image = np.array([0], dtype=np.uint8)
        out = np.zeros(128, dtype=np.uint32)  # Should be 256
        with pytest.raises(ValueError, match="does not match expected"):
            ihist.histogram(image, out=out)

    def test_out_wrong_shape_2d(self):
        """Test that wrong 2D shape raises error."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        out = np.zeros((3, 128), dtype=np.uint32)  # Should be (3, 256)
        with pytest.raises(ValueError, match="does not match expected"):
            ihist.histogram(image, out=out)

    def test_out_wrong_ndim(self):
        """Test that 3D out raises error."""
        image = np.array([0], dtype=np.uint8)
        out = np.zeros((1, 1, 256), dtype=np.uint32)
        with pytest.raises(ValueError, match="Output must be 1D or 2D"):
            ihist.histogram(image, out=out)

    def test_out_1d_shape(self):
        """Test that 1D out with correct shape works."""
        image = np.array([0, 1], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)
        result = ihist.histogram(image, out=out)

        assert result is out
        assert result[0] == 1
        assert result[1] == 1

    def test_out_2d_shape(self):
        """Test that 2D out with correct shape works."""
        image = np.array([[[0, 1]]], dtype=np.uint8)
        out = np.zeros((2, 256), dtype=np.uint32)
        result = ihist.histogram(image, out=out)

        assert result is out
        assert result[0, 0] == 1
        assert result[1, 1] == 1

    def test_out_1d_rejected_for_multicomponent(self):
        """Test that 1D out is rejected for multi-component histogram."""
        image = np.array([[[0, 1]]], dtype=np.uint8)
        out = np.zeros(2 * 256, dtype=np.uint32)
        with pytest.raises(
            ValueError, match="Output must be 2D for multi-component"
        ):
            ihist.histogram(image, out=out)

    def test_out_overlaps_image_rejected(self):
        """Test that out overlapping with image is rejected."""
        # Create a buffer large enough for histogram (256 uint32 = 1024 bytes)
        # and view part of it as a small image
        buf = np.zeros(256, dtype=np.uint32)
        image = buf.view(np.uint8)[:100].reshape(10, 10)
        with pytest.raises(ValueError, match="overlaps with input"):
            ihist.histogram(image, out=buf)

    def test_out_overlaps_mask_rejected(self):
        """Test that out overlapping with mask is rejected."""
        image = np.zeros((10, 10), dtype=np.uint8)
        # Create histogram buffer and view part of it as mask
        buf = np.zeros(256, dtype=np.uint32)
        mask = buf.view(np.uint8)[:100].reshape(10, 10)
        with pytest.raises(ValueError, match="overlaps with mask"):
            ihist.histogram(image, mask=mask, out=buf)


class TestParameterCombinations:
    """Test various parameter combinations."""

    def test_all_parameters_specified(self):
        """Test with all parameters explicitly specified."""
        image = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
        mask = np.ones((1, 1), dtype=np.uint8)
        out = np.zeros((3, 256), dtype=np.uint32)

        result = ihist.histogram(
            image,
            bits=8,
            mask=mask,
            components=[0, 1, 2],
            out=out,
            accumulate=False,
            parallel=True,
        )

        assert result is out
        assert result[0, 10] == 1
        assert result[1, 20] == 1
        assert result[2, 30] == 1

    def test_minimal_parameters(self):
        """Test with only required parameter."""
        image = np.array([0, 1, 2], dtype=np.uint8)
        hist = ihist.histogram(image)

        assert hist.shape == (256,)
        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 1

    def test_16bit_with_custom_bits_and_components(self):
        """Test 16-bit with custom bits and component selection."""
        image = np.zeros((10, 10, 4), dtype=np.uint16)
        image[:, :, 0] = 100
        image[:, :, 1] = 200
        image[:, :, 2] = 300
        image[:, :, 3] = 400

        hist = ihist.histogram(image, bits=12, components=[1, 2])

        assert hist.shape == (2, 4096)
        assert hist[0, 200] == 100
        assert hist[1, 300] == 100


class TestAccumulateParameter:
    """Test accumulate parameter behavior.

    The bindings only do one thing: zero the buffer or not (std::fill).
    We test this minimal binding logic, not C implementation details.
    """

    def test_accumulate_false_zeros_buffer(self):
        """Test that accumulate=False (default) zeros the output buffer."""
        image = np.array([1, 2], dtype=np.uint8)
        out = np.ones(256, dtype=np.uint32) * 999  # Pre-fill with junk

        result = ihist.histogram(image, out=out, accumulate=False)

        # Buffer should be zeroed before histogramming
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 0

    def test_accumulate_true_preserves_values(self):
        """Test that accumulate=True preserves existing buffer values."""
        image = np.array([1, 2], dtype=np.uint8)
        out = np.zeros(256, dtype=np.uint32)
        out[1] = 10
        out[2] = 20

        result = ihist.histogram(image, out=out, accumulate=True)

        # Values should be added to existing
        assert result[1] == 11  # 10 + 1
        assert result[2] == 21  # 20 + 1

    def test_accumulate_with_multicomponent(self):
        """Test accumulation with multi-component (2D output)."""
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        out = np.ones((3, 256), dtype=np.uint32) * 999

        result = ihist.histogram(image, out=out, accumulate=False)

        # Buffer should be zeroed
        assert result[0, 10] == 1
        assert result[0, 0] == 0
        assert result[0, 11] == 0

    def test_accumulate_ignored_without_out(self):
        """Test that accumulate is ignored when out is not provided."""
        # accumulate has no effect since a new buffer is created each time
        image = np.array([1, 2], dtype=np.uint8)

        hist1 = ihist.histogram(image, accumulate=False)
        hist2 = ihist.histogram(image, accumulate=True)

        # Both should give the same result
        np.testing.assert_array_equal(hist1, hist2)


class TestEmptyInputs:
    """Test handling of empty inputs (zero-sized dimensions)."""

    def test_zero_height_image(self):
        """Test histogram of image with zero height."""
        image = np.zeros((0, 10), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_zero_width_image(self):
        """Test histogram of image with zero width."""
        image = np.zeros((10, 0), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_zero_component_image(self):
        """Test histogram of 3D image with zero components."""
        image = np.zeros((10, 10, 0), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (0, 256)
        assert hist.dtype == np.uint32

    def test_zero_height_with_components(self):
        """Test histogram of zero-height RGB image."""
        image = np.zeros((0, 10, 3), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)
        assert hist.sum() == 0

    def test_zero_width_with_components(self):
        """Test histogram of zero-width RGB image."""
        image = np.zeros((10, 0, 3), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (3, 256)
        assert hist.sum() == 0

    def test_empty_1d_image(self):
        """Test histogram of empty 1D image."""
        image = np.zeros((0,), dtype=np.uint8)
        hist = ihist.histogram(image)
        assert hist.shape == (256,)
        assert hist.sum() == 0

    def test_empty_components_with_out(self):
        """Test empty components with pre-allocated output."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        out = np.zeros((0, 256), dtype=np.uint32)
        result = ihist.histogram(image, components=[], out=out)
        assert result is out
        assert result.shape == (0, 256)

    def test_zero_height_16bit(self):
        """Test empty histogram for 16-bit image."""
        image = np.zeros((0, 10), dtype=np.uint16)
        hist = ihist.histogram(image, bits=12)
        assert hist.shape == (4096,)
        assert hist.sum() == 0

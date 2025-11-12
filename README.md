# ihist

<!--
This file is part of ihist
Copyright 2025 Board of Regents of the University of Wisconsin System
SPDX-License-Identifier: MIT
-->

Fast histogram computation for image data.

Currently experimental and API may change.

## C API

The C API provides two functions for computing histograms of 2D image data:

- `ihist_hist8_2d()` - for 8-bit samples (`uint8_t`)
- `ihist_hist16_2d()` - for 16-bit samples (`uint16_t`)

Both functions have identical behavior except for the sample data type.

### Function Signatures

```c
#include <ihist/ihist.h>

void ihist_hist8_2d(
    size_t sample_bits,
    uint8_t const *restrict image,
    uint8_t const *restrict mask,
    size_t height,
    size_t width,
    size_t stride,
    size_t samples_per_pixel,
    size_t n_histogram_samples,
    size_t const *restrict sample_indices,
    uint32_t *restrict histogram,
    bool maybe_parallel);

void ihist_hist16_2d(
    size_t sample_bits,
    uint16_t const *restrict image,
    uint8_t const *restrict mask,
    size_t height,
    size_t width,
    size_t stride,
    size_t samples_per_pixel,
    size_t n_histogram_samples,
    size_t const *restrict sample_indices,
    uint32_t *restrict histogram,
    bool maybe_parallel);
```

### Overview

These functions compute histograms for one or more samples (components) from
image data. They support:

- Multi-channel images (e.g., grayscale, RGB, RGBA)
- Selective histogramming of specific components
- Optional per-pixel masking
- Region of interest (ROI) processing via stride and pointer offset
- Automatic parallelization for large images
- Arbitrary bit depths (not just full 8 or 16 bits)

### Parameters

**`sample_bits`**
Number of significant bits per sample. Valid range: 1-8 for `ihist_hist8_2d()`,
1-16 for `ihist_hist16_2d()`. The histogram will contain 2^`sample_bits` bins
per sample.

Values with bits set beyond `sample_bits` are discarded and not counted in any
bin.

**`image`**
Pointer to image data. Samples are interleaved in row-major order:

- Row 0, pixel 0: all samples
- Row 0, pixel 1: all samples
- ...
- Row 1, pixel 0: all samples
- ...

May be `NULL` if `height` or `width` is 0.

**`mask`** *(optional)*
Per-pixel mask for selective histogramming. If non-`NULL`, must point to
`height * stride` `uint8_t` values. Only pixels where the corresponding mask
value is non-zero are included in the histogram.

Pass `NULL` to histogram all pixels.

**`height`**
Image height in pixels. May be 0 for empty input.

**`width`**
Image width in pixels. May be 0 for empty input.

**`stride`**
Row stride in pixels (not bytes). Must be ≥ `width`.

When `stride` equals `width`, the image is treated as contiguous. Use `stride` >
`width` together with an offset `image` pointer to process a rectangular region
of interest (ROI) within a larger image.

**`samples_per_pixel`**
Number of interleaved samples per pixel. Examples:

- 1 for grayscale
- 3 for RGB
- 4 for RGBA

Must be > 0.

**`n_histogram_samples`**
Number of samples to histogram. Must be > 0.

This allows histogramming a subset of components, such as skipping the alpha
component in RGBA images.

**`sample_indices`**
Array of `n_histogram_samples` indices specifying which samples to histogram.
Each index must be in the range [0, `samples_per_pixel`).

Examples:

- `{0}` - histogram only the first component (e.g., red in RGB)
- `{0, 1, 2}` - histogram first three components (e.g., RGB in RGBA, skipping
  alpha)
- `{1, 2, 3}` - histogram last three components (e.g., skip first component in
  ARGB)

Must not be `NULL`.

**`histogram`** *(output, accumulated)*
Output buffer for histogram data. Must point to `n_histogram_samples *
2^sample_bits` `uint32_t` values.

Histograms for each sample are stored consecutively:

- Bins for `sample_indices[0]`: `histogram[0]` to `histogram[2^sample_bits - 1]`
- Bins for `sample_indices[1]`: `histogram[2^sample_bits]` to `histogram[2 *
  2^sample_bits - 1]`
- ...

**Important:** The histogram is **accumulated** into this buffer. Existing
values are added to, not replaced. To obtain a fresh histogram, zero-initialize
the buffer before calling the function.

**`maybe_parallel`**
Controls parallelization.

- `true` - Allows automatic multi-threaded execution for large images, if ihist
  was built with parallelization support (TBB).
- `false` - Guarantees single-threaded execution.

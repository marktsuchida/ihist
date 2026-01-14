# ihist

<!--
This file is part of ihist
Copyright 2025 Board of Regents of the University of Wisconsin System
SPDX-License-Identifier: MIT
-->

Fast histogram computation for image data with APIs in Python, Java, and C.

Currently in early development and API may change.

Only 64-bit platforms are supported (let us know in an issue if you have a use
case requiring 32-bit support).

Jump to: [Python API](#python-api), [Java API](#java-api), [C API](#c-api),
[Performance notes](#performance).

## Python API

### Python Quick Start

```sh
pip install ihist
```

```python
import numpy as np
import ihist

# Grayscale image
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
hist = ihist.histogram(image)  # Shape: (256,)

# RGB image
rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
hist = ihist.histogram(rgb)  # Shape: (3, 256)

# With masking
mask = np.ones((100, 100), dtype=np.uint8)
hist = ihist.histogram(image, mask=mask)
```

### Python Function

The Python API provides a single function for computing histograms:

```python
import ihist

histogram = ihist.histogram(image, bits=None, mask=None,
                            components=None, out=None,
                            accumulate=False, parallel=True)
```

### Python Parameters

**`image`** : *array_like*
Input image data. Must be uint8 or uint16, and 1D, 2D, or 3D.

- 1D arrays `(W,)` are interpreted as `(1, W, 1)`
- 2D arrays `(H, W)` are interpreted as `(H, W, 1)`
- 3D arrays `(H, W, C)` use C as number of components

Total pixel count must not exceed `2^32-1`.

**`bits`** : *int, optional*
Number of significant bits per sample. If not specified, defaults to full depth
(8 for uint8, 16 for uint16). Valid range: `[0, 8]` for uint8, `[0, 16]` for
uint16.

**`mask`** : *array_like, optional*
Per-pixel mask. Must be uint8. Shape must match image dimensions:

- For 1D images `(W,)`: mask must be 1D, shape `(W,)`
- For 2D/3D images: mask must be 2D, shape `(H, W)`

Only pixels with non-zero mask values are included. If not specified, all
pixels are included.

**`components`** : *sequence of int, optional*
Indices of components to histogram. If not specified, all components are
histogrammed. Each index must be in range `[0, n_components)`.

Examples:

- `[0]` - histogram only the first component (e.g., red in RGB)
- `[0, 1, 2]` - histogram first three components (e.g., RGB in RGBA, skipping
  alpha)

**`out`** : *array_like, optional*
Pre-allocated output buffer. Must be uint32, and either:

- 1D with shape `(2^bits,)` for single-component histogram, or
- 2D with shape `(n_hist_components, 2^bits)`

If not specified, a new array is allocated and returned.

**`accumulate`** : *bool, optional*
If `False` (default), the output buffer is zeroed before computing the
histogram. If `True`, histogram values are accumulated into the existing buffer
values. No effect if `out` is not given.

**`parallel`** : *bool, optional*
If `True` (default), allows automatic multi-threaded execution for large images.
If `False`, guarantees single-threaded execution.

### Python Return Value

**histogram** : *ndarray*
Histogram(s) as uint32 array.

- If the image is 1D or 2D and `components` is not specified, returns 1D array
  of shape `(2^bits,)`
- If the image is 3D or `components` is explicitly specified, returns 2D array
  of shape `(n_hist_components, 2^bits)`
- If `out` was provided, returns the same array after filling

## Java API

### Java Installation

The Java bindings are available on [Maven
Central](https://central.sonatype.com/artifact/io.github.marktsuchida/ihist).
You need both the main JAR and a platform-specific natives JAR containing the
JNI library. The native library is automatically extracted and loaded at
runtime.

Available platforms: Linux x86_64, macOS arm64, macOS x86_64, Windows x86_64.

<details>
<summary>Maven</summary>

```xml
<properties>
    <ihist.version>0.1.2</ihist.version>
</properties>

<dependencies>
    <dependency>
        <groupId>io.github.marktsuchida</groupId>
        <artifactId>ihist</artifactId>
        <version>${ihist.version}</version>
    </dependency>
    <dependency>
        <groupId>io.github.marktsuchida</groupId>
        <artifactId>ihist</artifactId>
        <version>${ihist.version}</version>
        <classifier>${ihist.natives.classifier}</classifier>
    </dependency>
</dependencies>

<profiles>
    <profile>
        <id>natives-linux-x86_64</id>
        <activation>
            <os><name>Linux</name><arch>amd64</arch></os>
        </activation>
        <properties>
            <ihist.natives.classifier>natives-linux-x86_64</ihist.natives.classifier>
        </properties>
    </profile>
    <profile>
        <id>natives-macos-arm64</id>
        <activation>
            <os><family>mac</family><arch>aarch64</arch></os>
        </activation>
        <properties>
            <ihist.natives.classifier>natives-macos-arm64</ihist.natives.classifier>
        </properties>
    </profile>
    <profile>
        <id>natives-macos-x86_64</id>
        <activation>
            <os><family>mac</family><arch>x86_64</arch></os>
        </activation>
        <properties>
            <ihist.natives.classifier>natives-macos-x86_64</ihist.natives.classifier>
        </properties>
    </profile>
    <profile>
        <id>natives-windows-x86_64</id>
        <activation>
            <os><family>windows</family><arch>amd64</arch></os>
        </activation>
        <properties>
            <ihist.natives.classifier>natives-windows-x86_64</ihist.natives.classifier>
        </properties>
    </profile>
</profiles>
```

</details>

<details>
<summary>Gradle (Kotlin DSL)</summary>

```kotlin
plugins {
    id("com.google.osdetector") version "1.7.3"
}

val ihistVersion = "0.1.2"
val ihistNativesClassifier = when {
    osdetector.os == "linux" && osdetector.arch == "x86_64" -> "natives-linux-x86_64"
    osdetector.os == "osx" && osdetector.arch == "aarch_64" -> "natives-macos-arm64"
    osdetector.os == "osx" && osdetector.arch == "x86_64" -> "natives-macos-x86_64"
    osdetector.os == "windows" && osdetector.arch == "x86_64" -> "natives-windows-x86_64"
    else -> throw GradleException("Unsupported platform: ${osdetector.os}-${osdetector.arch}")
}

dependencies {
    implementation("io.github.marktsuchida:ihist:$ihistVersion")
    runtimeOnly("io.github.marktsuchida:ihist:$ihistVersion:$ihistNativesClassifier")
}
```

</details>

<details>
<summary>Gradle (Groovy DSL)</summary>

```groovy
plugins {
    id 'com.google.osdetector' version '1.7.3'
}

def ihistVersion = '0.1.2'
def ihistNativesClassifier
switch ("${osdetector.os}-${osdetector.arch}") {
    case 'linux-x86_64': ihistNativesClassifier = 'natives-linux-x86_64'; break
    case 'osx-aarch_64': ihistNativesClassifier = 'natives-macos-arm64'; break
    case 'osx-x86_64': ihistNativesClassifier = 'natives-macos-x86_64'; break
    case 'windows-x86_64': ihistNativesClassifier = 'natives-windows-x86_64'; break
    default:
        throw new GradleException(
            "Unsupported platform: ${osdetector.os}-${osdetector.arch}")
}

dependencies {
    implementation "io.github.marktsuchida:ihist:$ihistVersion"
    runtimeOnly "io.github.marktsuchida:ihist:$ihistVersion:$ihistNativesClassifier"
}
```

</details>

### Java Version Compatibility

The library targets Java 8 and works with all later versions:

- **Java 8**: Use as a regular classpath dependency
- **Java 9+**: Can be used on the classpath or as an automatic module with
  name `io.github.marktsuchida.ihist`

### Java Quick Start

```java
import io.github.marktsuchida.ihist.HistogramRequest;
import java.nio.IntBuffer;

// Grayscale image
byte[] image = new byte[100 * 100];
IntBuffer hist = HistogramRequest.forImage(image, 100, 100).compute(); // 256 bins
// hist.remaining() == 256, access with hist.get(i)

// RGB image
byte[] rgb = new byte[100 * 100 * 3];
IntBuffer hist = HistogramRequest.forImage(rgb, 100, 100, 3).compute(); // 3 * 256 bins

// With advanced options
IntBuffer hist = HistogramRequest.forImage(image, 100, 100)
    .roi(10, 10, 80, 80)       // Region of interest
    .mask(maskData, 80, 80)    // Per-pixel mask
    .bits(8)                   // Significant bits
    .parallel(true)            // Allow multi-threading
    .compute();
```

### Java Classes

**`HistogramRequest`** - Builder-style interface:

```java
// Multi-component image (e.g., RGB)
IntBuffer hist = HistogramRequest.forImage(image, width, height, 3)
    .selectComponents(0, 1, 2)      // Which components to histogram
    .roi(x, y, roiWidth, roiHeight) // Region of interest
    .mask(maskData, maskWidth, maskHeight)  // Per-pixel mask with dimensions
    .maskOffset(offsetX, offsetY)   // Mask offset for ROI alignment
    .bits(sampleBits)               // Significant bits per sample
    .output(preallocatedBuffer)     // Pre-allocated output (size must be exact)
    .accumulate(true)               // Add to existing values
    .parallel(true)                 // Allow multi-threading
    .compute();

// All methods except for forImage() and compute() are optional.

// The returned IntBuffer has position/limit set to cover the histogram data.
// If output(IntBuffer) was used, the returned buffer is a duplicate that shares
// storage with the original; the original's position/limit are not modified.
// For heap buffers (default or when output(int[]) was used), get the array:
int[] array = hist.array();
```

**`IHistNative`** - Low-level JNI wrapper (advanced):

```java
// Buffer-based (direct or array-backed buffers)
IHistNative.histogram8(sampleBits, imageBuffer, maskBuffer,
    height, width, imageStride, maskStride, nComponents, componentIndices,
    histogramBuffer, parallel);
```

(And, similarly, `histogram16()` for 16-bit images.)

### Java Input Types

The Java API supports both arrays and NIO buffers:

- **8-bit images**: `byte[]` or `ByteBuffer`
- **16-bit images**: `short[]` or `ShortBuffer`
- **Mask**: `byte[]` or `ByteBuffer`
- **Histogram output**: `int[]` or `IntBuffer`

### Java Notes

- **Signed types**: Java `byte` is signed (-128 to 127), but pixel values are
  interpreted as unsigned (0 to 255). The native code correctly handles this.
  Similarly for `short` with 16-bit images.

- **Accumulation**: Like the C API, histogram values are accumulated. Use
  `.accumulate(false)` (default) to zero the output first.

- **Exact buffer sizes**: Buffer and array parameters must have exactly the
  required size (not just at least the required size). For the image buffer,
  this is `width * height * nComponents`; for the mask, `width * height`; for
  the histogram output, `nHistComponents * 2^bits`. For buffers, the size that
  must match is the value of `remaining()` (distance from `position()` to
  `limit()`), not `capacity()`.

- **Thread safety**: The histogram functions are thread-safe for independent
  calls. Multiple threads can compute histograms simultaneously.

## C API

### C Installation

**Dependencies:** [oneTBB](https://github.com/oneapi-src/oneTBB) (optional but
recommended for parallelization). TBB should be installed as a shared library
and discoverable via pkg-config. (TBB as a static library works but is not
recommended unless ihist is the only code using TBB in your application.)

**Build and install:**

```sh
meson setup builddir --prefix=/your/install/path
meson compile -C builddir
meson install -C builddir
```

On Windows, add `--vsenv` to have Meson set up the Visual Studio environment.

Building with Clang (`CXX=clang++` or, on Windows, `CXX=clang-cl`) is
recommended for best performance, as benchmarking and tuning were done with
Clang.

To build without TBB, add `-Dtbb=disabled` to the `meson setup` command.

([`just`](https://just.systems) and the provided `justfile` can be used for
development builds but is not recommended for production builds.)

**Using in your project:**

Meson:

```meson
ihist_dep = dependency('ihist')
```

CMake:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(IHIST REQUIRED IMPORTED_TARGET ihist)
target_link_libraries(your_target PkgConfig::IHIST)
```

### C Functions

The C API provides two functions for computing histograms of 2D image data:

- `ihist_hist8_2d()` - for 8-bit samples (`uint8_t`)
- `ihist_hist16_2d()` - for 16-bit samples (`uint16_t`)

Both functions have identical behavior except for the sample data type.

```c
#include <ihist/ihist.h>

void ihist_hist8_2d(
    size_t sample_bits,
    uint8_t const *restrict image,
    uint8_t const *restrict mask,
    size_t height,
    size_t width,
    size_t image_stride,
    size_t mask_stride,
    size_t n_components,
    size_t n_hist_components,
    size_t const *restrict component_indices,
    uint32_t *restrict histogram,
    bool maybe_parallel);

void ihist_hist16_2d(
    size_t sample_bits,
    uint16_t const *restrict image,
    uint8_t const *restrict mask,
    size_t height,
    size_t width,
    size_t image_stride,
    size_t mask_stride,
    size_t n_components,
    size_t n_hist_components,
    size_t const *restrict component_indices,
    uint32_t *restrict histogram,
    bool maybe_parallel);
```

These functions compute histograms for one or more components (stored as
interleaved multi-sample pixels) from image data. They support:

- Multi-component images (e.g., grayscale, RGB, RGBA)
- Selective histogramming of specific components
- Optional per-pixel masking
- Region of interest (ROI) processing via stride and pointer offset
- Automatic parallelization for large images
- Arbitrary bit depths (not just full 8 or 16 bits)

### C Parameters

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
`height * mask_stride` `uint8_t` values. Only pixels where the corresponding
mask value is non-zero are included in the histogram.

Pass `NULL` to histogram all pixels.

**`height`**
Image height in pixels. May be 0 for empty input.

**`width`**
Image width in pixels. May be 0 for empty input.

**`image_stride`**
Row stride for the image in pixels (not bytes). Must be ≥ `width`.

When `image_stride` equals `width`, the image is treated as contiguous. Use
`image_stride` > `width` together with an offset `image` pointer to process a
rectangular region of interest (ROI) within a larger image.

**`mask_stride`**
Row stride for the mask in pixels (not bytes). Must be ≥ `width`.

When `mask_stride` equals `width`, the mask is treated as contiguous. This
parameter allows the mask to have a different stride from the image, which is
useful, for example, if you have a mask that covers only a rectanglular ROI of
the image.

**`n_components`**
Number of interleaved  per pixel. Examples:

- 1 for grayscale
- 3 for RGB
- 4 for RGBA

Must be > 0.

**`n_hist_components`**
Number of components to histogram. Must be > 0.

This allows histogramming a subset of components, such as skipping the alpha
component in RGBA images.

**`component_indices`**
Array of `n_hist_components` indices specifying which components to histogram.
Each index must be in the range [0, `n_components`).

Examples:

- `{0}` - histogram only the first component (e.g., red in RGB)
- `{0, 1, 2}` - histogram first three components (e.g., RGB in RGBA, skipping
  alpha)
- `{1, 2, 3}` - histogram last three components (e.g., skip first component in
  ARGB)

Must not be `NULL`.

**`histogram`** *(output, accumulated)*
Output buffer for histogram data. Must point to `n_hist_components *
2^sample_bits` `uint32_t` values.

Histograms for each component are stored consecutively:

- Bins for `component_indices[0]`: `histogram[0]` to
  `histogram[2^sample_bits - 1]`
- Bins for `component_indices[1]`: `histogram[2^sample_bits]` to
  `histogram[2 * 2^sample_bits - 1]`
- ...

**Important:** The histogram is **accumulated** into this buffer. Existing
values are added to, not replaced. To obtain a fresh histogram, zero-initialize
the buffer before calling the function.

**`maybe_parallel`**
Controls parallelization.

- `true` - Allows automatic multi-threaded execution for large images, if ihist
  was built with parallelization support (TBB).
- `false` - Guarantees single-threaded execution.

## Performance

The library uses cache-conscious algorithms with platform-specific tuning for
x86_64 and Apple ARM64. The details below are subject to change upon future
tuning.

### Parallelization

When built with TBB support (default for released binaries), histogram
computation automatically parallelizes for large images (currently ≥ 1 M
pixels). Only physical CPU cores are used (hyperthreads are excluded) because
histogramming is almost always backend-bound (that is, bottlenecked by CPU
resources that are shared between logical cores).

Although minimizing latency is a goal, we also want to maintain efficiency: CPU
time should not be consumed when it does not contribute much to speedup. This
is the reason for the relatively large parallelization threshold; we also
increase the thread count only gradually above the threshold.

Use `parallel=False` (Python), `.parallel(false)` (Java), or
`maybe_parallel=false` (C) to force single-threaded execution.

### Optimized Pixel Formats

The following pixel formats have specialized optimized code paths:

- Grayscale (1 component)
- RGB (3 components, histogramming all)
- RGBx (or BGRx, RGBA, etc.; 4 components, histogramming first 3)
- xRGB (or xBGR, ARGB, etc.; 4 components, histogramming last 3)

Other component selections (e.g., histogramming only the red and blue channels
of an RGB image) fall back to a slower generic implementation. More optimized
pixel formats could be added, subject to tradeoffs with binary size and tuning
effort.

### Bit Depth

- 8-bit images use 256-bin histograms
- 16-bit images with 12 or fewer significant bits use 4096-bin histograms
- 16-bit images with more than 12 significant bits use 65536-bin histograms

When a bit depth other than 8 (for 8-bit images) or 12 or 16 (for 16-bit
images) is requested, the resulting histogram is simply truncated.

### Memory Layout

Contiguous images are the fastest but 2D strided images are also optimized, so
that histogramming rectangular regions of interest does not require copying the
image or mask.

### Zero-copy Language Bindings

The Python and Java bindings try to avoid copying the input and output arrays
as much as possible. However, there are cases where this cannot be avoided:

- Python: C-contiguous arrays are used directly, as are 2D (single-component)
  Fortran-contiguous arrays. 3D Fortran-contiguous arrays require a copy
  because color components must be interleaved in memory. Sliced views where
  neither spatial axis is contiguous (such as `arr[:, ::2]` on a row-major
  array) are also copied.

- Java: Buffers that are not direct and whose `.hasArray()` method returns
  `false` (this includes read-only heap buffers and `ShortBuffer`/`IntBuffer`
  that are views of heap byte buffers) cannot be accessed via JNI; these are
  copied to temporary direct buffers. Direct buffers are preferred if you have
  the choice, especially for large images, because pinning heap buffers for JNI
  access can interfere with the smooth operation of the garbage collector.

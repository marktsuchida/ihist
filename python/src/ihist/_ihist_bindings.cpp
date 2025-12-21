// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <ihist/ihist.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

// Analyzed image array for zero-copy histogram processing.
// Requires pixels to be contiguous along one axis (which becomes "width");
// the other axis can have arbitrary stride (row padding). For 2D/3D, either
// axis 0 or 1 may be contiguous. For 3D, components must also be contiguous
// (axis 2). For incompatible layouts, creates a C-contiguous copy.
class ImageView {
  public:
    ImageView(nb::ndarray<nb::ro> &image, std::size_t ndim, std::size_t height,
              std::size_t width, std::size_t n_components, bool is_8bit)
        : data_(image.data()), height_(height), width_(width), stride_(width),
          pixel_size_((is_8bit ? 1 : 2) * n_components), transposed_(false) {
        std::size_t const elem_size = is_8bit ? 1 : 2;
        bool compat = false;

        if (ndim == 1) {
            compat =
                (image.stride(0) == static_cast<std::intptr_t>(elem_size));
        } else {
            bool const components_contig =
                (ndim == 2) ||
                (image.stride(2) == static_cast<std::intptr_t>(elem_size));

            if (components_contig && pixel_size_ != 0) {
                if (image.stride(1) ==
                    static_cast<std::intptr_t>(pixel_size_)) {
                    // Axis 1 contiguous: rows along axis 0
                    if (image.stride(0) >=
                            static_cast<std::intptr_t>(width * pixel_size_) &&
                        static_cast<std::size_t>(image.stride(0)) %
                                pixel_size_ ==
                            0) {
                        compat = true;
                        stride_ = static_cast<std::size_t>(image.stride(0)) /
                                  pixel_size_;
                    }
                } else if (image.stride(0) ==
                           static_cast<std::intptr_t>(pixel_size_)) {
                    // Axis 0 contiguous: rows along axis 1
                    if (image.stride(1) >=
                            static_cast<std::intptr_t>(height * pixel_size_) &&
                        static_cast<std::size_t>(image.stride(1)) %
                                pixel_size_ ==
                            0) {
                        compat = true;
                        transposed_ = true;
                        std::swap(height_, width_);
                        stride_ = static_cast<std::size_t>(image.stride(1)) /
                                  pixel_size_;
                    }
                }
            }
        }

        if (!compat) {
            auto image_c =
                nb::cast<nb::ndarray<nb::ro, nb::c_contig>>(image.cast());
            owner_ = nb::cast(image_c);
            data_ = image_c.data();
            height_ = height;
            width_ = width;
            stride_ = width;
            transposed_ = false;
        }
    }

    [[nodiscard]] auto data() const -> void const * { return data_; }
    [[nodiscard]] auto height() const -> std::size_t { return height_; }
    [[nodiscard]] auto width() const -> std::size_t { return width_; }
    [[nodiscard]] auto stride() const -> std::size_t { return stride_; }
    [[nodiscard]] auto pixel_size() const -> std::size_t {
        return pixel_size_;
    }
    [[nodiscard]] auto transposed() const -> bool { return transposed_; }

    [[nodiscard]] auto memory_span() const -> std::size_t {
        return (height_ > 0) ? (height_ - 1) * stride_ * pixel_size_ +
                                   width_ * pixel_size_
                             : 0;
    }

    [[nodiscard]] auto overlaps_with(void const *ptr, std::size_t size) const
        -> bool {
        auto a_start = static_cast<char const *>(data_);
        auto b_start = static_cast<char const *>(ptr);
        auto a_end = a_start + memory_span();
        auto b_end = b_start + size;
        return !(a_end <= b_start || b_end <= a_start);
    }

  private:
    void const *data_;
    std::size_t height_;     // Effective height (may be swapped for F-order)
    std::size_t width_;      // Effective width (may be swapped for F-order)
    std::size_t stride_;     // Row stride in pixels
    std::size_t pixel_size_; // Bytes per pixel
    nb::object owner_;       // Prevents deallocation of any copy
    bool transposed_;        // True if axis 0 is contiguous (H/W swapped)
};

// Analyzed mask array for zero-copy histogram processing.
// Mask must have the same contiguous axis as the image. On mismatch, creates
// a copy with the required layout.
class MaskView {
  public:
    // Construct an empty (no-mask) view.
    explicit MaskView(std::size_t effective_width)
        : data_(nullptr), stride_(effective_width) {}

    // Construct from a mask array. The mask must have the same contiguous axis
    // as the image (axis 1 if not transposed, axis 0 if transposed).
    MaskView(nb::ndarray<nb::ro> &mask, std::size_t orig_height,
             std::size_t orig_width, bool image_transposed)
        : data_(nullptr),
          stride_(image_transposed ? orig_height : orig_width) {
        std::size_t const contig_axis = image_transposed ? 0 : 1;
        std::size_t const row_axis = image_transposed ? 1 : 0;
        std::size_t const min_row_stride =
            image_transposed ? orig_height : orig_width;

        if (mask.stride(contig_axis) == 1 &&
            mask.stride(row_axis) >=
                static_cast<std::intptr_t>(min_row_stride)) {
            // Mask has matching contiguous axis
            data_ = static_cast<std::uint8_t const *>(mask.data());
            stride_ = static_cast<std::size_t>(mask.stride(row_axis));
        } else {
            // Copy mask to required layout
            if (!image_transposed) {
                auto mask_c =
                    nb::cast<nb::ndarray<nb::ro, nb::c_contig>>(mask.cast());
                owner_ = nb::cast(mask_c);
                data_ = static_cast<std::uint8_t const *>(mask_c.data());
                stride_ = orig_width;
            } else {
                nb::module_ np = nb::module_::import_("numpy");
                nb::object mask_obj = mask.cast();
                owner_ = np.attr("asfortranarray")(mask_obj);
                auto mask_f = nb::cast<nb::ndarray<nb::ro>>(owner_);
                data_ = static_cast<std::uint8_t const *>(mask_f.data());
                stride_ = orig_height;
            }
        }
    }

    [[nodiscard]] auto data() const -> std::uint8_t const * { return data_; }
    [[nodiscard]] auto stride() const -> std::size_t { return stride_; }

    [[nodiscard]] auto memory_span(std::size_t height, std::size_t width) const
        -> std::size_t {
        return (height > 0) ? (height - 1) * stride_ + width : 0;
    }

    [[nodiscard]] auto overlaps_with(void const *ptr, std::size_t size,
                                     std::size_t height,
                                     std::size_t width) const -> bool {
        if (data_ == nullptr)
            return false;
        auto a_start =
            static_cast<char const *>(static_cast<void const *>(data_));
        auto b_start = static_cast<char const *>(ptr);
        auto a_end = a_start + memory_span(height, width);
        auto b_end = b_start + size;
        return !(a_end <= b_start || b_end <= a_start);
    }

  private:
    std::uint8_t const *data_; // nullptr if no mask
    std::size_t stride_;       // Row stride in bytes
    nb::object owner_;         // Prevents deallocation of any copy
};

} // namespace

nb::object histogram(nb::ndarray<nb::ro> image,
                     nb::object bits_obj = nb::none(),
                     nb::object mask_obj = nb::none(),
                     nb::object components_obj = nb::none(),
                     nb::object out_obj = nb::none(), bool accumulate = false,
                     bool parallel = true) {
    bool const is_8bit = image.dtype() == nb::dtype<std::uint8_t>();
    bool const is_16bit = image.dtype() == nb::dtype<std::uint16_t>();
    if (!is_8bit && !is_16bit) {
        throw std::invalid_argument("Image must have dtype uint8 or uint16");
    }
    std::size_t const max_bits = is_8bit ? 8 : 16;

    std::size_t const ndim = image.ndim();
    if (ndim < 1 || ndim > 3) {
        throw std::invalid_argument("Image must be 1D, 2D, or 3D, got " +
                                    std::to_string(ndim) + "D");
    }

    std::size_t height, width, n_components;
    if (ndim == 1) {
        height = 1;
        width = image.shape(0);
        n_components = 1;
    } else if (ndim == 2) {
        height = image.shape(0);
        width = image.shape(1);
        n_components = 1;
    } else {
        height = image.shape(0);
        width = image.shape(1);
        n_components = image.shape(2);
    }

    ImageView const img(image, ndim, height, width, n_components, is_8bit);

    std::size_t const n_pixels = height * width;
    if (n_pixels > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Image has too many pixels (" + std::to_string(n_pixels) +
            "); maximum is " +
            std::to_string(std::numeric_limits<std::uint32_t>::max()) +
            " to avoid histogram overflow");
    }

    std::size_t sample_bits = max_bits;
    if (!bits_obj.is_none()) {
        sample_bits = nb::cast<std::size_t>(bits_obj);
        if (sample_bits < 1 || sample_bits > max_bits) {
            throw std::invalid_argument("bits must be in range [1, " +
                                        std::to_string(max_bits) + "], got " +
                                        std::to_string(sample_bits));
        }
    }

    std::size_t const n_hist_components =
        components_obj.is_none()
            ? n_components
            : nb::len(nb::cast<nb::sequence>(components_obj));

    std::vector<std::size_t> component_indices(n_hist_components);
    std::iota(component_indices.begin(), component_indices.end(), 0);
    if (!components_obj.is_none()) {
        auto const components_seq = nb::cast<nb::sequence>(components_obj);
        std::transform(component_indices.begin(), component_indices.end(),
                       component_indices.begin(), [&](std::size_t i) {
                           std::size_t const idx =
                               nb::cast<std::size_t>(components_seq[i]);
                           if (idx >= n_components) {
                               throw std::invalid_argument(
                                   "Component index " + std::to_string(idx) +
                                   " out of range [0, " +
                                   std::to_string(n_components) + ")");
                           }
                           return idx;
                       });
    }

    MaskView const msk = [&]() {
        if (mask_obj.is_none())
            return MaskView(img.width());

        auto mask = nb::cast<nb::ndarray<nb::ro>>(mask_obj);
        if (mask.dtype() != nb::dtype<std::uint8_t>()) {
            throw std::invalid_argument("Mask must have dtype uint8");
        }
        if (mask.ndim() != 2) {
            throw std::invalid_argument("Mask must be 2D, got " +
                                        std::to_string(mask.ndim()) + "D");
        }
        // Validate against original image shape (before any transposition)
        if (mask.shape(0) != height || mask.shape(1) != width) {
            throw std::invalid_argument(
                "Mask shape " + std::to_string(mask.shape(0)) + "x" +
                std::to_string(mask.shape(1)) +
                " does not match image shape " + std::to_string(height) + "x" +
                std::to_string(width));
        }

        return MaskView(mask, height, width, img.transposed());
    }();

    std::size_t const n_bins = 1uLL << sample_bits;
    std::size_t const hist_size = n_hist_components * n_bins;

    std::uint32_t *hist_ptr = nullptr;
    nb::object out_array;
    if (!out_obj.is_none()) {
        auto out = nb::cast<nb::ndarray<nb::c_contig>>(out_obj);
        if (out.dtype() != nb::dtype<std::uint32_t>()) {
            throw std::invalid_argument("Output must have dtype uint32");
        }

        if (out.ndim() == 1) {
            if (n_hist_components > 1) {
                throw std::invalid_argument(
                    "Output must be 2D for multi-component histogram");
            }
            if (out.shape(0) != n_bins) {
                throw std::invalid_argument("Output shape (" +
                                            std::to_string(out.shape(0)) +
                                            ",) does not match expected (" +
                                            std::to_string(n_bins) + ",)");
            }
        } else if (out.ndim() == 2) {
            if (out.shape(0) != n_hist_components || out.shape(1) != n_bins) {
                throw std::invalid_argument(
                    "Output shape (" + std::to_string(out.shape(0)) + ", " +
                    std::to_string(out.shape(1)) +
                    ") does not match expected (" +
                    std::to_string(n_hist_components) + ", " +
                    std::to_string(n_bins) + ")");
            }
        } else {
            throw std::invalid_argument("Output must be 1D or 2D, got " +
                                        std::to_string(out.ndim()) + "D");
        }

        if (img.overlaps_with(out.data(), out.nbytes())) {
            throw std::invalid_argument(
                "Output buffer overlaps with input image");
        }
        if (msk.overlaps_with(out.data(), out.nbytes(), img.height(),
                              img.width())) {
            throw std::invalid_argument("Output buffer overlaps with mask");
        }

        out_array = out_obj;
        hist_ptr = static_cast<std::uint32_t *>(out.data());
    } else {
        std::size_t shape[2];
        std::size_t out_ndim;

        // Return 2D if image is 3D or components explicitly specified; this
        // makes it easy to write generic code that handles any number of
        // components.
        bool const force_2d = (ndim == 3) || !components_obj.is_none();
        if (n_hist_components == 1 && !force_2d) {
            out_ndim = 1;
            shape[0] = n_bins;
        } else {
            out_ndim = 2;
            shape[0] = n_hist_components;
            shape[1] = n_bins;
        }

        nb::ndarray<nb::numpy, std::uint32_t> arr(nullptr, out_ndim, shape,
                                                  nb::handle());
        out_array = nb::cast(arr);
        auto out_arr = nb::cast<nb::ndarray<std::uint32_t>>(out_array);
        hist_ptr = out_arr.data();
    }

    if (out_obj.is_none() || !accumulate) {
        std::fill(hist_ptr, std::next(hist_ptr, hist_size), 0);
    }

    // We could keep the GIL acquired when data size is small (say, less than
    // 500 elements; should benchmark), but always release for now.
    {
        nb::gil_scoped_release gil_released;

        if (is_8bit) {
            ihist_hist8_2d(sample_bits,
                           static_cast<std::uint8_t const *>(img.data()),
                           msk.data(), img.height(), img.width(), img.stride(),
                           msk.stride(), n_components, n_hist_components,
                           component_indices.data(), hist_ptr, parallel);
        } else {
            ihist_hist16_2d(
                sample_bits, static_cast<std::uint16_t const *>(img.data()),
                msk.data(), img.height(), img.width(), img.stride(),
                msk.stride(), n_components, n_hist_components,
                component_indices.data(), hist_ptr, parallel);
        }
    }

    return out_array;
}

NB_MODULE(_ihist_bindings, m) {
    m.doc() = "Fast image histograms";

    m.def("histogram", &histogram, nb::arg("image"), nb::kw_only(),
          nb::arg("bits") = nb::none(), nb::arg("mask") = nb::none(),
          nb::arg("components") = nb::none(), nb::arg("out") = nb::none(),
          nb::arg("accumulate") = false, nb::arg("parallel") = true,
          R"doc(
        Compute histogram of image pixel values.

        Parameters
        ----------
        image : array_like
            Input image data. Must be uint8 or uint16, and 1D, 2D, or 3D.
            - 1D arrays (W,) are interpreted as (1, W, 1)
            - 2D arrays (H, W) are interpreted as (H, W, 1)
            - 3D arrays (H, W, C) use C as number of components per pixel
            Total pixel count must not exceed `2^32-1`.

        bits : int, optional
            Number of significant bits per sample. If not specified, defaults
            to full depth (8 for uint8, 16 for uint16). Valid range: [1, 8] for
            uint8, [1, 16] for uint16.

        mask : array_like, optional
            Per-pixel mask. Must be uint8, 2D, shape (H, W). Only pixels with
            non-zero mask values are included. If not specified, all pixels are
            included.

        components : sequence of int, optional
            Indices of components to histogram. If not specified, all
            components are histogrammed. Each index must be in range
            [0, n_components). For example, given an RGBA image (C = 4),
            components=[0, 1, 2] will histogram the R, G, and B components only.

        out : array_like, optional
            Pre-allocated output buffer. Must be uint32, and either 1D with
            shape (2^bits,) for single-component histogram, or 2D with shape
            (n_hist_components, 2^bits). If not specified, a new array is
            allocated and returned.

        accumulate : bool, optional
            If False (default), the output buffer is zeroed before computing
            the histogram. If True, histogram values are accumulated into the
            existing buffer values.

        parallel : bool, optional
            If True (default), allows automatic multi-threaded execution for
            large images. If False, guarantees single-threaded execution.

        Returns
        -------
        histogram : ndarray
            Histogram(s) as uint32 array. If the image is 1D or 2D and
            'components' is not specified, returns 1D array of shape
            (2^bits,). If the image is 3D or 'components' is explicitly
            specified, returns 2D array of shape (n_hist_components, 2^bits).
            If 'out' was provided, returns the same array after filling.
        )doc");
}

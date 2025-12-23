// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

#include <jni.h>

#include <ihist/ihist.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ihistj_IHistNative.h"

namespace {

void throw_illegal_argument(JNIEnv *env, char const *message) {
    jclass clazz = env->FindClass("java/lang/IllegalArgumentException");
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message);
    }
}

// Convert Java int[] to vector<size_t>, checking for negative values
auto to_size_t_vector(JNIEnv *env, jintArray arr) -> std::vector<std::size_t> {
    if (arr == nullptr) {
        return {};
    }
    jsize const len = env->GetArrayLength(arr);
    std::vector<std::size_t> result(static_cast<std::size_t>(len));
    jint *elements = env->GetIntArrayElements(arr, nullptr);
    if (elements == nullptr) {
        return {};
    }
    for (jsize i = 0; i < len; ++i) {
        if (elements[i] < 0) {
            env->ReleaseIntArrayElements(arr, elements, JNI_ABORT);
            throw_illegal_argument(env, "Component index cannot be negative");
            return {};
        }
        result[static_cast<std::size_t>(i)] =
            static_cast<std::size_t>(elements[i]);
    }
    env->ReleaseIntArrayElements(arr, elements, JNI_ABORT);
    return result;
}

auto validate_component_indices(JNIEnv *env,
                                std::vector<std::size_t> const &indices,
                                std::size_t n_components) -> bool {
    for (std::size_t const idx : indices) {
        if (idx >= n_components) {
            throw_illegal_argument(
                env, "component index out of range [0, nComponents)");
            return false;
        }
    }
    return true;
}

auto get_buffer_position(JNIEnv *env, jobject buffer) -> jint {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID position_method =
        env->GetMethodID(buffer_class, "position", "()I");
    return env->CallIntMethod(buffer, position_method);
}

template <typename PixelT> struct jni_pixel_traits;

template <> struct jni_pixel_traits<std::uint8_t> {
    using jni_array_type = jbyteArray;
    using jni_element_type = jbyte;
    using pixel_type = std::uint8_t;
    static constexpr int max_sample_bits = 8;
    static constexpr char const *bit_error_msg =
        "sampleBits must be in range [1, 8] for 8-bit";
    static constexpr char const *buffer_type_error =
        "ByteBuffer must be direct";

    static auto get_array_elements(JNIEnv *env, jni_array_type arr)
        -> jni_element_type * {
        return env->GetByteArrayElements(arr, nullptr);
    }

    static void release_array_elements(JNIEnv *env, jni_array_type arr,
                                       jni_element_type *elems, jint mode) {
        env->ReleaseByteArrayElements(arr, elems, mode);
    }

    static auto get_critical(JNIEnv *env, jni_array_type arr)
        -> jni_element_type * {
        return static_cast<jni_element_type *>(
            env->GetPrimitiveArrayCritical(arr, nullptr));
    }

    static void release_critical(JNIEnv *env, jni_array_type arr,
                                 jni_element_type *elems, jint mode) {
        env->ReleasePrimitiveArrayCritical(arr, elems, mode);
    }

    static void call_ihist(std::size_t sample_bits, pixel_type const *image,
                           std::uint8_t const *mask, std::size_t height,
                           std::size_t width, std::size_t image_stride,
                           std::size_t mask_stride, std::size_t n_components,
                           std::size_t n_hist_components,
                           std::size_t const *component_indices,
                           std::uint32_t *histogram, bool parallel) {
        ihist_hist8_2d(sample_bits, image, mask, height, width, image_stride,
                       mask_stride, n_components, n_hist_components,
                       component_indices, histogram, parallel);
    }
};

template <> struct jni_pixel_traits<std::uint16_t> {
    using jni_array_type = jshortArray;
    using jni_element_type = jshort;
    using pixel_type = std::uint16_t;
    static constexpr int max_sample_bits = 16;
    static constexpr char const *bit_error_msg =
        "sampleBits must be in range [1, 16] for 16-bit";
    static constexpr char const *buffer_type_error =
        "ShortBuffer must be direct";

    static auto get_array_elements(JNIEnv *env, jni_array_type arr)
        -> jni_element_type * {
        return env->GetShortArrayElements(arr, nullptr);
    }

    static void release_array_elements(JNIEnv *env, jni_array_type arr,
                                       jni_element_type *elems, jint mode) {
        env->ReleaseShortArrayElements(arr, elems, mode);
    }

    static auto get_critical(JNIEnv *env, jni_array_type arr)
        -> jni_element_type * {
        return static_cast<jni_element_type *>(
            env->GetPrimitiveArrayCritical(arr, nullptr));
    }

    static void release_critical(JNIEnv *env, jni_array_type arr,
                                 jni_element_type *elems, jint mode) {
        env->ReleasePrimitiveArrayCritical(arr, elems, mode);
    }

    static void call_ihist(std::size_t sample_bits, pixel_type const *image,
                           std::uint8_t const *mask, std::size_t height,
                           std::size_t width, std::size_t image_stride,
                           std::size_t mask_stride, std::size_t n_components,
                           std::size_t n_hist_components,
                           std::size_t const *component_indices,
                           std::uint32_t *histogram, bool parallel) {
        ihist_hist16_2d(sample_bits, image, mask, height, width, image_stride,
                        mask_stride, n_components, n_hist_components,
                        component_indices, histogram, parallel);
    }
};

template <typename PixelT>
auto validate_params(JNIEnv *env, jint sample_bits, jint height, jint width,
                     jint image_stride, jint mask_stride, jint n_components,
                     jintArray component_indices) -> bool {
    using traits = jni_pixel_traits<PixelT>;
    if (sample_bits < 1 || sample_bits > traits::max_sample_bits) {
        throw_illegal_argument(env, traits::bit_error_msg);
        return false;
    }
    if (height < 0 || width < 0) {
        throw_illegal_argument(env, "height and width must be non-negative");
        return false;
    }
    if (image_stride < width) {
        throw_illegal_argument(env, "imageStride must be >= width");
        return false;
    }
    if (mask_stride < width) {
        throw_illegal_argument(env, "maskStride must be >= width");
        return false;
    }
    if (n_components < 1) {
        throw_illegal_argument(env, "nComponents must be >= 1");
        return false;
    }
    if (component_indices == nullptr) {
        throw_illegal_argument(env, "componentIndices cannot be null");
        return false;
    }
    jsize const n_hist_components = env->GetArrayLength(component_indices);
    if (n_hist_components < 1) {
        throw_illegal_argument(env, "must histogram at least one component");
        return false;
    }
    return true;
}

// Returns pointer to direct buffer data, or nullptr with exception if not
// direct
template <typename PixelT>
auto get_image_buffer_data(JNIEnv *env, jobject image_buffer)
    -> PixelT const * {
    using traits = jni_pixel_traits<PixelT>;

    void *image_direct = env->GetDirectBufferAddress(image_buffer);
    if (image_direct != nullptr) {
        jint const position = get_buffer_position(env, image_buffer);
        return reinterpret_cast<PixelT const *>(
            static_cast<char *>(image_direct) +
            position * sizeof(typename traits::jni_element_type));
    }

    throw_illegal_argument(env, traits::buffer_type_error);
    return nullptr;
}

// Returns pointer to direct mask buffer data, or nullptr with exception if not
// direct
auto get_mask_buffer_data(JNIEnv *env, jobject mask_buffer)
    -> std::uint8_t const * {
    void *mask_direct = env->GetDirectBufferAddress(mask_buffer);
    if (mask_direct != nullptr) {
        jint const position = get_buffer_position(env, mask_buffer);
        return static_cast<std::uint8_t const *>(mask_direct) + position;
    }

    throw_illegal_argument(env, "mask ByteBuffer must be direct");
    return nullptr;
}

// Buffer-based histogram implementation template (direct buffers only)
template <typename PixelT>
void histogram_buffer_impl(JNIEnv *env, jint sample_bits, jobject image_buffer,
                           jobject mask_buffer, jint height, jint width,
                           jint image_stride, jint mask_stride,
                           jint n_components, jintArray component_indices,
                           jobject histogram_buffer, jboolean parallel) {
    using traits = jni_pixel_traits<PixelT>;

    if (!validate_params<PixelT>(env, sample_bits, height, width, image_stride,
                                 mask_stride, n_components,
                                 component_indices)) {
        return;
    }

    std::vector<std::size_t> indices =
        to_size_t_vector(env, component_indices);
    if (env->ExceptionCheck()) {
        return;
    }
    std::size_t const n_hist_components = indices.size();

    if (!validate_component_indices(env, indices,
                                    static_cast<std::size_t>(n_components))) {
        return;
    }

    if (image_buffer == nullptr) {
        throw_illegal_argument(env, "image buffer cannot be null");
        return;
    }
    if (histogram_buffer == nullptr) {
        throw_illegal_argument(env, "histogram buffer cannot be null");
        return;
    }

    PixelT const *image_ptr = get_image_buffer_data<PixelT>(env, image_buffer);
    if (image_ptr == nullptr) {
        return;
    }

    std::uint8_t const *mask_ptr = nullptr;
    if (mask_buffer != nullptr) {
        mask_ptr = get_mask_buffer_data(env, mask_buffer);
        if (mask_ptr == nullptr) {
            return;
        }
    }

    void *histogram_direct = env->GetDirectBufferAddress(histogram_buffer);
    if (histogram_direct == nullptr) {
        throw_illegal_argument(env, "histogram IntBuffer must be direct");
        return;
    }
    jint const position = get_buffer_position(env, histogram_buffer);
    auto *histogram_ptr = reinterpret_cast<std::uint32_t *>(
        static_cast<char *>(histogram_direct) + position * sizeof(jint));

    traits::call_ihist(
        static_cast<std::size_t>(sample_bits), image_ptr, mask_ptr,
        static_cast<std::size_t>(height), static_cast<std::size_t>(width),
        static_cast<std::size_t>(image_stride),
        static_cast<std::size_t>(mask_stride),
        static_cast<std::size_t>(n_components), n_hist_components,
        indices.data(), histogram_ptr, parallel != JNI_FALSE);
}

template <typename PixelT>
void histogram_array_impl(
    JNIEnv *env, jint sample_bits,
    typename jni_pixel_traits<PixelT>::jni_array_type image, jint image_offset,
    jbyteArray mask, jint mask_offset, jint height, jint width,
    jint image_stride, jint mask_stride, jint n_components,
    jintArray component_indices, jintArray histogram, jint histogram_offset,
    jboolean parallel) {

    using traits = jni_pixel_traits<PixelT>;

    if (!validate_params<PixelT>(env, sample_bits, height, width, image_stride,
                                 mask_stride, n_components,
                                 component_indices)) {
        return;
    }

    std::vector<std::size_t> indices =
        to_size_t_vector(env, component_indices);
    if (env->ExceptionCheck()) {
        return;
    }
    std::size_t const n_hist_components = indices.size();

    if (!validate_component_indices(env, indices,
                                    static_cast<std::size_t>(n_components))) {
        return;
    }

    if (image_offset < 0) {
        throw_illegal_argument(env, "imageOffset cannot be negative");
        return;
    }
    if (mask != nullptr && mask_offset < 0) {
        throw_illegal_argument(env, "maskOffset cannot be negative");
        return;
    }
    if (histogram_offset < 0) {
        throw_illegal_argument(env, "histogramOffset cannot be negative");
        return;
    }

    if (image == nullptr) {
        throw_illegal_argument(env, "image cannot be null");
        return;
    }
    jsize const image_len = env->GetArrayLength(image);
    jlong const required_image_len =
        (height > 0 && width > 0)
            ? static_cast<jlong>(image_offset) +
                  (static_cast<jlong>(height - 1) * image_stride +
                   static_cast<jlong>(width)) *
                      n_components
            : static_cast<jlong>(image_offset);
    if (image_len < required_image_len) {
        throw_illegal_argument(env,
                               "image array too small for given dimensions");
        return;
    }

    if (mask != nullptr) {
        jsize const mask_len = env->GetArrayLength(mask);
        jlong const required_mask_len =
            (height > 0 && width > 0)
                ? static_cast<jlong>(mask_offset) +
                      static_cast<jlong>(height - 1) * mask_stride +
                      static_cast<jlong>(width)
                : static_cast<jlong>(mask_offset);
        if (mask_len < required_mask_len) {
            throw_illegal_argument(
                env, "mask array too small for given dimensions");
            return;
        }
    }

    if (histogram == nullptr) {
        throw_illegal_argument(env, "histogram cannot be null");
        return;
    }
    jsize const histogram_len = env->GetArrayLength(histogram);
    jlong const required_histogram_len =
        static_cast<jlong>(histogram_offset) +
        static_cast<jlong>(n_hist_components) * (1L << sample_bits);
    if (histogram_len < required_histogram_len) {
        throw_illegal_argument(
            env, "histogram array too small for given parameters");
        return;
    }

    auto *image_ptr = traits::get_critical(env, image);
    if (image_ptr == nullptr) {
        return;
    }

    jbyte *mask_ptr = nullptr;
    if (mask != nullptr) {
        mask_ptr = static_cast<jbyte *>(
            env->GetPrimitiveArrayCritical(mask, nullptr));
        if (mask_ptr == nullptr) {
            traits::release_critical(env, image, image_ptr, JNI_ABORT);
            return;
        }
    }

    jint *histogram_ptr = static_cast<jint *>(
        env->GetPrimitiveArrayCritical(histogram, nullptr));
    if (histogram_ptr == nullptr) {
        if (mask_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(mask, mask_ptr, JNI_ABORT);
        }
        traits::release_critical(env, image, image_ptr, JNI_ABORT);
        return;
    }

    traits::call_ihist(
        static_cast<std::size_t>(sample_bits),
        reinterpret_cast<PixelT const *>(image_ptr + image_offset),
        mask_ptr != nullptr
            ? reinterpret_cast<std::uint8_t const *>(mask_ptr + mask_offset)
            : nullptr,
        static_cast<std::size_t>(height), static_cast<std::size_t>(width),
        static_cast<std::size_t>(image_stride),
        static_cast<std::size_t>(mask_stride),
        static_cast<std::size_t>(n_components), n_hist_components,
        indices.data(),
        reinterpret_cast<std::uint32_t *>(histogram_ptr + histogram_offset),
        parallel != JNI_FALSE);

    env->ReleasePrimitiveArrayCritical(histogram, histogram_ptr, 0);
    if (mask_ptr != nullptr) {
        env->ReleasePrimitiveArrayCritical(mask, mask_ptr, JNI_ABORT);
    }
    traits::release_critical(env, image, image_ptr, JNI_ABORT);
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram8__I_3BI_3BIIIIII_3I_3IIZ(
    JNIEnv *env, jclass, jint sample_bits, jbyteArray image, jint image_offset,
    jbyteArray mask, jint mask_offset, jint height, jint width,
    jint image_stride, jint mask_stride, jint n_components,
    jintArray component_indices, jintArray histogram, jint histogram_offset,
    jboolean parallel) {
    histogram_array_impl<std::uint8_t>(
        env, sample_bits, image, image_offset, mask, mask_offset, height,
        width, image_stride, mask_stride, n_components, component_indices,
        histogram, histogram_offset, parallel);
}

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram16__I_3SI_3BIIIIII_3I_3IIZ(
    JNIEnv *env, jclass, jint sample_bits, jshortArray image,
    jint image_offset, jbyteArray mask, jint mask_offset, jint height,
    jint width, jint image_stride, jint mask_stride, jint n_components,
    jintArray component_indices, jintArray histogram, jint histogram_offset,
    jboolean parallel) {
    histogram_array_impl<std::uint16_t>(
        env, sample_bits, image, image_offset, mask, mask_offset, height,
        width, image_stride, mask_stride, n_components, component_indices,
        histogram, histogram_offset, parallel);
}

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram8__ILjava_nio_ByteBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
    JNIEnv *env, jclass, jint sample_bits, jobject image_buffer,
    jobject mask_buffer, jint height, jint width, jint image_stride,
    jint mask_stride, jint n_components, jintArray component_indices,
    jobject histogram_buffer, jboolean parallel) {
    histogram_buffer_impl<std::uint8_t>(
        env, sample_bits, image_buffer, mask_buffer, height, width,
        image_stride, mask_stride, n_components, component_indices,
        histogram_buffer, parallel);
}

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram16__ILjava_nio_ShortBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
    JNIEnv *env, jclass, jint sample_bits, jobject image_buffer,
    jobject mask_buffer, jint height, jint width, jint image_stride,
    jint mask_stride, jint n_components, jintArray component_indices,
    jobject histogram_buffer, jboolean parallel) {
    histogram_buffer_impl<std::uint16_t>(
        env, sample_bits, image_buffer, mask_buffer, height, width,
        image_stride, mask_stride, n_components, component_indices,
        histogram_buffer, parallel);
}

} // extern "C"

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
        env->DeleteLocalRef(clazz);
    }
}

void throw_null_pointer(JNIEnv *env, char const *message) {
    jclass clazz = env->FindClass("java/lang/NullPointerException");
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message);
        env->DeleteLocalRef(clazz);
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

// Buffer helper functions - call Java Buffer methods
// These functions properly clean up local references and check for exceptions.

auto get_buffer_position(JNIEnv *env, jobject buffer) -> jint {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID position_method =
        env->GetMethodID(buffer_class, "position", "()I");
    env->DeleteLocalRef(buffer_class);
    if (position_method == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    return env->CallIntMethod(buffer, position_method);
}

auto is_direct_buffer(JNIEnv *env, jobject buffer) -> bool {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID is_direct_method =
        env->GetMethodID(buffer_class, "isDirect", "()Z");
    env->DeleteLocalRef(buffer_class);
    if (is_direct_method == nullptr || env->ExceptionCheck()) {
        return false;
    }
    return env->CallBooleanMethod(buffer, is_direct_method) != JNI_FALSE;
}

auto has_array(JNIEnv *env, jobject buffer) -> bool {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID has_array_method =
        env->GetMethodID(buffer_class, "hasArray", "()Z");
    env->DeleteLocalRef(buffer_class);
    if (has_array_method == nullptr || env->ExceptionCheck()) {
        return false;
    }
    return env->CallBooleanMethod(buffer, has_array_method) != JNI_FALSE;
}

auto get_buffer_array(JNIEnv *env, jobject buffer) -> jarray {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID array_method =
        env->GetMethodID(buffer_class, "array", "()Ljava/lang/Object;");
    env->DeleteLocalRef(buffer_class);
    if (array_method == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    return static_cast<jarray>(env->CallObjectMethod(buffer, array_method));
}

auto get_array_offset(JNIEnv *env, jobject buffer) -> jint {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID array_offset_method =
        env->GetMethodID(buffer_class, "arrayOffset", "()I");
    env->DeleteLocalRef(buffer_class);
    if (array_offset_method == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    return env->CallIntMethod(buffer, array_offset_method);
}

auto is_read_only(JNIEnv *env, jobject buffer) -> bool {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID is_read_only_method =
        env->GetMethodID(buffer_class, "isReadOnly", "()Z");
    env->DeleteLocalRef(buffer_class);
    if (is_read_only_method == nullptr || env->ExceptionCheck()) {
        return false;
    }
    return env->CallBooleanMethod(buffer, is_read_only_method) != JNI_FALSE;
}

auto get_buffer_remaining(JNIEnv *env, jobject buffer) -> jint {
    jclass buffer_class = env->GetObjectClass(buffer);
    jmethodID remaining_method =
        env->GetMethodID(buffer_class, "remaining", "()I");
    env->DeleteLocalRef(buffer_class);
    if (remaining_method == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    return env->CallIntMethod(buffer, remaining_method);
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
        "image buffer must be direct or array-backed";

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
        "image buffer must be direct or array-backed";

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
        throw_null_pointer(env, "componentIndices cannot be null");
        return false;
    }
    jsize const n_hist_components = env->GetArrayLength(component_indices);
    if (n_hist_components < 1) {
        throw_illegal_argument(env, "must histogram at least one component");
        return false;
    }
    return true;
}

// Structure to hold buffer data pointer and cleanup info.
// For array-backed buffers, backing_array holds the local reference to the
// array returned by Buffer.array(). This reference must be deleted after
// ReleasePrimitiveArrayCritical is called.
struct buffer_data {
    void *ptr = nullptr;
    jarray backing_array = nullptr; // Non-null if using critical array access
    bool valid = false;
};

// Get image buffer data - handles both direct and array-backed buffers.
// Validates buffer capacity against required size.
template <typename PixelT>
auto get_image_buffer_data(JNIEnv *env, jobject image_buffer,
                           std::size_t required_elements) -> buffer_data {
    using traits = jni_pixel_traits<PixelT>;
    buffer_data result;

    // Validate buffer capacity
    jint remaining = get_buffer_remaining(env, image_buffer);
    if (env->ExceptionCheck()) {
        return result;
    }
    if (static_cast<std::size_t>(remaining) < required_elements) {
        throw_illegal_argument(env, "image buffer has insufficient capacity");
        return result;
    }

    if (is_direct_buffer(env, image_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        void *direct = env->GetDirectBufferAddress(image_buffer);
        if (direct != nullptr) {
            jint const position = get_buffer_position(env, image_buffer);
            if (env->ExceptionCheck()) {
                return result;
            }
            result.ptr = static_cast<char *>(direct) +
                         position * sizeof(typename traits::jni_element_type);
            result.valid = true;
            return result;
        }
    }

    if (has_array(env, image_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        jarray arr = get_buffer_array(env, image_buffer);
        if (env->ExceptionCheck()) {
            if (arr != nullptr) {
                env->DeleteLocalRef(arr);
            }
            return result;
        }
        if (arr != nullptr) {
            void *critical = env->GetPrimitiveArrayCritical(arr, nullptr);
            if (critical != nullptr) {
                jint const array_offset = get_array_offset(env, image_buffer);
                jint const position = get_buffer_position(env, image_buffer);
                if (env->ExceptionCheck()) {
                    env->ReleasePrimitiveArrayCritical(arr, critical,
                                                       JNI_ABORT);
                    env->DeleteLocalRef(arr);
                    return result;
                }
                result.ptr = static_cast<typename traits::jni_element_type *>(
                                 critical) +
                             array_offset + position;
                result.backing_array = arr;
                result.valid = true;
                return result;
            }
            env->DeleteLocalRef(arr);
        }
    }

    throw_illegal_argument(env, traits::buffer_type_error);
    return result;
}

// Get mask buffer data - handles both direct and array-backed buffers.
// Validates buffer capacity against required size.
auto get_mask_buffer_data(JNIEnv *env, jobject mask_buffer,
                          std::size_t required_elements) -> buffer_data {
    buffer_data result;

    // Validate buffer capacity
    jint remaining = get_buffer_remaining(env, mask_buffer);
    if (env->ExceptionCheck()) {
        return result;
    }
    if (static_cast<std::size_t>(remaining) < required_elements) {
        throw_illegal_argument(env, "mask buffer has insufficient capacity");
        return result;
    }

    if (is_direct_buffer(env, mask_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        void *direct = env->GetDirectBufferAddress(mask_buffer);
        if (direct != nullptr) {
            jint const position = get_buffer_position(env, mask_buffer);
            if (env->ExceptionCheck()) {
                return result;
            }
            result.ptr = static_cast<std::uint8_t *>(direct) + position;
            result.valid = true;
            return result;
        }
    }

    if (has_array(env, mask_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        jarray arr = get_buffer_array(env, mask_buffer);
        if (env->ExceptionCheck()) {
            if (arr != nullptr) {
                env->DeleteLocalRef(arr);
            }
            return result;
        }
        if (arr != nullptr) {
            void *critical = env->GetPrimitiveArrayCritical(arr, nullptr);
            if (critical != nullptr) {
                jint const array_offset = get_array_offset(env, mask_buffer);
                jint const position = get_buffer_position(env, mask_buffer);
                if (env->ExceptionCheck()) {
                    env->ReleasePrimitiveArrayCritical(arr, critical,
                                                       JNI_ABORT);
                    env->DeleteLocalRef(arr);
                    return result;
                }
                result.ptr =
                    static_cast<jbyte *>(critical) + array_offset + position;
                result.backing_array = arr;
                result.valid = true;
                return result;
            }
            env->DeleteLocalRef(arr);
        }
    }

    throw_illegal_argument(env, "mask buffer must be direct or array-backed");
    return result;
}

// Get histogram buffer data - handles both direct and array-backed buffers.
// Also checks for read-only buffers and validates capacity.
auto get_histogram_buffer_data(JNIEnv *env, jobject histogram_buffer,
                               std::size_t required_elements) -> buffer_data {
    buffer_data result;

    if (is_read_only(env, histogram_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        throw_illegal_argument(env, "histogram buffer cannot be read-only");
        return result;
    }

    // Validate buffer capacity
    jint remaining = get_buffer_remaining(env, histogram_buffer);
    if (env->ExceptionCheck()) {
        return result;
    }
    if (static_cast<std::size_t>(remaining) < required_elements) {
        throw_illegal_argument(env,
                               "histogram buffer has insufficient capacity");
        return result;
    }

    if (is_direct_buffer(env, histogram_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        void *direct = env->GetDirectBufferAddress(histogram_buffer);
        if (direct != nullptr) {
            jint const position = get_buffer_position(env, histogram_buffer);
            if (env->ExceptionCheck()) {
                return result;
            }
            result.ptr = static_cast<jint *>(direct) + position;
            result.valid = true;
            return result;
        }
    }

    if (has_array(env, histogram_buffer)) {
        if (env->ExceptionCheck()) {
            return result;
        }
        jarray arr = get_buffer_array(env, histogram_buffer);
        if (env->ExceptionCheck()) {
            if (arr != nullptr) {
                env->DeleteLocalRef(arr);
            }
            return result;
        }
        if (arr != nullptr) {
            void *critical = env->GetPrimitiveArrayCritical(arr, nullptr);
            if (critical != nullptr) {
                jint const array_offset =
                    get_array_offset(env, histogram_buffer);
                jint const position =
                    get_buffer_position(env, histogram_buffer);
                if (env->ExceptionCheck()) {
                    env->ReleasePrimitiveArrayCritical(arr, critical,
                                                       JNI_ABORT);
                    env->DeleteLocalRef(arr);
                    return result;
                }
                result.ptr =
                    static_cast<jint *>(critical) + array_offset + position;
                result.backing_array = arr;
                result.valid = true;
                return result;
            }
            env->DeleteLocalRef(arr);
        }
    }

    throw_illegal_argument(env,
                           "histogram buffer must be direct or array-backed");
    return result;
}

// Release buffer data - releases critical array and deletes local reference.
void release_buffer_data(JNIEnv *env, buffer_data const &data, jint mode) {
    if (data.backing_array != nullptr) {
        env->ReleasePrimitiveArrayCritical(data.backing_array, data.ptr, mode);
        env->DeleteLocalRef(data.backing_array);
    }
    // Direct buffers don't need cleanup
}

// Buffer-based histogram implementation template.
// Handles both direct and array-backed buffers.
//
// Note on critical array access: This function may hold multiple critical
// arrays simultaneously (image, mask, histogram) during histogram computation.
// This is safe because:
// 1. The histogram computation is CPU-bound and does not call back into Java
// 2. No JNI calls are made while holding critical arrays
// 3. The computation typically completes quickly
// Per JNI specification, critical regions should be short and non-blocking,
// which this satisfies for typical image sizes.
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
        throw_null_pointer(env, "image buffer cannot be null");
        return;
    }
    if (histogram_buffer == nullptr) {
        throw_null_pointer(env, "histogram buffer cannot be null");
        return;
    }

    // Calculate required buffer capacities
    std::size_t const h = static_cast<std::size_t>(height);
    std::size_t const w = static_cast<std::size_t>(width);
    std::size_t const img_stride = static_cast<std::size_t>(image_stride);
    std::size_t const msk_stride = static_cast<std::size_t>(mask_stride);
    std::size_t const n_comp = static_cast<std::size_t>(n_components);

    std::size_t image_required =
        (h > 0 && w > 0) ? ((h - 1) * img_stride + w) * n_comp : 0;
    std::size_t mask_required =
        (h > 0 && w > 0) ? (h - 1) * msk_stride + w : 0;
    std::size_t hist_required =
        n_hist_components * (static_cast<std::size_t>(1) << sample_bits);

    // Get image data
    buffer_data image_data =
        get_image_buffer_data<PixelT>(env, image_buffer, image_required);
    if (!image_data.valid) {
        return;
    }

    // Get mask data (optional)
    buffer_data mask_data;
    if (mask_buffer != nullptr) {
        mask_data = get_mask_buffer_data(env, mask_buffer, mask_required);
        if (!mask_data.valid) {
            release_buffer_data(env, image_data, JNI_ABORT);
            return;
        }
    }

    // Get histogram data
    buffer_data histogram_data =
        get_histogram_buffer_data(env, histogram_buffer, hist_required);
    if (!histogram_data.valid) {
        if (mask_data.valid) {
            release_buffer_data(env, mask_data, JNI_ABORT);
        }
        release_buffer_data(env, image_data, JNI_ABORT);
        return;
    }

    // Call the histogram function
    traits::call_ihist(
        static_cast<std::size_t>(sample_bits),
        static_cast<typename traits::pixel_type const *>(image_data.ptr),
        mask_data.valid ? static_cast<std::uint8_t const *>(mask_data.ptr)
                        : nullptr,
        h, w, img_stride, msk_stride, n_comp, n_hist_components,
        indices.data(), static_cast<std::uint32_t *>(histogram_data.ptr),
        parallel != JNI_FALSE);

    // Release buffers - histogram with mode 0 to copy back changes
    release_buffer_data(env, histogram_data, 0);
    if (mask_data.valid) {
        release_buffer_data(env, mask_data, JNI_ABORT);
    }
    release_buffer_data(env, image_data, JNI_ABORT);
}

} // namespace

extern "C" {

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

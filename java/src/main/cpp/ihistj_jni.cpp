// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

#include <jni.h>

#include <ihist/ihist.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ihistj_IHistNative.h"

namespace {

struct cached_ids {
    jclass buffer_class;
    jmethodID position;
    jmethodID remaining;
    jmethodID is_direct;
    jmethodID has_array;
    jmethodID array;
    jmethodID array_offset;
    jmethodID is_read_only;
};

cached_ids g_ids{};

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
            throw_illegal_argument(env, "component index cannot be negative");
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

// Buffer helper functions using cached method IDs.

auto get_buffer_position(JNIEnv *env, jobject buffer) -> jint {
    return env->CallIntMethod(buffer, g_ids.position);
}

auto is_direct_buffer(JNIEnv *env, jobject buffer) -> bool {
    return env->CallBooleanMethod(buffer, g_ids.is_direct) != JNI_FALSE;
}

auto has_array(JNIEnv *env, jobject buffer) -> bool {
    return env->CallBooleanMethod(buffer, g_ids.has_array) != JNI_FALSE;
}

auto get_buffer_array(JNIEnv *env, jobject buffer) -> jarray {
    return static_cast<jarray>(env->CallObjectMethod(buffer, g_ids.array));
}

auto get_array_offset(JNIEnv *env, jobject buffer) -> jint {
    return env->CallIntMethod(buffer, g_ids.array_offset);
}

auto is_read_only(JNIEnv *env, jobject buffer) -> bool {
    return env->CallBooleanMethod(buffer, g_ids.is_read_only) != JNI_FALSE;
}

auto get_buffer_remaining(JNIEnv *env, jobject buffer) -> jint {
    return env->CallIntMethod(buffer, g_ids.remaining);
}

template <typename PixelT> struct jni_pixel_traits;

template <> struct jni_pixel_traits<std::uint8_t> {
    using jni_array_type = jbyteArray;
    using jni_element_type = jbyte;
    using pixel_type = std::uint8_t;
    static constexpr int max_sample_bits = 8;
    static constexpr char const *bit_error_msg =
        "sampleBits must be in range [0, 8] for 8-bit";

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
        "sampleBits must be in range [0, 16] for 16-bit";

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
    if (sample_bits < 0 || sample_bits > traits::max_sample_bits) {
        throw_illegal_argument(env, traits::bit_error_msg);
        return false;
    }
    if (height < 0 || width < 0) {
        throw_illegal_argument(env, "height and width must be >= 0");
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
    if (n_components < 0) {
        throw_illegal_argument(env, "nComponents must be >= 0");
        return false;
    }
    if (component_indices == nullptr) {
        throw_null_pointer(env, "componentIndices cannot be null");
        return false;
    }
    return true;
}

// RAII wrapper (unique_ptr-like semantics) for access to a Java buffer.
class buffer_access {
  public:
    // For direct buffers (no cleanup needed)
    explicit buffer_access(void *data_ptr) : data_ptr_(data_ptr) {}

    // For array-backed buffers
    buffer_access(JNIEnv *env, void *critical_ptr, void *data_ptr,
                  jarray backing_array, jint release_mode)
        : env_(env), critical_ptr_(critical_ptr), data_ptr_(data_ptr),
          backing_array_(backing_array), release_mode_(release_mode) {}

    ~buffer_access() {
        if (backing_array_ != nullptr) {
            env_->ReleasePrimitiveArrayCritical(backing_array_, critical_ptr_,
                                                release_mode_);
            env_->DeleteLocalRef(backing_array_);
        }
    }

    buffer_access(buffer_access const &) = delete;
    buffer_access &operator=(buffer_access const &) = delete;

    buffer_access(buffer_access &&other) noexcept
        : env_(other.env_), critical_ptr_(other.critical_ptr_),
          data_ptr_(other.data_ptr_), backing_array_(other.backing_array_),
          release_mode_(other.release_mode_) {
        other.backing_array_ = nullptr;
        other.critical_ptr_ = nullptr;
        other.data_ptr_ = nullptr;
    }

    buffer_access &operator=(buffer_access &&other) noexcept {
        if (this != &other) {
            if (backing_array_ != nullptr) {
                env_->ReleasePrimitiveArrayCritical(
                    backing_array_, critical_ptr_, release_mode_);
                env_->DeleteLocalRef(backing_array_);
            }
            env_ = other.env_;
            critical_ptr_ = other.critical_ptr_;
            data_ptr_ = other.data_ptr_;
            backing_array_ = other.backing_array_;
            release_mode_ = other.release_mode_;
            other.backing_array_ = nullptr;
        }
        return *this;
    }

    void *ptr() const { return data_ptr_; }

  private:
    JNIEnv *env_ = nullptr;
    void *critical_ptr_ = nullptr; // Original pointer for release
    void *data_ptr_ = nullptr;     // Offset-adjusted pointer for use
    jarray backing_array_ = nullptr;
    jint release_mode_ = JNI_ABORT;
};

// Get buffer data - handles both direct and array-backed buffers.
// Validates buffer capacity against required size.
// Returns std::nullopt on failure (exception will have been thrown).
// Template parameters:
//   ElementT: The element type for pointer arithmetic
//   IsWritable: If true, checks for read-only and uses release mode 0
template <typename ElementT, bool IsWritable = false>
auto get_buffer_access(JNIEnv *env, jobject buffer,
                       std::size_t required_elements, char const *buffer_name)
    -> std::optional<buffer_access> {
    constexpr int release_mode = IsWritable ? 0 : JNI_ABORT;

    if constexpr (IsWritable) {
        if (is_read_only(env, buffer)) {
            if (env->ExceptionCheck()) {
                return std::nullopt;
            }
            throw_illegal_argument(
                env, (std::string(buffer_name) + " buffer cannot be read-only")
                         .c_str());
            return std::nullopt;
        }
    }

    jint remaining = get_buffer_remaining(env, buffer);
    if (env->ExceptionCheck()) {
        return std::nullopt;
    }
    if (static_cast<std::size_t>(remaining) < required_elements) {
        throw_illegal_argument(env, (std::string(buffer_name) +
                                     " buffer has insufficient capacity")
                                        .c_str());
        return std::nullopt;
    }

    if (is_direct_buffer(env, buffer)) {
        if (env->ExceptionCheck()) {
            return std::nullopt;
        }
        void *direct = env->GetDirectBufferAddress(buffer);
        if (direct != nullptr) {
            jint const position = get_buffer_position(env, buffer);
            if (env->ExceptionCheck()) {
                return std::nullopt;
            }
            void *data_ptr = static_cast<ElementT *>(direct) + position;
            return buffer_access(data_ptr);
        }
    }

    if (has_array(env, buffer)) {
        if (env->ExceptionCheck()) {
            return std::nullopt;
        }
        jarray arr = get_buffer_array(env, buffer);
        if (env->ExceptionCheck()) {
            if (arr != nullptr) {
                env->DeleteLocalRef(arr);
            }
            return std::nullopt;
        }
        if (arr != nullptr) {
            void *critical = env->GetPrimitiveArrayCritical(arr, nullptr);
            if (critical != nullptr) {
                jint const array_offset = get_array_offset(env, buffer);
                jint const position = get_buffer_position(env, buffer);
                if (env->ExceptionCheck()) {
                    env->ReleasePrimitiveArrayCritical(arr, critical,
                                                       JNI_ABORT);
                    env->DeleteLocalRef(arr);
                    return std::nullopt;
                }
                void *data_ptr = static_cast<ElementT *>(critical) +
                                 array_offset + position;
                return buffer_access(env, critical, data_ptr, arr,
                                     release_mode);
            }
            env->DeleteLocalRef(arr);
        }
    }

    throw_illegal_argument(env, (std::string(buffer_name) +
                                 " buffer must be direct or array-backed")
                                    .c_str());
    return std::nullopt;
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
void histogram_impl(JNIEnv *env, jint sample_bits, jobject image_buffer,
                    jobject mask_buffer, jint height, jint width,
                    jint image_stride, jint mask_stride, jint n_components,
                    jintArray component_indices, jobject histogram_buffer,
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

    if (n_hist_components == 0) {
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

    std::optional<buffer_access> image_data =
        get_buffer_access<typename traits::jni_element_type>(
            env, image_buffer, image_required, "image");
    if (!image_data) {
        return;
    }

    std::optional<buffer_access> mask_data;
    if (mask_buffer != nullptr) {
        mask_data =
            get_buffer_access<jbyte>(env, mask_buffer, mask_required, "mask");
        if (!mask_data) {
            return;
        }
    }

    std::optional<buffer_access> histogram_data =
        get_buffer_access<jint, true>(env, histogram_buffer, hist_required,
                                      "histogram");
    if (!histogram_data) {
        return;
    }

    traits::call_ihist(
        static_cast<std::size_t>(sample_bits),
        static_cast<typename traits::pixel_type const *>(image_data->ptr()),
        mask_data ? static_cast<std::uint8_t const *>(mask_data->ptr())
                  : nullptr,
        h, w, img_stride, msk_stride, n_comp, n_hist_components,
        indices.data(), static_cast<std::uint32_t *>(histogram_data->ptr()),
        parallel != JNI_FALSE);
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram8__ILjava_nio_ByteBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
    JNIEnv *env, jclass, jint sample_bits, jobject image_buffer,
    jobject mask_buffer, jint height, jint width, jint image_stride,
    jint mask_stride, jint n_components, jintArray component_indices,
    jobject histogram_buffer, jboolean parallel) {
    histogram_impl<std::uint8_t>(env, sample_bits, image_buffer, mask_buffer,
                                 height, width, image_stride, mask_stride,
                                 n_components, component_indices,
                                 histogram_buffer, parallel);
}

JNIEXPORT void JNICALL
Java_ihistj_IHistNative_histogram16__ILjava_nio_ShortBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
    JNIEnv *env, jclass, jint sample_bits, jobject image_buffer,
    jobject mask_buffer, jint height, jint width, jint image_stride,
    jint mask_stride, jint n_components, jintArray component_indices,
    jobject histogram_buffer, jboolean parallel) {
    histogram_impl<std::uint16_t>(env, sample_bits, image_buffer, mask_buffer,
                                  height, width, image_stride, mask_stride,
                                  n_components, component_indices,
                                  histogram_buffer, parallel);
}

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) !=
        JNI_OK) {
        return JNI_ERR;
    }

    jclass local_class = env->FindClass("java/nio/Buffer");
    if (local_class == nullptr) {
        return JNI_ERR;
    }
    g_ids.buffer_class = static_cast<jclass>(env->NewGlobalRef(local_class));
    env->DeleteLocalRef(local_class);
    if (g_ids.buffer_class == nullptr) {
        return JNI_ERR;
    }

    g_ids.position = env->GetMethodID(g_ids.buffer_class, "position", "()I");
    g_ids.remaining = env->GetMethodID(g_ids.buffer_class, "remaining", "()I");
    g_ids.is_direct = env->GetMethodID(g_ids.buffer_class, "isDirect", "()Z");
    g_ids.has_array = env->GetMethodID(g_ids.buffer_class, "hasArray", "()Z");
    g_ids.array =
        env->GetMethodID(g_ids.buffer_class, "array", "()Ljava/lang/Object;");
    g_ids.array_offset =
        env->GetMethodID(g_ids.buffer_class, "arrayOffset", "()I");
    g_ids.is_read_only =
        env->GetMethodID(g_ids.buffer_class, "isReadOnly", "()Z");

    if (g_ids.position == nullptr || g_ids.remaining == nullptr ||
        g_ids.is_direct == nullptr || g_ids.has_array == nullptr ||
        g_ids.array == nullptr || g_ids.array_offset == nullptr ||
        g_ids.is_read_only == nullptr) {
        env->DeleteGlobalRef(g_ids.buffer_class);
        g_ids.buffer_class = nullptr;
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) ==
        JNI_OK) {
        if (g_ids.buffer_class != nullptr) {
            env->DeleteGlobalRef(g_ids.buffer_class);
            g_ids.buffer_class = nullptr;
        }
    }
}

} // extern "C"

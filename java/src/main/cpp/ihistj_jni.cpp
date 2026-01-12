// This file is part of ihist
// Copyright 2025 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: MIT

#include "io_github_marktsuchida_ihist_IHistNative.h"

#include <jni.h>

#include <ihist/ihist.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

// RAII wrapper for JNI local references.
template <typename T> class local_ref {
    JNIEnv *env_ = nullptr;
    T ref_ = nullptr;

  public:
    local_ref() = default;
    local_ref(JNIEnv *env, T ref) : env_(env), ref_(ref) {}
    ~local_ref() {
        if (ref_ != nullptr) {
            env_->DeleteLocalRef(ref_);
        }
    }

    local_ref(local_ref const &) = delete;
    local_ref &operator=(local_ref const &) = delete;

    local_ref(local_ref &&other) noexcept
        : env_(other.env_), ref_(other.ref_) {
        other.env_ = nullptr;
        other.ref_ = nullptr;
    }

    local_ref &operator=(local_ref &&other) noexcept {
        if (this != &other) {
            if (ref_ != nullptr) {
                env_->DeleteLocalRef(ref_);
            }
            env_ = other.env_;
            ref_ = other.ref_;
            other.env_ = nullptr;
            other.ref_ = nullptr;
        }
        return *this;
    }

    T get() const { return ref_; }
    explicit operator bool() const { return ref_ != nullptr; }

    T release() noexcept {
        T r = ref_;
        ref_ = nullptr;
        return r;
    }
};

// Base class for type-erased array access lifetime management.
class array_access_base {
  public:
    virtual ~array_access_base() = default;
};

// Traits for JNI array region operations, mapping element types to the
// appropriate Get*ArrayRegion and Set*ArrayRegion functions.
template <typename ElementT> struct jni_array_region_traits;

template <> struct jni_array_region_traits<jbyte> {
    static void get_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jbyte *buf) {
        env->GetByteArrayRegion(static_cast<jbyteArray>(array), start, len,
                                buf);
    }
    static void set_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jbyte const *buf) {
        env->SetByteArrayRegion(static_cast<jbyteArray>(array), start, len,
                                buf);
    }
};

template <> struct jni_array_region_traits<jshort> {
    static void get_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jshort *buf) {
        env->GetShortArrayRegion(static_cast<jshortArray>(array), start, len,
                                 buf);
    }
    static void set_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jshort const *buf) {
        env->SetShortArrayRegion(static_cast<jshortArray>(array), start, len,
                                 buf);
    }
};

template <> struct jni_array_region_traits<jint> {
    static void get_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jint *buf) {
        env->GetIntArrayRegion(static_cast<jintArray>(array), start, len, buf);
    }
    static void set_region(JNIEnv *env, jarray array, jsize start, jsize len,
                           jint const *buf) {
        env->SetIntArrayRegion(static_cast<jintArray>(array), start, len, buf);
    }
};

// RAII wrapper for JNI array access with fallback.
// Tries GetPrimitiveArrayCritical first; if that fails (and no exception is
// pending), falls back to Get*ArrayRegion with a temporary buffer.
// Supports two modes:
// - Non-owning: array ref managed by caller (e.g., parameter from Java)
// - Owning: takes ownership of a local_ref<jarray>, deletes after release
template <typename ElementT> class array_access : public array_access_base {
    using traits = jni_array_region_traits<ElementT>;

    JNIEnv *env_ = nullptr;
    jarray array_ = nullptr;
    std::optional<local_ref<jarray>> owned_array_;
    ElementT *data_ptr_ = nullptr;
    std::vector<ElementT> fallback_buffer_;
    bool using_fallback_ = false;
    jint release_mode_ = JNI_ABORT;

    void init_access() {
        if (array_ == nullptr) {
            return;
        }
        void *critical = env_->GetPrimitiveArrayCritical(array_, nullptr);
        if (critical != nullptr) {
            data_ptr_ = static_cast<ElementT *>(critical);
            return;
        }
        if (env_->ExceptionCheck()) {
            return;
        }
        // Critical access failed without exception; fall back to region copy.
        jsize const len = env_->GetArrayLength(array_);
        if (env_->ExceptionCheck() || len < 0) {
            return;
        }
        fallback_buffer_.resize(static_cast<std::size_t>(len));
        traits::get_region(env_, array_, 0, len, fallback_buffer_.data());
        if (env_->ExceptionCheck()) {
            fallback_buffer_.clear();
            return;
        }
        data_ptr_ = fallback_buffer_.data();
        using_fallback_ = true;
    }

  public:
    array_access() = default;

    // Non-owning: array ref managed by caller.
    array_access(JNIEnv *env, jarray array, jint mode = JNI_ABORT)
        : env_(env), array_(array), release_mode_(mode) {
        init_access();
    }

    // Owning: takes ownership of local ref.
    array_access(JNIEnv *env, local_ref<jarray> &&array, jint mode)
        : env_(env), array_(array.get()), owned_array_(std::move(array)),
          release_mode_(mode) {
        init_access();
    }

    ~array_access() override {
        if (data_ptr_ == nullptr) {
            return;
        }
        if (using_fallback_) {
            if (release_mode_ != JNI_ABORT) {
                jsize const len = static_cast<jsize>(fallback_buffer_.size());
                traits::set_region(env_, array_, 0, len,
                                   fallback_buffer_.data());
                // Clear any exception from set_region; cannot propagate from
                // destructor.
                env_->ExceptionClear();
            }
        } else {
            env_->ReleasePrimitiveArrayCritical(array_, data_ptr_,
                                                release_mode_);
        }
    }

    array_access(array_access const &) = delete;
    array_access &operator=(array_access const &) = delete;
    array_access(array_access &&) = delete;
    array_access &operator=(array_access &&) = delete;

    ElementT *get() const { return data_ptr_; }
    explicit operator bool() const { return data_ptr_ != nullptr; }
};

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
    local_ref<jclass> clazz(
        env, env->FindClass("java/lang/IllegalArgumentException"));
    if (clazz) {
        env->ThrowNew(clazz.get(), message);
    }
}

void throw_null_pointer(JNIEnv *env, char const *message) {
    local_ref<jclass> clazz(env,
                            env->FindClass("java/lang/NullPointerException"));
    if (clazz) {
        env->ThrowNew(clazz.get(), message);
    }
}

// Convert Java int[] to vector<size_t>, checking for negative values.
// Returns nullopt if an exception was thrown.
[[nodiscard]] auto to_size_t_vector(JNIEnv *env, jintArray arr)
    -> std::optional<std::vector<std::size_t>> {
    if (arr == nullptr) {
        return std::vector<std::size_t>{};
    }
    jsize const len = env->GetArrayLength(arr);
    std::vector<std::size_t> result(static_cast<std::size_t>(len));
    array_access<jint> access(env, arr);
    if (!access) {
        return {};
    }
    jint const *elements = access.get();
    for (jsize i = 0; i < len; ++i) {
        if (elements[i] < 0) {
            throw_illegal_argument(env, "component index cannot be negative");
            return {};
        }
        result[static_cast<std::size_t>(i)] =
            static_cast<std::size_t>(elements[i]);
    }
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

[[nodiscard]] auto get_buffer_position(JNIEnv *env, jobject buffer)
    -> std::optional<jint> {
    jint result = env->CallIntMethod(buffer, g_ids.position);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result;
}

[[nodiscard]] auto is_direct_buffer(JNIEnv *env, jobject buffer)
    -> std::optional<bool> {
    jboolean result = env->CallBooleanMethod(buffer, g_ids.is_direct);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result != JNI_FALSE;
}

[[nodiscard]] auto has_array(JNIEnv *env, jobject buffer)
    -> std::optional<bool> {
    jboolean result = env->CallBooleanMethod(buffer, g_ids.has_array);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result != JNI_FALSE;
}

[[nodiscard]] auto get_buffer_array(JNIEnv *env, jobject buffer)
    -> std::optional<local_ref<jarray>> {
    local_ref<jarray> result(
        env, static_cast<jarray>(env->CallObjectMethod(buffer, g_ids.array)));
    if (env->ExceptionCheck()) {
        return {};
    }
    return result;
}

[[nodiscard]] auto get_array_offset(JNIEnv *env, jobject buffer)
    -> std::optional<jint> {
    jint result = env->CallIntMethod(buffer, g_ids.array_offset);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result;
}

[[nodiscard]] auto is_read_only(JNIEnv *env, jobject buffer)
    -> std::optional<bool> {
    jboolean result = env->CallBooleanMethod(buffer, g_ids.is_read_only);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result != JNI_FALSE;
}

[[nodiscard]] auto get_buffer_remaining(JNIEnv *env, jobject buffer)
    -> std::optional<jint> {
    jint result = env->CallIntMethod(buffer, g_ids.remaining);
    if (env->ExceptionCheck()) {
        return {};
    }
    return result;
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
                     jint image_stride, jint mask_stride, bool has_mask,
                     jint n_components, jintArray component_indices) -> bool {
    using traits = jni_pixel_traits<PixelT>;
    if (sample_bits < 0 || sample_bits > traits::max_sample_bits) {
        throw_illegal_argument(env, traits::bit_error_msg);
        return false;
    }
    if (height < 0 || width < 0) {
        throw_illegal_argument(env, "height and width must be >= 0");
        return false;
    }
    if (static_cast<std::int64_t>(width) * height >
        std::numeric_limits<jint>::max()) {
        throw_illegal_argument(
            env, "width * height must not exceed Integer.MAX_VALUE");
        return false;
    }
    if (image_stride < width) {
        throw_illegal_argument(env, "imageStride must be >= width");
        return false;
    }
    if (has_mask) {
        if (mask_stride < width) {
            throw_illegal_argument(env, "maskStride must be >= width");
            return false;
        }
    } else {
        if (mask_stride != 0) {
            throw_illegal_argument(env,
                                   "maskStride must be 0 when mask is null");
            return false;
        }
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

// RAII wrapper for access to a Java buffer.
// Composes array_access for array-backed buffers.
class buffer_access {
    std::unique_ptr<array_access_base> array_access_;
    void *data_ptr_ = nullptr;

  public:
    // For direct buffers (no cleanup needed)
    explicit buffer_access(void *data_ptr) : data_ptr_(data_ptr) {}

    // For array-backed buffers
    template <typename ElementT>
    buffer_access(std::unique_ptr<array_access<ElementT>> access,
                  void *data_ptr)
        : array_access_(std::move(access)), data_ptr_(data_ptr) {}

    void *ptr() const { return data_ptr_; }
};

// Get buffer data - handles both direct and array-backed buffers.
// Validates buffer remaining size against required size.
// Returns std::nullopt on failure (exception will have been thrown).
// Template parameters:
//   ElementT: The element type for pointer arithmetic
//   IsWritable: If true, checks for read-only and uses release mode 0
template <typename ElementT, bool IsWritable = false>
auto get_buffer_access(JNIEnv *env, jobject buffer,
                       std::size_t required_elements, char const *buffer_name)
    -> std::optional<buffer_access> {
    constexpr jint release_mode = IsWritable ? 0 : JNI_ABORT;

    if constexpr (IsWritable) {
        auto read_only = is_read_only(env, buffer);
        if (!read_only) {
            return {};
        }
        if (*read_only) {
            throw_illegal_argument(
                env, (std::string(buffer_name) + " buffer cannot be read-only")
                         .c_str());
            return {};
        }
    }

    auto remaining = get_buffer_remaining(env, buffer);
    if (!remaining) {
        return {};
    }
    if (static_cast<std::size_t>(*remaining) != required_elements) {
        throw_illegal_argument(
            env, (std::string(buffer_name) + " buffer has incorrect size " +
                  std::to_string(*remaining) + " (expected " +
                  std::to_string(required_elements) + ")")
                     .c_str());
        return {};
    }

    auto is_direct = is_direct_buffer(env, buffer);
    if (!is_direct) {
        return {};
    }
    if (*is_direct) {
        void *direct = env->GetDirectBufferAddress(buffer);
        if (env->ExceptionCheck()) {
            return {};
        }
        if (direct != nullptr) {
            auto position = get_buffer_position(env, buffer);
            if (!position) {
                return {};
            }
            void *data_ptr = static_cast<ElementT *>(direct) + *position;
            return buffer_access(data_ptr);
        }
    }

    auto has_arr = has_array(env, buffer);
    if (!has_arr) {
        return {};
    }
    if (*has_arr) {
        auto arr = get_buffer_array(env, buffer);
        if (!arr) {
            return {};
        }
        if (*arr) {
            auto access = std::make_unique<array_access<ElementT>>(
                env, std::move(*arr), release_mode);
            if (*access) {
                auto array_offset = get_array_offset(env, buffer);
                if (!array_offset) {
                    return {};
                }
                auto position = get_buffer_position(env, buffer);
                if (!position) {
                    return {};
                }
                void *data_ptr = access->get() + *array_offset + *position;
                return buffer_access(std::move(access), data_ptr);
            }
        }
    }

    throw_illegal_argument(env, (std::string(buffer_name) +
                                 " buffer must be direct or array-backed")
                                    .c_str());
    return {};
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
                                 mask_stride, mask_buffer != nullptr,
                                 n_components, component_indices)) {
        return;
    }

    auto indices = to_size_t_vector(env, component_indices);
    if (!indices) {
        return;
    }
    std::size_t const n_hist_components = indices->size();

    if (!validate_component_indices(env, *indices,
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
        indices->data(), static_cast<std::uint32_t *>(histogram_data->ptr()),
        parallel != JNI_FALSE);
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_io_github_marktsuchida_ihist_IHistNative_histogram8__ILjava_nio_ByteBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
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
Java_io_github_marktsuchida_ihist_IHistNative_histogram16__ILjava_nio_ShortBuffer_2Ljava_nio_ByteBuffer_2IIIII_3ILjava_nio_IntBuffer_2Z(
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

    local_ref<jclass> local_class(env, env->FindClass("java/nio/Buffer"));
    if (!local_class) {
        return JNI_ERR;
    }
    g_ids.buffer_class =
        static_cast<jclass>(env->NewGlobalRef(local_class.release()));
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

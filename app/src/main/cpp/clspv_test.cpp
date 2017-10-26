/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "clspv_utils.hpp"
#include "half.hpp"
#include "util_init.hpp"
#include "vulkan_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <iterator>
#include <string>

#include <vulkan/vulkan.hpp>

/* ============================================================================================== */

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
           // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}

/* ============================================================================================== */

enum cl_channel_order {
    CL_R = 0x10B0,
    CL_A = 0x10B1,
    CL_RG = 0x10B2,
    CL_RA = 0x10B3,
    CL_RGB = 0x10B4,
    CL_RGBA = 0x10B5,
    CL_BGRA = 0x10B6,
    CL_ARGB = 0x10B7,
    CL_INTENSITY = 0x10B8,
    CL_LUMINANCE = 0x10B9,
    CL_Rx = 0x10BA,
    CL_RGx = 0x10BB,
    CL_RGBx = 0x10BC,
    CL_DEPTH = 0x10BD,
    CL_DEPTH_STENCIL = 0x10BE,
};

enum cl_channel_type {
    CL_SNORM_INT8 = 0x10D0,
    CL_SNORM_INT16 = 0x10D1,
    CL_UNORM_INT8 = 0x10D2,
    CL_UNORM_INT16 = 0x10D3,
    CL_UNORM_SHORT_565 = 0x10D4,
    CL_UNORM_SHORT_555 = 0x10D5,
    CL_UNORM_INT_101010 = 0x10D6,
    CL_SIGNED_INT8 = 0x10D7,
    CL_SIGNED_INT16 = 0x10D8,
    CL_SIGNED_INT32 = 0x10D9,
    CL_UNSIGNED_INT8 = 0x10DA,
    CL_UNSIGNED_INT16 = 0x10DB,
    CL_UNSIGNED_INT32 = 0x10DC,
    CL_HALF_FLOAT = 0x10DD,
    CL_FLOAT = 0x10DE,
    CL_UNORM_INT24 = 0x10DF,
};

enum {
    CLK_ADDRESS_NONE = 0x0000,
    CLK_ADDRESS_CLAMP_TO_EDGE = 0x0002,
    CLK_ADDRESS_CLAMP = 0x0004,
    CLK_ADDRESS_REPEAT = 0x0006,
    CLK_ADDRESS_MIRRORED_REPEAT = 0x0008,
    CLK_ADDRESS_MASK = 0x000E,

    CLK_NORMALIZED_COORDS_FALSE = 0x0000,
    CLK_NORMALIZED_COORDS_TRUE = 0x0001,
    CLK_NORMALIZED_COORDS_MASK = 0x0001,

    CLK_FILTER_NEAREST = 0x0010,
    CLK_FILTER_LINEAR = 0x0020,
    CLK_FILTER_MASK = 0x0030
};

/* ============================================================================================== */

std::pair<int,int> operator+=(std::pair<int,int>& l, const std::pair<int,int>& r) {
    l.first += r.first;
    l.second += r.second;
    return l;
};

/* ============================================================================================== */

template <typename T>
struct alignas(2*sizeof(T)) vec2 {
    vec2(T a, T b) : x(a), y(b) {}
    vec2(T a) : vec2(a, T(0)) {}
    vec2() : vec2(T(0), T(0)) {}

    vec2(const vec2<T>& other) : x(other.x), y(other.y) {}
    vec2(vec2<T>&& other) : vec2() {
        swap(*this, other);
    }

    vec2<T>& operator=(vec2<T> other) {
        swap(*this, other);
        return *this;
    }

    T   x;
    T   y;
};

template <typename T>
void swap(vec2<T>& first, vec2<T>& second) {
    using std::swap;

    swap(first.x, second.x);
    swap(first.y, second.y);
}

template <typename T>
bool operator==(const vec2<T>& l, const vec2<T>& r) {
    return (l.x == r.x) && (l.y == r.y);
}

template <typename T>
struct alignas(4*sizeof(T)) vec4 {
    vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
    vec4(T a, T b, T c) : vec4(a, b, c, T(0)) {}
    vec4(T a, T b) : vec4(a, b, T(0), T(0)) {}
    vec4(T a) : vec4(a, T(0), T(0), T(0)) {}
    vec4() : vec4(T(0), T(0), T(0), T(0)) {}

    vec4(const vec4<T>& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
    vec4(vec4<T>&& other) : vec4() {
        swap(*this, other);
    }

    vec4<T>& operator=(vec4<T> other) {
        swap(*this, other);
        return *this;
    }

    T   x;
    T   y;
    T   z;
    T   w;
};

template <typename T>
void swap(vec4<T>& first, vec4<T>& second) {
    using std::swap;

    swap(first.x, second.x);
    swap(first.y, second.y);
    swap(first.z, second.z);
    swap(first.w, second.w);
}

template <typename T>
bool operator==(const vec4<T>& l, const vec4<T>& r) {
    return (l.x == r.x) && (l.y == r.y) && (l.z == r.z) && (l.w == r.w);
}

static_assert(sizeof(float) == 4, "bad size for float");

typedef vec2<float> float2;
static_assert(sizeof(float2) == 8, "bad size for float2");

template<>
bool operator==(const float2& l, const float2& r) {
    const int ulp = 2;
    return almost_equal(l.x, r.x, ulp)
           && almost_equal(l.y, r.y, ulp);
}

typedef vec4<float> float4;
static_assert(sizeof(float4) == 16, "bad size for float4");

template<>
bool operator==(const float4& l, const float4& r) {
    const int ulp = 2;
    return almost_equal(l.x, r.x, ulp)
           && almost_equal(l.y, r.y, ulp)
           && almost_equal(l.z, r.z, ulp)
           && almost_equal(l.w, r.w, ulp);
}

typedef half_float::half    half;
static_assert(sizeof(half) == 2, "bad size for half");

template <>
struct std::is_floating_point<half> : std::true_type {};
static_assert(std::is_floating_point<half>::value, "half should be floating point");

typedef vec2<half> half2;
static_assert(sizeof(half2) == 4, "bad size for half2");

typedef vec4<half> half4;
static_assert(sizeof(half4) == 8, "bad size for half4");

typedef unsigned short   ushort;
static_assert(sizeof(ushort) == 2, "bad size for ushort");

typedef vec2<ushort> ushort2;
static_assert(sizeof(ushort2) == 4, "bad size for ushort2");

typedef vec4<ushort> ushort4;
static_assert(sizeof(ushort4) == 8, "bad size for ushort4");

typedef unsigned char   uchar;
static_assert(sizeof(uchar) == 1, "bad size for uchar");

typedef vec2<uchar> uchar2;
static_assert(sizeof(uchar2) == 2, "bad size for uchar2");

typedef vec4<uchar> uchar4;
static_assert(sizeof(uchar4) == 4, "bad size for uchar4");

/* ============================================================================================== */

template <typename ComponentType, int N>
struct pixel_vector { };

template <typename ComponentType>
struct pixel_vector<ComponentType,1> {
    typedef ComponentType   type;
};

template <typename ComponentType>
struct pixel_vector<ComponentType,2> {
    typedef vec2<ComponentType> type;
};

template <typename ComponentType>
struct pixel_vector<ComponentType,4> {
    typedef vec4<ComponentType> type;
};

/* ============================================================================================== */

template <typename T>
struct pixel_traits {};

template <>
struct pixel_traits<float> {
    typedef float   component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "float";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_FLOAT;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32_SFLOAT;

    static float translate(const float& pixel) { return pixel; }

    static float translate(half pixel) {
        return pixel;
    }

    static float translate(uchar pixel) {
        return (pixel / (float) std::numeric_limits<uchar>::max());
    }

    template <typename T>
    static float translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static float translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<float2> {
    typedef float   component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "float2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32G32_SFLOAT;

    template <typename T>
    static float2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static float2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static float2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<float4> {
    typedef float   component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "float4";

    static const int device_pixel_format = 1; // kDevicePixelFormat_BGRA_4444_32f
    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R32G32B32A32_SFLOAT;

    template <typename T>
    static float4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static float4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                component_t(0),
                component_t(0)
        };
    }

    template <typename T>
    static float4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0),
                component_t(0),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<half> {
    typedef half    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "half";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_HALF_FLOAT;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16_SFLOAT;

    static half translate(float pixel) { return half(pixel); }

    static half translate(const half& pixel) { return pixel; }

    static half translate(uchar pixel) {
        return translate(pixel / (float) std::numeric_limits<uchar>::max());
    }

    template <typename T>
    static half translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static half translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<half2> {
    typedef half    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "half2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_SFLOAT;

    template <typename T>
    static half2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static half2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static half2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<half4> {
    typedef half    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "half4";

    static const int device_pixel_format = 0; // kDevicePixelFormat_BGRA_4444_16f
    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_SFLOAT;

    template <typename T>
    static half4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static half4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                component_t(0),
                component_t(0)
        };
    }

    template <typename T>
    static half4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                component_t(0),
                component_t(0),
                component_t(0)
        };
    }
};

template <>
struct pixel_traits<ushort> {
    typedef ushort    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "ushort";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_UNSIGNED_INT16;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16_UINT;

    static ushort translate(float pixel) {
        return (ushort) (pixel * std::numeric_limits<ushort>::max());
    }

    static ushort translate(ushort pixel) { return pixel; }

    template <typename T>
    static ushort translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static ushort translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<ushort2> {
    typedef ushort    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "ushort2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_UINT;

    template <typename T>
    static ushort2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static ushort2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static ushort2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0
        };
    }
};

template <>
struct pixel_traits<ushort4> {
    typedef ushort    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "ushort4";

    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_UINT;

    template <typename T>
    static ushort4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static ushort4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                0,
                0
        };
    }

    template <typename T>
    static ushort4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0,
                0,
                0
        };
    }
};

template <>
struct pixel_traits<uchar> {
    typedef uchar    component_t;

    static constexpr const int num_components = 1;
    static constexpr const char* const type_name = "uchar";

    static const cl_channel_order cl_pixel_order = CL_R;
    static const cl_channel_type cl_pixel_type = CL_UNORM_INT8;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8_UNORM;

    static uchar translate(float pixel) {
        return (uchar) (pixel * std::numeric_limits<uchar>::max());
    }

    static uchar translate(half pixel) {
        return (uchar) (pixel * std::numeric_limits<uchar>::max());
    }

    static uchar translate(uchar pixel) { return pixel; }

    template <typename T>
    static uchar translate(const vec2<T>& pixel) {
        return translate(pixel.x);
    }

    template <typename T>
    static uchar translate(const vec4<T>& pixel) {
        return translate(pixel.x);
    }
};

template <>
struct pixel_traits<uchar2> {
    typedef uchar    component_t;

    static constexpr const int num_components = 2;
    static constexpr const char* const type_name = "uchar2";

    static const cl_channel_order cl_pixel_order = CL_RG;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8G8_UNORM;

    template <typename T>
    static uchar2 translate(const vec4<T>& pixel) {
        return translate((vec2<T>){ pixel.x, pixel.y });
    }

    template <typename T>
    static uchar2 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y)
        };
    }

    template <typename T>
    static uchar2 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0
        };
    }
};

template <>
struct pixel_traits<uchar4> {
    typedef uchar    component_t;

    static constexpr const int num_components = 4;
    static constexpr const char* const type_name = "uchar4";

    static const cl_channel_order cl_pixel_order = CL_RGBA;
    static const cl_channel_type cl_pixel_type = pixel_traits<component_t>::cl_pixel_type;
    static const VkFormat vk_pixel_type = VK_FORMAT_R8G8B8A8_UNORM;

    template <typename T>
    static uchar4 translate(const vec4<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                pixel_traits<component_t>::translate(pixel.z),
                pixel_traits<component_t>::translate(pixel.w)
        };
    }

    template <typename T>
    static uchar4 translate(const vec2<T>& pixel) {
        return {
                pixel_traits<component_t>::translate(pixel.x),
                pixel_traits<component_t>::translate(pixel.y),
                0,
                0
        };
    }

    template <typename T>
    static uchar4 translate(T pixel) {
        return {
                pixel_traits<component_t>::translate(pixel),
                0,
                0,
                0
        };
    }
};

/* ============================================================================================== */

VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT msgFlags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject,
                                       size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg,
                                       void *pUserData) {
    std::ostringstream message;

    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        message << "ERROR: ";
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        message << "WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        message << "PERFORMANCE WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        message << "INFO: ";
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        message << "DEBUG: ";
    }
    message << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

    std::cout << message.str() << std::endl;

    /*
     * false indicates that layer should not bail-out of an
     * API call that had validation failures. This may mean that the
     * app dies inside the driver due to invalid parameter(s).
     * That's what would happen without validation layers, so we'll
     * keep that behavior here.
     */
    return VK_FALSE;
}

/* ============================================================================================== */

void init_compute_queue_family_index(struct sample_info &info) {
    /* This routine simply finds a compute queue for a later vkCreateDevice.
     */
    bool found = false;
    for (unsigned int i = 0; i < info.queue_props.size(); i++) {
        if (info.queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            info.graphics_queue_family_index = i;
            found = true;
            break;
        }
    }
    assert(found);
}

void my_init_descriptor_pool(struct sample_info &info) {
    const VkDescriptorPoolSize type_count[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,    16 },
            { VK_DESCRIPTOR_TYPE_SAMPLER,           16 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,     16 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,     16 }
    };

    VkDescriptorPoolCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.maxSets = 64;
    createInfo.poolSizeCount = sizeof(type_count) / sizeof(type_count[0]);
    createInfo.pPoolSizes = type_count;
    createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    vulkan_utils::throwIfNotSuccess(vkCreateDescriptorPool(info.device,
                                                           &createInfo,
                                                           NULL,
                                                           &info.desc_pool),
                                    "vkCreateDescriptorPool");
}

VkSampler create_compatible_sampler(VkDevice device, int opencl_flags) {
    typedef std::pair<int,VkSamplerAddressMode> address_mode_map;
    const address_mode_map address_mode_translator[] = {
            { CLK_ADDRESS_NONE, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
            { CLK_ADDRESS_CLAMP, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER },
            { CLK_ADDRESS_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT }
    };

    const VkFilter filter = ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ?
                             VK_FILTER_LINEAR :
                             VK_FILTER_NEAREST);
    const VkBool32 unnormalizedCoordinates = ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_FALSE : VK_TRUE);

    const auto found_map = std::find_if(std::begin(address_mode_translator), std::end(address_mode_translator), [&opencl_flags](const address_mode_map& am) {
        return (am.first == (opencl_flags & CLK_ADDRESS_MASK));
    });
    const VkSamplerAddressMode addressMode = (found_map == std::end(address_mode_translator) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : found_map->second);

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter ;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = addressMode;
    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.unnormalizedCoordinates = unnormalizedCoordinates;

    VkSampler result = VK_NULL_HANDLE;
    vulkan_utils::throwIfNotSuccess(vkCreateSampler(device, &samplerCreateInfo, NULL, &result),
                                    "vkCreateSampler");

    return result;
}

/* ============================================================================================== */

void invoke_fill_kernel(const clspv_utils::kernel_module&   module,
                        const clspv_utils::kernel&          kernel,
                        const sample_info&                  info,
                        const std::vector<VkSampler>&       samplers,
                        VkBuffer                            dst_buffer,
                        int                                 pitch,
                        int                                 device_format,
                        int                                 offset_x,
                        int                                 offset_y,
                        int                                 width,
                        int                                 height,
                        const float4&                       color) {
    struct scalar_args {
        int     inPitch;        // offset 0
        int     inDeviceFormat; // DevicePixelFormat offset 4
        int     inOffsetX;      // offset 8
        int     inOffsetY;      // offset 12
        int     inWidth;        // offset 16
        int     inHeight;       // offset 20
        float4  inColor;        // offset 32
    };
    static_assert(0 == offsetof(scalar_args, inPitch), "inPitch offset incorrect");
    static_assert(4 == offsetof(scalar_args, inDeviceFormat), "inDeviceFormat offset incorrect");
    static_assert(8 == offsetof(scalar_args, inOffsetX), "inOffsetX offset incorrect");
    static_assert(12 == offsetof(scalar_args, inOffsetY), "inOffsetY offset incorrect");
    static_assert(16 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(20 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");
    static_assert(32 == offsetof(scalar_args, inColor), "inColor offset incorrect");

    const scalar_args scalars = {
            pitch,
            device_format,
            offset_x,
            offset_y,
            width,
            height,
            color
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (scalars.inWidth + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (scalars.inHeight + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addBufferArgument(dst_buffer);
    invocation.addPodArgument(scalars);
    invocation.run(info.graphics_queue, kernel, num_workgroups);
}

void invoke_copybuffertoimage_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkBuffer  src_buffer,
                                  VkImageView   dst_image,
                                  int src_offset,
                                  int src_pitch,
                                     cl_channel_order src_channel_order,
                                     cl_channel_type src_channel_type,
                                  bool swap_components,
                                  bool premultiply,
                                  int width,
                                  int height) {
    struct scalar_args {
        int inSrcOffset;        // offset 0
        int inSrcPitch;         // offset 4
        int inSrcChannelOrder;  // offset 8 -- cl_channel_order
        int inSrcChannelType;   // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inPremultiply;      // offset 20 -- bool
        int inWidth;            // offset 24
        int inHeight;           // offset 28
    };
    static_assert(0 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inSrcChannelOrder), "inSrcChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inSrcChannelType), "inSrcChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inPremultiply), "inPremultiply offset incorrect");
    static_assert(24 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(28 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            src_offset,
            src_pitch,
            src_channel_order,
            src_channel_type,
            (swap_components ? 1 : 0),
            (premultiply ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addBufferArgument(src_buffer);
    invocation.addWriteOnlyImageArgument(dst_image);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, kernel, num_workgroups);
}

void invoke_copyimagetobuffer_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkImageView src_image,
                                  VkBuffer dst_buffer,
                                  int dst_offset,
                                  int dst_pitch,
                                     cl_channel_order dst_channel_order,
                                     cl_channel_type dst_channel_type,
                                  bool swap_components,
                                  int width,
                                  int height) {
    struct scalar_args {
        int inDestOffset;       // offset 0
        int inDestPitch;        // offset 4
        int inDestChannelOrder; // offset 8 -- cl_channel_order
        int inDestChannelType;  // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inWidth;            // offset 20
        int inHeight;           // offset 24
    };
    static_assert(0 == offsetof(scalar_args, inDestOffset), "inDestOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inDestPitch), "inDestPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inDestChannelOrder), "inDestChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inDestChannelType), "inDestChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(24 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            dst_offset,
            width,
            dst_channel_order,
            dst_channel_type,
            (swap_components ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addReadOnlyImageArgument(src_image);
    invocation.addBufferArgument(dst_buffer);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, kernel, num_workgroups);
}

std::tuple<int,int,int> invoke_localsize_kernel(const clspv_utils::kernel_module&   module,
                                                const clspv_utils::kernel&          kernel,
                                                const sample_info&                  info,
                                                const std::vector<VkSampler>&       samplers) {
    struct scalar_args {
        int outWorkgroupX;  // offset 0
        int outWorkgroupY;  // offset 4
        int outWorkgroupZ;  // offset 8
    };
    static_assert(0 == offsetof(scalar_args, outWorkgroupX), "outWorkgroupX offset incorrect");
    static_assert(4 == offsetof(scalar_args, outWorkgroupY), "outWorkgroupY offset incorrect");
    static_assert(8 == offsetof(scalar_args, outWorkgroupZ), "outWorkgroupZ offset incorrect");

    vulkan_utils::buffer outArgs(info, sizeof(scalar_args));

    // The localsize kernel needs only a single workgroup with a single workitem
    const clspv_utils::WorkgroupDimensions num_workgroups(1, 1);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addBufferArgument(outArgs.buf);

    invocation.run(info.graphics_queue, kernel, num_workgroups);

    vulkan_utils::memory_map argMap(outArgs);
    auto outScalars = static_cast<const scalar_args*>(argMap.data);

    const auto result = std::make_tuple(outScalars->outWorkgroupX,
                                        outScalars->outWorkgroupY,
                                        outScalars->outWorkgroupZ);
    return result;
}


/* ============================================================================================== */

namespace test_utils {

    namespace {
        template<typename ExpectedPixelType, typename ObservedPixelType>
        struct pixel_promotion {
            static constexpr const int expected_vec_size = pixel_traits<ExpectedPixelType>::num_components;
            static constexpr const int observed_vec_size = pixel_traits<ObservedPixelType>::num_components;
            static constexpr const int vec_size = (expected_vec_size > observed_vec_size
                                                   ? observed_vec_size : expected_vec_size);

            typedef typename pixel_traits<ExpectedPixelType>::component_t expected_comp_type;
            typedef typename pixel_traits<ObservedPixelType>::component_t observed_comp_type;

            static constexpr const bool expected_is_smaller =
                    sizeof(expected_comp_type) < sizeof(observed_comp_type);
            typedef typename std::conditional<expected_is_smaller, expected_comp_type, observed_comp_type>::type smaller_comp_type;
            typedef typename std::conditional<!expected_is_smaller, expected_comp_type, observed_comp_type>::type larger_comp_type;

            static constexpr const bool smaller_is_floating = std::is_floating_point<smaller_comp_type>::value;
            typedef typename std::conditional<smaller_is_floating, smaller_comp_type, larger_comp_type>::type comp_type;

            typedef typename pixel_vector<comp_type, vec_size>::type promotion_type;
        };

        template<typename T>
        struct pixel_comparator {
        };

        template<>
        struct pixel_comparator<float> {
            static bool is_equal(float l, float r) {
                const int ulp = 2;
                return almost_equal(l, r, ulp);
            }
        };

        template<>
        struct pixel_comparator<half> {
            static bool is_equal(half l, half r) {
                const int ulp = 2;
                return almost_equal(l, r, ulp);
            }
        };

        template<>
        struct pixel_comparator<uchar> {
            static bool is_equal(uchar l, uchar r) {
                return pixel_comparator<float>::is_equal(pixel_traits<float>::translate(l),
                                                         pixel_traits<float>::translate(r));
            }
        };

        template<typename T>
        struct pixel_comparator<vec2<T> > {
            static bool is_equal(const vec2<T> &l, const vec2<T> &r) {
                return pixel_comparator<T>::is_equal(l.x, r.x)
                       && pixel_comparator<T>::is_equal(l.y, r.y);
            }
        };

        template<typename T>
        struct pixel_comparator<vec4<T> > {
            static bool is_equal(const vec4<T> &l, const vec4<T> &r) {
                return pixel_comparator<T>::is_equal(l.x, r.x)
                       && pixel_comparator<T>::is_equal(l.y, r.y)
                       && pixel_comparator<T>::is_equal(l.z, r.z)
                       && pixel_comparator<T>::is_equal(l.w, r.w);
            }
        };

        template<typename T>
        bool pixel_compare(const T &l, const T &r) {
            return pixel_comparator<T>::is_equal(l, r);
        }
    }

    typedef std::pair<int,int>  Results;

    int count_successes(Results r) { return r.first; }
    int count_failures(Results r) { return r.second; }

    const Results   no_result = std::make_pair(0, 0);
    const Results   success = std::make_pair(1, 0);
    const Results   failure = std::make_pair(0, 1);

    struct options {
        bool    logVerbose;
        bool    logIncorrect;
        bool    logCorrect;
    };

    typedef Results (*test_kernel_fn)(const clspv_utils::kernel_module& module,
                                      const clspv_utils::kernel&        kernel,
                                      const sample_info&                info,
                                      const std::vector<VkSampler>&     samplers,
                                      const options&                    opts);

    struct kernel_test_map {
        std::string                         entry;
        test_kernel_fn                      test;
        clspv_utils::WorkgroupDimensions    workgroupSize;
    };

    struct module_test_bundle {
        std::string                     name;
        std::vector<kernel_test_map>    kernelTests;
    };

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_result(ExpectedPixelType expected_pixel,
                      ObservedPixelType observed_pixel,
                      const char*       label,
                      int               row,
                      int               column,
                      const options&    opts) {
        typedef typename pixel_promotion<ExpectedPixelType, ObservedPixelType>::promotion_type promotion_type;

        auto expected = pixel_traits<promotion_type>::translate(expected_pixel);
        auto observed = pixel_traits<promotion_type>::translate(observed_pixel);

        auto t_expected = pixel_traits<float4>::translate(expected);
        auto t_observed = pixel_traits<float4>::translate(observed);

        const bool pixel_is_correct = pixel_compare(observed, expected);
        if (opts.logVerbose && ((pixel_is_correct && opts.logCorrect) || (!pixel_is_correct && opts.logIncorrect))) {
            const float4 log_expected = pixel_traits<float4>::translate(expected_pixel);
            const float4 log_observed = pixel_traits<float4>::translate(observed_pixel);

            LOGE("%s: %s pixel{row:%d, col%d} expected{x=%f, y=%f, z=%f, w=%f} observed{x=%f, y=%f, z=%f, w=%f}",
                 pixel_is_correct ? "CORRECT" : "INCORRECT",
                 label, row, column,
                 log_expected.x, log_expected.y, log_expected.z, log_expected.w,
                 log_observed.x, log_observed.y, log_observed.z, log_observed.w);
        }

        return pixel_is_correct;
    }

    template<typename ObservedPixelType, typename ExpectedPixelType>
    bool check_results(const ObservedPixelType* observed_pixels,
                       int                      width,
                       int                      height,
                       int                      pitch,
                       ExpectedPixelType        expected,
                       const char*              label,
                       const options&           opts) {
        unsigned int num_correct_pixels = 0;
        unsigned int num_incorrect_pixels = 0;

        auto row = observed_pixels;
        for (int r = 0; r < height; ++r, row += pitch) {
            auto p = row;
            for (int c = 0; c < width; ++c, ++p) {
                if (check_result(expected, *p, label, r, c, opts)) {
                    ++num_correct_pixels;
                } else {
                    ++num_incorrect_pixels;
                }
            }
        }

        if (opts.logVerbose) {
            LOGE("%s: Correct pixels=%d; Incorrect pixels=%d",
                 label, num_correct_pixels, num_incorrect_pixels);
        }

        return (0 == num_incorrect_pixels && 0 < num_correct_pixels);
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_results(const ExpectedPixelType* expected_pixels,
                       const ObservedPixelType* observed_pixels,
                       int                      width,
                       int                      height,
                       int                      pitch,
                       const char*              label,
                       const options&           opts) {
        unsigned int num_correct_pixels = 0;
        unsigned int num_incorrect_pixels = 0;

        auto expected_row = expected_pixels;
        auto observed_row = observed_pixels;
        for (int r = 0; r < height; ++r, expected_row += pitch, observed_row += pitch) {
            auto expected_p = expected_row;
            auto observed_p = observed_row;
            for (int c = 0; c < width; ++c, ++expected_p, ++observed_p) {
                if (check_result(*expected_p, *observed_p, label, r, c, opts)) {
                    ++num_correct_pixels;
                } else {
                    ++num_incorrect_pixels;
                }
            }
        }

        if (opts.logVerbose) {
            LOGE("%s: Correct pixels=%d; Incorrect pixels=%d", label, num_correct_pixels,
                 num_incorrect_pixels);
        }

        return (0 == num_incorrect_pixels && 0 < num_correct_pixels);
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    bool check_results(const vulkan_utils::device_memory&   expected,
                       const vulkan_utils::device_memory&   observed,
                       int                                  width,
                       int                                  height,
                       int                                  pitch,
                       const char*                          label,
                       const options&                       opts) {
        vulkan_utils::memory_map src_map(expected);
        vulkan_utils::memory_map dst_map(observed);
        auto src_pixels = static_cast<const ExpectedPixelType *>(src_map.data);
        auto dst_pixels = static_cast<const ObservedPixelType *>(dst_map.data);

        return check_results(src_pixels, dst_pixels, width, height, pitch, label, opts);
    }

    template<typename Fn>
    Results runInExceptionContext(const std::string& label, const std::string& stage, Fn f) {
        Results result = no_result;

        try {
            result = f();
        }
        catch(const vk::SystemError& e) {
            std::ostringstream os;
            os << label << '/' << stage << ": vk::SystemError : " << e.code() << " (" << e.code().message() << ')';
            LOGE("%s", os.str().c_str());
            result = failure;
        }
        catch(const std::system_error& e) {
            std::ostringstream os;
            os << label << '/' << stage << ": std::system_error : " << e.code() << " (" << e.code().message() << ')';
            LOGE("%s", os.str().c_str());
            result = failure;
        }
        catch(const std::exception& e) {
            std::ostringstream os;
            os << label << '/' << stage << ": std::exception : " << e.what();
            LOGE("%s", os.str().c_str());
            result = failure;
        }
        catch(...) {
            std::ostringstream os;
            os << label << '/' << stage << ": unknonwn error";
            LOGE("%s", os.str().c_str());
            result = failure;
        }

        return result;
    }

    template<typename ObservedPixelType>
    bool check_results(const vulkan_utils::device_memory&   observed,
                       int                                  width,
                       int                                  height,
                       int                                  pitch,
                       const float4&                        expected,
                       const char*                          label,
                       const options&                       opts) {
        vulkan_utils::memory_map map(observed);
        auto pixels = static_cast<const ObservedPixelType *>(map.data);
        return check_results(pixels, width, height, pitch, expected, label, opts);
    }

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                                   const clspv_utils::kernel&           kernel,
                                   test_utils::test_kernel_fn           testFn,
                                   const sample_info&                   info,
                                   const std::vector<VkSampler>&        samplers,
                                   const options&                       opts) {
        Results result = no_result;

        if (testFn) {
            const std::string label = module.getName() + "/" + kernel.getEntryPoint();
            result = runInExceptionContext(label,
                                  "invoking kernel",
                                  [&]() {
                                      return testFn(module, kernel, info, samplers, opts);
                                  });

            if ((count_successes(result) > 0 && opts.logCorrect) ||
                    (count_failures(result) > 0 && opts.logIncorrect)) {
                LOGE("%s: Successes=%d Failures=%d",
                     label.c_str(),
                     count_successes(result), count_failures(result));
            }
        }

        return result;
    }

    Results test_kernel_invocation(const clspv_utils::kernel_module&    module,
                                   const clspv_utils::kernel&           kernel,
                                   const test_utils::test_kernel_fn*    first,
                                   const test_utils::test_kernel_fn*    last,
                                   const sample_info&                   info,
                                   const std::vector<VkSampler>&        samplers,
                                   const options&                       opts) {
        Results result = no_result;

        for (; first != last; ++first) {
            result += test_kernel_invocation(module, kernel, *first, info, samplers, opts);
        }

        return result;
    }

    Results test_kernel(const clspv_utils::kernel_module&       module,
                        const std::string&                      entryPoint,
                        test_utils::test_kernel_fn              testFn,
                        const clspv_utils::WorkgroupDimensions& numWorkgroups,
                        const sample_info&                      info,
                        const std::vector<VkSampler>&           samplers,
                        const options&                          opts) {
        return runInExceptionContext(module.getName() + "/" + entryPoint,
                                     "compiling kernel",
                                     [&]() {
                                         Results results = no_result;

                                         clspv_utils::kernel kernel(info.device, module,
                                                                    entryPoint, numWorkgroups);
                                         results += success;
                                         results += test_kernel_invocation(module,
                                                                           kernel,
                                                                           testFn,
                                                                           info,
                                                                           samplers,
                                                                           opts);

                                         return results;
                                     });
    }

    Results test_module(const std::string&                  moduleName,
                        const std::vector<kernel_test_map>& kernelTests,
                        const sample_info&                  info,
                        const std::vector<VkSampler>&       samplers,
                        const options&                      opts) {
        return runInExceptionContext(moduleName, "loading module", [&]() {
            Results result = no_result;

            clspv_utils::kernel_module module(info.device, info.desc_pool, moduleName);
            result += success;

            std::vector<std::string> entryPoints(module.getEntryPoints());
            int num_success_kernels = 0;
            for (auto ep : entryPoints) {
                const auto epTest = std::find_if(kernelTests.begin(), kernelTests.end(),
                                                 [&ep](const kernel_test_map& ktm) {
                                                     return ktm.entry == ep;
                                                 });

                clspv_utils::WorkgroupDimensions num_workgroups;
                if (epTest != kernelTests.end()) {
                    num_workgroups = epTest->workgroupSize;
                }

                if (0 == num_workgroups.x && 0 == num_workgroups.y) {
                    // WorkgroupDimensions(0, 0) is a sentinel to skip this kernel entirely
                    LOGI("%s/%s: Skipping kernel", moduleName.c_str(), ep.c_str());
                }
                else {

                    auto kernel_result = test_kernel(
                            module,
                            ep,
                            epTest == kernelTests.end() ? nullptr : epTest->test,
                            num_workgroups,
                            info,
                            samplers,
                            opts);
                    result += kernel_result;

                    if (count_failures(kernel_result) == 0 && count_successes(kernel_result) > 0) {
                        ++num_success_kernels;
                    }
                }
            }

            LOGI("%s: %d/%d kernel successes",
                 moduleName.c_str(),
                 num_success_kernels,
                 (int)entryPoints.size());

            return result;
        });
    }
} // namespace test_utils

/* ============================================================================================== */

test_utils::Results test_readlocalsize(const clspv_utils::kernel_module& module,
                                       const clspv_utils::kernel&        kernel,
                                       const sample_info&                info,
                                       const std::vector<VkSampler>&     samplers,
                                       const test_utils::options&        opts) {
    const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

    const auto observed = invoke_localsize_kernel(module, kernel, info, samplers);

    const bool success = (expected.x == std::get<0>(observed) &&
                          expected.y == std::get<1>(observed) &&
                          1 == std::get<2>(observed));

    if (opts.logVerbose && ((success && opts.logCorrect) || (!success && opts.logIncorrect))) {
        const std::string label = module.getName() + "/" + kernel.getEntryPoint();
        LOGE("%s: %s workgroup_size expected{x=%d, y=%d, z=1} observed{x=%d, y=%d, z=%d}",
             success ? "CORRECT" : "INCORRECT",
             label.c_str(),
             expected.x, expected.y,
             std::get<0>(observed), std::get<1>(observed), std::get<2>(observed));
    }

    return (success ? std::make_pair(1, 0) : std::make_pair(0, 1));
};

template <typename PixelType>
test_utils::Results test_fill(const clspv_utils::kernel_module&     module,
                              const clspv_utils::kernel&            kernel,
                              const sample_info&                    info,
                              const std::vector<VkSampler>&         samplers,
                              const test_utils::options&            opts) {
    const std::string typeLabel = pixel_traits<PixelType>::type_name;

    std::string testLabel = "fills.spv/FillWithColorKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;
    const float4 color = { 0.25f, 0.50f, 0.75f, 1.0f };

    // allocate image buffer
    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(PixelType);
    vulkan_utils::buffer dst_buffer(info, buffer_size);

    {
        const PixelType src_value = pixel_traits<PixelType>::translate((float4){ 0.0f, 0.0f, 0.0f, 0.0f });

        vulkan_utils::memory_map dst_map(dst_buffer);
        auto dst_data = static_cast<PixelType*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), src_value);
    }

    invoke_fill_kernel(module,
                       kernel,
                       info,
                       samplers,
                       dst_buffer.buf, // dst_buffer
                       buffer_width,   // pitch
                       pixel_traits<PixelType>::device_pixel_format, // device_format
                       0, 0, // offset_x, offset_y
                       buffer_width, buffer_height, // width, height
                       color); // color

    const bool success = test_utils::check_results<PixelType>(dst_buffer.mem,
                                                  buffer_width, buffer_height,
                                                  buffer_width,
                                                  color,
                                                  testLabel.c_str(),
                                                  opts);

    dst_buffer.reset();

    return (success ? test_utils::success : std::make_pair(0, 1));
}

test_utils::Results test_fill_series(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     const std::vector<VkSampler>&      samplers,
                                     const test_utils::options&         opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_fill<float4>,
            test_fill<half4>,
    };

    return test_utils::test_kernel_invocation(module,
                                              kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copytoimage(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     const std::vector<VkSampler>&      samplers,
                                     const test_utils::options&         opts) {
    std::string typeLabel = pixel_traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixel_traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyBufferToImageKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  src_buffer(info, buffer_size);
    vulkan_utils::image   dstImage(info, buffer_width, buffer_height, pixel_traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixel_traits<BufferPixelType>::translate((float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(src_buffer);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixel_traits<ImagePixelType>::translate((float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dstImage);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    invoke_copybuffertoimage_kernel(module, kernel,
                                    info,
                                    samplers,
                                    src_buffer.buf,
                                    dstImage.view,
                                    0,
                                    buffer_width,
                                    pixel_traits<BufferPixelType>::cl_pixel_order,
                                    pixel_traits<BufferPixelType>::cl_pixel_type,
                                    false,
                                    false,
                                    buffer_width,
                                    buffer_height);

    const bool success = test_utils::check_results<BufferPixelType, ImagePixelType>(src_buffer.mem, dstImage.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        opts);

    dstImage.reset();
    src_buffer.reset();

    return (success ? test_utils::success : std::make_pair(0, 1));
}

template <typename ImagePixelType>
test_utils::Results test_copytoimage_series(const clspv_utils::kernel_module&   module,
                                            const clspv_utils::kernel&          kernel,
                                            const sample_info&                  info,
                                            const std::vector<VkSampler>&       samplers,
                                            const test_utils::options&          opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copytoimage<uchar, ImagePixelType>,
            test_copytoimage<uchar4, ImagePixelType>,
            test_copytoimage<half, ImagePixelType>,
            test_copytoimage<half4, ImagePixelType>,
            test_copytoimage<float, ImagePixelType>,
            test_copytoimage<float2, ImagePixelType>,
            test_copytoimage<float4, ImagePixelType>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

test_utils::Results test_copytoimage_matrix(const clspv_utils::kernel_module&   module,
                                            const clspv_utils::kernel&          kernel,
                                            const sample_info&                  info,
                                            const std::vector<VkSampler>&       samplers,
                                            const test_utils::options&          opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copytoimage_series<float4>,
            test_copytoimage_series<half4>,
            test_copytoimage_series<uchar4>,
            test_copytoimage_series<float2>,
            test_copytoimage_series<half2>,
            test_copytoimage_series<uchar2>,
            test_copytoimage_series<float>,
            test_copytoimage_series<half>,
            test_copytoimage_series<uchar>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copyfromimage(const clspv_utils::kernel_module&    module,
                                       const clspv_utils::kernel&           kernel,
                                       const sample_info&                   info,
                                       const std::vector<VkSampler>&        samplers,
                                       const test_utils::options&           opts) {
    std::string typeLabel = pixel_traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixel_traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyImageToBufferKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  dst_buffer(info, buffer_size);
    vulkan_utils::image   srcImage(info, buffer_width, buffer_height, pixel_traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixel_traits<ImagePixelType>::translate((float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(srcImage);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixel_traits<BufferPixelType>::translate((float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dst_buffer);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    invoke_copyimagetobuffer_kernel(module, kernel,
                                    info,
                                    samplers,
                                    srcImage.view,
                                    dst_buffer.buf,
                                    0,
                                    buffer_width,
                                    pixel_traits<BufferPixelType>::cl_pixel_order,
                                    pixel_traits<BufferPixelType>::cl_pixel_type,
                                    false,
                                    buffer_width,
                                    buffer_height);

    const bool success = test_utils::check_results<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        opts);

    srcImage.reset();
    dst_buffer.reset();

    return (success ? test_utils::success : test_utils::failure);
}

template <typename ImagePixelType>
test_utils::Results test_copyfromimage_series(const clspv_utils::kernel_module&     module,
                                              const clspv_utils::kernel&            kernel,
                                              const sample_info&                    info,
                                              const std::vector<VkSampler>&         samplers,
                                              const test_utils::options&            opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copyfromimage<uchar, ImagePixelType>,
            test_copyfromimage<uchar4, ImagePixelType>,
            test_copyfromimage<half, ImagePixelType>,
            test_copyfromimage<half4, ImagePixelType>,
            test_copyfromimage<float, ImagePixelType>,
            test_copyfromimage<float2, ImagePixelType>,
            test_copyfromimage<float4, ImagePixelType>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

test_utils::Results test_copyfromimage_matrix(const clspv_utils::kernel_module&     module,
                                              const clspv_utils::kernel&            kernel,
                                              const sample_info&                    info,
                                              const std::vector<VkSampler>&         samplers,
                                              const test_utils::options&            opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copyfromimage_series<float4>,
            test_copyfromimage_series<half4>,
            test_copyfromimage_series<uchar4>,
            test_copyfromimage_series<float2>,
            test_copyfromimage_series<half2>,
            test_copyfromimage_series<uchar2>,
            test_copyfromimage_series<float>,
            test_copyfromimage_series<half>,
            test_copyfromimage_series<uchar>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}


/* ============================================================================================== */

const test_utils::module_test_bundle module_tests[] = {
        {
                "clspv_tests/localsize",
                {
                        {"ReadLocalSize", test_readlocalsize}
                },
        },
        {
                "clspv_tests/Fills",
                {
                        { "FillWithColorKernel", test_fill_series, { 32, 32 } }
                }
        },
        {
                "clspv_tests/Memory",
                {
                        { "CopyBufferToImageKernel", test_copytoimage_matrix, { 32, 32 } },
                        { "CopyImageToBufferKernel", test_copyfromimage_matrix, { 32, 32 } }
                }
        },
};

std::vector<test_utils::module_test_bundle> read_loadmodule_file() {
    std::vector<test_utils::module_test_bundle> result;

    std::unique_ptr<std::FILE, decltype(&std::fclose)> spvmap_file(AndroidFopen("loadmodules.txt", "r"),
                                                                   &std::fclose);
    if (spvmap_file) {
        std::fseek(spvmap_file.get(), 0, SEEK_END);
        std::string buffer(std::ftell(spvmap_file.get()), ' ');
        std::fseek(spvmap_file.get(), 0, SEEK_SET);
        std::fread(&buffer.front(), 1, buffer.length(), spvmap_file.get());
        spvmap_file.reset();

        std::istringstream in(buffer);
        while (!in.eof()) {
            std::string line;
            std::getline(in, line);

            std::istringstream in_line(line);

            std::string op;
            in_line >> op;
            if (op.empty() || op[0] == '#') {
                // line is either blank or a comment, skip it
            }
            else if (op == "+") {
                // add module to list of modules to load
                std::string moduleName;
                in_line >> moduleName;
                result.push_back({ moduleName });
            }
            else if (op == "-") {
                // skip kernel in module
                std::string moduleName;
                in_line >> moduleName;

                auto mod_map = std::find_if(result.begin(),
                                            result.end(),
                                            [&moduleName](const test_utils::module_test_bundle& mb) {
                                                return mb.name == moduleName;
                                            });
                if (mod_map != result.end()) {
                    std::string entryPoint;
                    in_line >> entryPoint;

                    mod_map->kernelTests.push_back({ entryPoint, nullptr, clspv_utils::WorkgroupDimensions(0, 0) });
                }
                else {
                    LOGE("read_loadmodule_file: cannot find module '%s' from command '%s'",
                         moduleName.c_str(),
                         line.c_str());
                }
            }
            else {
                LOGE("read_loadmodule_file: ignoring ill-formed line '%s'", line.c_str());
            }
        }
    }

    return result;
}

test_utils::Results run_all_tests(const sample_info& info, const std::vector<VkSampler>& samplers) {
    const test_utils::options opts = {
            false,  // logVerbose
            true,   // logIncorrect
            false   // logCorrect
    };

    auto test_results = test_utils::no_result;

    for (auto m : module_tests) {
        test_results += test_utils::test_module(m.name, m.kernelTests, info, samplers, opts);
    }

    auto loadmodule_tests = read_loadmodule_file();
    for (auto m : loadmodule_tests) {
        test_results += test_utils::test_module(m.name, m.kernelTests, info, samplers, opts);
    }

    return test_results;
}

/* ============================================================================================== */

int sample_main(int argc, char *argv[]) {
    struct sample_info info = {};
    init_global_layer_properties(info);

    /* Use standard_validation meta layer that enables all
     * recommended validation layers
     */
    info.instance_layer_names.push_back("VK_LAYER_LUNARG_standard_validation");
    if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
        /* If standard validation is not present, search instead for the
         * individual layers that make it up, in the correct order.
         */
        info.instance_layer_names.clear();
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_threading");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_parameter_validation");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_object_tracker");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_core_validation");
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_unique_objects");

        if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
            LOGE("Cannot find validation layers! :(");
            info.instance_layer_names.clear();
        }
    }

    info.instance_extension_names.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    init_instance(info, "vulkansamples_device");

    init_enumerate_device(info);
    init_compute_queue_family_index(info);

    // The clspv solution we're using requires two Vulkan extensions to be enabled.
    info.device_extension_names.push_back("VK_KHR_storage_buffer_storage_class");
    info.device_extension_names.push_back("VK_KHR_variable_pointers");
    init_device(info);
    init_device_queue(info);

    init_debug_report_callback(info, dbgFunc);

    init_command_pool(info);
    my_init_descriptor_pool(info);

    // This sample presumes that all OpenCL C kernels were compiled with the same samplermap file,
    // whose contents and order are statically known to the application. Thus, the app can create
    // a set of compatible samplers thusly.
    const int sampler_flags[] = {
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_NONE            | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_TRUE
    };
    std::vector<VkSampler> samplers;
    std::transform(std::begin(sampler_flags), std::end(sampler_flags),
                   std::back_inserter(samplers),
                   std::bind(create_compatible_sampler, info.device, std::placeholders::_1));


    const auto test_results = run_all_tests(info, samplers);

    //
    // Clean up
    //

    std::for_each(samplers.begin(), samplers.end(), [&info] (VkSampler s) { vkDestroySampler(info.device, s, nullptr); });

    // Cannot use the shader module desctruction built into the sampel framework because it is too
    // tightly tied to the graphics pipeline (e.g. hard-coding the number and type of shaders).

    destroy_descriptor_pool(info);
    destroy_command_pool(info);
    destroy_debug_report_callback(info);
    destroy_device(info);
    destroy_instance(info);

    LOGI("Complete! %d tests passed. %d tests failed",
         test_utils::count_successes(test_results),
         test_utils::count_failures(test_results));

    return 0;
}

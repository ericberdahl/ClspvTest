//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_PIXELS_HPP
#define CLSPVTEST_PIXELS_HPP

#include "gpu_types.hpp"
#include "opencl_types.hpp"

#include <vulkan/vulkan.h>

#include <iomanip>
#include <limits>
#include <sstream>

namespace pixels {
    template<typename ComponentType, int N>
    struct vector {
    };

    template<typename ComponentType>
    struct vector<ComponentType, 1> {
        typedef ComponentType type;
    };

    template<typename ComponentType>
    struct vector<ComponentType, 2> {
        typedef gpu_types::vec2<ComponentType> type;
    };

    template<typename ComponentType>
    struct vector<ComponentType, 4> {
        typedef gpu_types::vec4<ComponentType> type;
    };

/* ============================================================================================== */

    template<typename T>
    struct traits {
    };

    template<>
    struct traits<std::int32_t> {
        typedef std::int32_t component_t;
        typedef std::int32_t pixel_t;

        static constexpr const int num_components = 1;
        static constexpr const char *const type_name = "int32_t";

        static const cl_channel_order cl_pixel_order = CL_R;
        static const cl_channel_type cl_pixel_type = CL_SIGNED_INT32;
        static const VkFormat vk_pixel_type = VK_FORMAT_R32_SINT;

        static pixel_t translate(float pixel) {
            return (pixel_t) (pixel * std::numeric_limits<component_t>::max());
        }

        static pixel_t translate(component_t pixel) { return pixel; }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return translate(pixel.x);
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate(pixel.x);
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<float> {
        typedef float component_t;
        typedef float pixel_t;

        static constexpr const int num_components = 1;
        static constexpr const char *const type_name = "float";

        static const cl_channel_order cl_pixel_order = CL_R;
        static const cl_channel_type cl_pixel_type = CL_FLOAT;
        static const VkFormat vk_pixel_type = VK_FORMAT_R32_SFLOAT;

        static pixel_t translate(component_t pixel) { return pixel; }

        static pixel_t translate(gpu_types::half pixel) {
            return pixel;
        }

        static pixel_t translate(gpu_types::uchar pixel) {
            return (pixel / (float) std::numeric_limits<gpu_types::uchar>::max());
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return translate(pixel.x);
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate(pixel.x);
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::float2> {
        typedef float component_t;
        typedef gpu_types::float2 pixel_t;

        static constexpr const int num_components = 2;
        static constexpr const char *const type_name = "float2";

        static const cl_channel_order cl_pixel_order = CL_RG;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R32G32_SFLOAT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate((gpu_types::vec2<T>) {pixel.x, pixel.y});
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    component_t(0)
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel.x << ", " << pixel.y << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::float4> {
        typedef float component_t;
        typedef gpu_types::float4 pixel_t;

        static constexpr const int num_components = 4;
        static constexpr const char *const type_name = "float4";

        static const int device_pixel_format = 1; // kDevicePixelFormat_BGRA_4444_32f
        static const cl_channel_order cl_pixel_order = CL_RGBA;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R32G32B32A32_SFLOAT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    traits<component_t>::translate(pixel.z),
                    traits<component_t>::translate(pixel.w)
            };
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    component_t(0),
                    component_t(0)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    component_t(0),
                    component_t(0),
                    component_t(0)
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel.x << ", " << pixel.y << ", " << pixel.z << ", " << pixel.w << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::half> {
        typedef gpu_types::half component_t;
        typedef gpu_types::half pixel_t;

        static constexpr const int num_components = 1;
        static constexpr const char *const type_name = "half";

        static const cl_channel_order cl_pixel_order = CL_R;
        static const cl_channel_type cl_pixel_type = CL_HALF_FLOAT;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16_SFLOAT;

        static pixel_t translate(float pixel) { return pixel_t(pixel); }

        static pixel_t translate(const component_t &pixel) { return pixel; }

        static pixel_t translate(gpu_types::uchar pixel) {
            return translate(pixel / (float) std::numeric_limits<gpu_types::uchar>::max());
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return translate(pixel.x);
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate(pixel.x);
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::half2> {
        typedef gpu_types::half component_t;
        typedef gpu_types::half2 pixel_t;

        static constexpr const int num_components = 2;
        static constexpr const char *const type_name = "half2";

        static const cl_channel_order cl_pixel_order = CL_RG;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_SFLOAT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate((gpu_types::vec2<T>) {pixel.x, pixel.y});
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    component_t(0)
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel.x << ", " << pixel.y << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::half4> {
        typedef gpu_types::half component_t;
        typedef gpu_types::half4 pixel_t;

        static constexpr const int num_components = 4;
        static constexpr const char *const type_name = "half4";

        static const int device_pixel_format = 0; // kDevicePixelFormat_BGRA_4444_16f
        static const cl_channel_order cl_pixel_order = CL_RGBA;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_SFLOAT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    traits<component_t>::translate(pixel.z),
                    traits<component_t>::translate(pixel.w)
            };
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    component_t(0),
                    component_t(0)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    component_t(0),
                    component_t(0),
                    component_t(0)
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << "{ " << pixel.x << ", " << pixel.y << ", " << pixel.z << ", " << pixel.w << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::ushort> {
        typedef gpu_types::ushort component_t;
        typedef gpu_types::ushort pixel_t;

        static constexpr const int num_components = 1;
        static constexpr const char *const type_name = "ushort";

        static const cl_channel_order cl_pixel_order = CL_R;
        static const cl_channel_type cl_pixel_type = CL_UNSIGNED_INT16;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16_UINT;

        static pixel_t translate(float pixel) {
            return (pixel_t) (pixel * std::numeric_limits<component_t>::max());
        }

        static pixel_t translate(component_t pixel) { return pixel; }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return translate(pixel.x);
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate(pixel.x);
        }
    };

    template<>
    struct traits<gpu_types::ushort2> {
        typedef gpu_types::ushort component_t;
        typedef gpu_types::ushort2 pixel_t;

        static constexpr const int num_components = 2;
        static constexpr const char *const type_name = "ushort2";

        static const cl_channel_order cl_pixel_order = CL_RG;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16G16_UINT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate((gpu_types::vec2<T>) {pixel.x, pixel.y});
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    0
            };
        }
    };

    template<>
    struct traits<gpu_types::ushort4> {
        typedef gpu_types::ushort component_t;
        typedef gpu_types::ushort4 pixel_t;

        static constexpr const int num_components = 4;
        static constexpr const char *const type_name = "ushort4";

        static const cl_channel_order cl_pixel_order = CL_RGBA;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R16G16B16A16_UINT;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    traits<component_t>::translate(pixel.z),
                    traits<component_t>::translate(pixel.w)
            };
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    0,
                    0
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    0,
                    0,
                    0
            };
        }
    };

    template<>
    struct traits<gpu_types::uchar> {
        typedef gpu_types::uchar component_t;
        typedef gpu_types::uchar pixel_t;

        static constexpr const int num_components = 1;
        static constexpr const char *const type_name = "uchar";

        static const cl_channel_order cl_pixel_order = CL_R;
        static const cl_channel_type cl_pixel_type = CL_UNORM_INT8;
        static const VkFormat vk_pixel_type = VK_FORMAT_R8_UNORM;

        static pixel_t translate(float pixel) {
            return (pixel_t) round(pixel * std::numeric_limits<component_t>::max());
        }

        static pixel_t translate(gpu_types::half pixel) {
            return (pixel_t) round(pixel * std::numeric_limits<component_t>::max());
        }

        static pixel_t translate(component_t pixel) { return pixel; }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return translate(pixel.x);
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate(pixel.x);
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << std::showbase << std::hex << std::setw(2) << "{ " << (unsigned int)pixel << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::uchar2> {
        typedef gpu_types::uchar component_t;
        typedef gpu_types::uchar2 pixel_t;

        static constexpr const int num_components = 2;
        static constexpr const char *const type_name = "uchar2";

        static const cl_channel_order cl_pixel_order = CL_RG;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R8G8_UNORM;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return translate((gpu_types::vec2<T>) {pixel.x, pixel.y});
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y)
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    0
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << std::showbase << std::hex << std::setw(2) << "{ " << (unsigned int)pixel.x << ", " << (unsigned int)pixel.y << ", " << " }";
            return stream.str();
        }
    };

    template<>
    struct traits<gpu_types::uchar4> {
        typedef gpu_types::uchar component_t;
        typedef gpu_types::uchar4 pixel_t;

        static constexpr const int num_components = 4;
        static constexpr const char *const type_name = "uchar4";

        static const cl_channel_order cl_pixel_order = CL_RGBA;
        static const cl_channel_type cl_pixel_type = traits<component_t>::cl_pixel_type;
        static const VkFormat vk_pixel_type = VK_FORMAT_R8G8B8A8_UNORM;

        template<typename T>
        static pixel_t translate(const gpu_types::vec4<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    traits<component_t>::translate(pixel.z),
                    traits<component_t>::translate(pixel.w)
            };
        }

        template<typename T>
        static pixel_t translate(const gpu_types::vec2<T> &pixel) {
            return {
                    traits<component_t>::translate(pixel.x),
                    traits<component_t>::translate(pixel.y),
                    0,
                    0
            };
        }

        template<typename T>
        static pixel_t translate(T pixel) {
            return {
                    traits<component_t>::translate(pixel),
                    0,
                    0,
                    0
            };
        }

        static std::string toString(pixel_t pixel) {
            std::ostringstream stream;
            stream << std::showbase << std::hex << std::setw(2) << "{ " << (unsigned int)pixel.x << ", " << (unsigned int)pixel.y << ", " << (unsigned int)pixel.z << ", " << (unsigned int)pixel.w << " }";
            return stream.str();
        }
    };
}

#endif //CLSPVTEST_PIXELS_HPP

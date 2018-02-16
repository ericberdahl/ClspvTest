//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_FILL_KERNEL_HPP
#define CLSPVTEST_FILL_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace fill_kernel {

    void invoke(const clspv_utils::kernel_module &module,
                const clspv_utils::kernel &kernel,
                const sample_info &info,
                vk::ArrayProxy<const vk::Sampler> samplers,
                vk::Buffer dst_buffer,
                int pitch,
                int device_format,
                int offset_x,
                int offset_y,
                int width,
                int height,
                const gpu_types::float4 &color);

    void test_series(const clspv_utils::kernel_module&   module,
                     const clspv_utils::kernel&          kernel,
                     const sample_info&                  info,
                     vk::ArrayProxy<const vk::Sampler>   samplers,
                     test_utils::InvocationResultSet&    resultSet);

    template <typename PixelType>
    void test(const clspv_utils::kernel_module&      module,
              const clspv_utils::kernel&             kernel,
              const sample_info&                     info,
              vk::ArrayProxy<const vk::Sampler>      samplers,
              test_utils::InvocationResultSet&       resultSet) {
        test_utils::InvocationResult invocationResult;
        invocationResult.mVariation = pixels::traits<PixelType>::type_name;

        const int buffer_height = 64;
        const int buffer_width = 64;
        const gpu_types::float4 color = { 0.25f, 0.50f, 0.75f, 1.0f };

        // allocate image buffer
        const std::size_t buffer_size = buffer_width * buffer_height * sizeof(PixelType);
        vulkan_utils::buffer dst_buffer(info, buffer_size);

        {
            const PixelType src_value = pixels::traits<PixelType>::translate((gpu_types::float4){ 0.0f, 0.0f, 0.0f, 0.0f });

            vulkan_utils::memory_map dst_map(dst_buffer);
            auto dst_data = static_cast<PixelType*>(dst_map.map());
            std::fill(dst_data, dst_data + (buffer_width * buffer_height), src_value);
        }

        invoke(module,
               kernel,
               info,
               samplers,
               *dst_buffer.buf, // dst_buffer
               buffer_width,   // pitch
               pixels::traits<PixelType>::device_pixel_format, // device_format
               0, 0, // offset_x, offset_y
               buffer_width, buffer_height, // width, height
               color); // color

        test_utils::check_results<PixelType>(dst_buffer.mem,
                                             buffer_width, buffer_height,
                                             buffer_width,
                                             color,
                                             invocationResult);

        resultSet.push_back(invocationResult);
    }
}

#endif //CLSPVTEST_FILL_KERNEL_HPP

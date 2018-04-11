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

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel_module&    module,
           const clspv_utils::kernel&           kernel,
           const sample_info&                   info,
           vk::ArrayProxy<const vk::Sampler>    samplers,
           vk::Buffer                           dst_buffer,
           int                                  pitch,
           int                                  device_format,
           int                                  offset_x,
           int                                  offset_y,
           int                                  width,
           int                                  height,
           const gpu_types::float4&             color);

    void test_series(const clspv_utils::kernel_module&   module,
                     const clspv_utils::kernel&          kernel,
                     const sample_info&                  info,
                     vk::ArrayProxy<const vk::Sampler>   samplers,
                     const std::vector<std::string>&     args,
                     test_utils::InvocationResultSet&    resultSet);

    template <typename PixelType>
    void test(const clspv_utils::kernel_module&      module,
              const clspv_utils::kernel&             kernel,
              const sample_info&                     info,
              vk::ArrayProxy<const vk::Sampler>      samplers,
              const std::vector<std::string>&        args,
              test_utils::InvocationResultSet&       resultSet) {
        test_utils::InvocationResult invocationResult;

        int buffer_height = 64;
        int buffer_width = 64;
        const gpu_types::float4 color = { 0.25f, 0.50f, 0.75f, 1.0f };

        for (auto arg = args.begin(); arg != args.end(); arg = std::next(arg)) {
            if (*arg == "-w") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                buffer_width = std::atoi(arg->c_str());
            }
            else if (*arg == "-h") {
                arg = std::next(arg);
                if (arg == args.end()) throw std::runtime_error("badly formed arguments to fill test");
                buffer_height = std::atoi(arg->c_str());
            }
        }

        std::ostringstream os;
        os << "<dst:" << pixels::traits<PixelType>::type_name << " w:" << buffer_width << " h:" << buffer_height << ">";
        invocationResult.mVariation = os.str();

        // allocate image buffer
        const std::size_t buffer_size = buffer_width * buffer_height * sizeof(PixelType);
        vulkan_utils::storage_buffer dst_buffer(info, buffer_size);

        const PixelType src_value = pixels::traits<PixelType>::translate((gpu_types::float4){ 0.0f, 0.0f, 0.0f, 0.0f });
        vulkan_utils::fillDeviceMemory(dst_buffer.mem, buffer_width * buffer_height, src_value);

        invocationResult.mExecutionTime = invoke(module,
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

        resultSet.push_back(std::move(invocationResult));
    }
}

#endif //CLSPVTEST_FILL_KERNEL_HPP

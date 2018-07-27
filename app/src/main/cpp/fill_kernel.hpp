//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_FILL_KERNEL_HPP
#define CLSPVTEST_FILL_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace fill_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              pitch,
           int                              device_format,
           int                              offset_x,
           int                              offset_y,
           int                              width,
           int                              height,
           const gpu_types::float4&         color);

    void test_series(clspv_utils::kernel&               kernel,
                     const std::vector<std::string>&    args,
                     bool                               verbose,
                     test_utils::InvocationResultSet&   resultSet);

    template <typename PixelType>
    void test(clspv_utils::kernel&              kernel,
              const std::vector<std::string>&   args,
              bool                              verbose,
              test_utils::InvocationResultSet&  resultSet) {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

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
        vulkan_utils::storage_buffer dst_buffer(device.mDevice, device.mMemoryProperties, buffer_size);

        const PixelType src_value = pixels::traits<PixelType>::translate((gpu_types::float4){ 0.0f, 0.0f, 0.0f, 0.0f });
        auto dstBufferMap = dst_buffer.map<PixelType>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + (buffer_width * buffer_height), src_value);
        dstBufferMap.reset();

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dst_buffer, // dst_buffer
                                                 buffer_width,   // pitch
                                                 pixels::traits<PixelType>::device_pixel_format, // device_format
                                                 0, 0, // offset_x, offset_y
                                                 buffer_width, buffer_height, // width, height
                                                 color); // color

        dstBufferMap = dst_buffer.map<PixelType>();
        test_utils::check_results(dstBufferMap.get(),
                                  buffer_width, buffer_height, 1,
                                  buffer_width,
                                  color,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

#endif //CLSPVTEST_FILL_KERNEL_HPP

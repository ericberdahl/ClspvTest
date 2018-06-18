//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP
#define CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace copybuffertoimage_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    src_buffer,
           vulkan_utils::image&             dst_image,
           int                              src_offset,
           int                              src_pitch,
           cl_channel_order                 src_channel_order,
           cl_channel_type                  src_channel_type,
           bool                             swap_components,
           bool                             premultiply,
           int                              width,
           int                              height);

    template <typename BufferPixelType, typename ImagePixelType>
    void test(clspv_utils::kernel&              kernel,
              const std::vector<std::string>&   args,
              bool                              verbose,
              test_utils::InvocationResultSet&  resultSet)
    {
        test_utils::InvocationResult invocationResult;
        invocationResult.mVariation = "<src:";
        invocationResult.mVariation += pixels::traits<BufferPixelType>::type_name;
        invocationResult.mVariation += " dst:";
        invocationResult.mVariation += pixels::traits<ImagePixelType>::type_name;
        invocationResult.mVariation += ">";

        auto& device = kernel.getDevice();

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::storage_buffer    srcBuffer(device.mDevice,
                                                  device.mMemoryProperties,
                                                  buffer_size);
        vulkan_utils::image             dstImage(device.mDevice,
                                                 device.mMemoryProperties,
                                                 buffer_width,
                                                 buffer_height,
                                                 vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                                 vulkan_utils::image::kUsage_ReadWrite);
        vulkan_utils::staging_buffer    dstImageStaging = dstImage.createStagingBuffer();

        // initialize source memory with random data
        auto srcBufferMap = srcBuffer.map<BufferPixelType>();
        test_utils::fill_random_pixels<BufferPixelType>(srcBufferMap.get(), srcBufferMap.get() + buffer_length);

        // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
        auto dstImageMap = dstImageStaging.map<ImagePixelType>();
        test_utils::copy_pixel_buffer<BufferPixelType, ImagePixelType>(srcBufferMap.get(), srcBufferMap.get() + buffer_length, dstImageMap.get());
        test_utils::invert_pixel_buffer<ImagePixelType>(dstImageMap.get(), dstImageMap.get() + buffer_length);

        dstImageMap.reset();
        srcBufferMap.reset();

        invocationResult.mExecutionTime = invoke(kernel,
                                                 srcBuffer,
                                                 dstImage,
                                                 0,
                                                 buffer_width,
                                                 pixels::traits<BufferPixelType>::cl_pixel_order,
                                                 pixels::traits<BufferPixelType>::cl_pixel_type,
                                                 false,
                                                 false,
                                                 buffer_width,
                                                 buffer_height);

        // readback the image data
        vk::UniqueCommandBuffer readbackCommand = vulkan_utils::allocate_command_buffer(device.mDevice, device.mCommandPool);
        readbackCommand->begin(vk::CommandBufferBeginInfo());
        dstImageStaging.copyFromImage(*readbackCommand);
        readbackCommand->end();

        vk::CommandBuffer rawCommand = *readbackCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        device.mComputeQueue.submit(submitInfo, nullptr);
        device.mComputeQueue.waitIdle();

        srcBufferMap = srcBuffer.map<BufferPixelType>();
        dstImageMap = dstImageStaging.map<ImagePixelType>();
        test_utils::check_results(srcBufferMap.get(),
                                  dstImageMap.get(),
                                  buffer_width, buffer_height,
                                  buffer_height,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }

    void test_matrix(clspv_utils::kernel&               kernel,
                     const std::vector<std::string>&    args,
                     bool                               verbose,
                     test_utils::InvocationResultSet&   resultSet);

    template <typename ImagePixelType>
    void test_series(clspv_utils::kernel&               kernel,
                     const std::vector<std::string>&    args,
                     bool                               verbose,
                     test_utils::InvocationResultSet&   resultSet)
    {
        const test_utils::test_kernel_fn tests[] = {
                test<gpu_types::uchar, ImagePixelType>,
                test<gpu_types::uchar4, ImagePixelType>,
                test<gpu_types::half, ImagePixelType>,
                test<gpu_types::half4, ImagePixelType>,
                test<float, ImagePixelType>,
                test<gpu_types::float2, ImagePixelType>,
                test<gpu_types::float4, ImagePixelType>,
        };

        test_utils::test_kernel_invocations(kernel,
                                            std::begin(tests), std::end(tests),
                                            args,
                                            verbose,
                                            resultSet);
    }

}

#endif //CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP

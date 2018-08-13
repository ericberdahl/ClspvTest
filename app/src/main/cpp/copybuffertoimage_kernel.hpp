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

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    template <typename BufferPixelType, typename ImagePixelType>
    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        if (vulkan_utils::image::supportsFormatUse(device.mPhysicalDevice,
                                                   vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                                   vulkan_utils::image::kUsage_ReadWrite))
        {
            const vk::Extent3D bufferExtent(64, 64, 1);
            const std::size_t buffer_length =
                    bufferExtent.width * bufferExtent.height * bufferExtent.depth;
            const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

            // allocate buffers and images
            vulkan_utils::storage_buffer srcBuffer(device.mDevice,
                                                   device.mMemoryProperties,
                                                   buffer_size);
            vulkan_utils::image dstImage(device.mDevice,
                                         device.mMemoryProperties,
                                         bufferExtent,
                                         vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                         vulkan_utils::image::kUsage_ReadWrite);
            vulkan_utils::staging_buffer dstImageStaging = dstImage.createStagingBuffer();

            // initialize source memory with random data
            auto srcBufferMap = srcBuffer.map<BufferPixelType>();
            test_utils::fill_random_pixels<BufferPixelType>(srcBufferMap.get(),
                                                            srcBufferMap.get() + buffer_length);

            // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
            auto dstImageMap = dstImageStaging.map<ImagePixelType>();
            test_utils::copy_pixel_buffer<BufferPixelType, ImagePixelType>(srcBufferMap.get(),
                                                                           srcBufferMap.get() +
                                                                           buffer_length,
                                                                           dstImageMap.get());
            test_utils::invert_pixel_buffer<ImagePixelType>(dstImageMap.get(),
                                                            dstImageMap.get() + buffer_length);

            dstImageMap.reset();
            srcBufferMap.reset();

            invocationResult.mExecutionTime = invoke(kernel,
                                                     srcBuffer,
                                                     dstImage,
                                                     0,
                                                     bufferExtent.width,
                                                     pixels::traits<BufferPixelType>::cl_pixel_order,
                                                     pixels::traits<BufferPixelType>::cl_pixel_type,
                                                     false,
                                                     false,
                                                     bufferExtent.width,
                                                     bufferExtent.height);

            // readback the image data
            vk::UniqueCommandBuffer readbackCommand = vulkan_utils::allocate_command_buffer(
                    device.mDevice, device.mCommandPool);
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
                                      bufferExtent,
                                      bufferExtent.width,
                                      verbose,
                                      invocationResult);
        }
        else
        {
            invocationResult.mSkipped = true;
            invocationResult.mMessages.push_back("Format not supported for storage");
        }

        return invocationResult;
    }

    template <typename BufferPixelType, typename ImagePixelType>
    test_utils::InvocationTest getTestVariant()
    {
        test_utils::InvocationTest result;

        std::ostringstream os;
        os << "<src:" << pixels::traits<BufferPixelType>::type_name << " dst:" << pixels::traits<ImagePixelType>::type_name << ">";
        result.mVariation = os.str();

        result.mTestFn = test<BufferPixelType, ImagePixelType>;

        return result;
    }
}

#endif //CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP

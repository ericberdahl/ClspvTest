//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP
#define CLSPVTEST_COPYBUFFERTOIMAGE_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

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
    struct Test
    {
        Test(const clspv_utils::device& device, const std::vector<std::string>& args) :
                mBufferExtent(64, 64, 1)
        {
            const std::size_t buffer_length =
                    mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
            const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

            // allocate buffers and images
            mSrcBuffer = vulkan_utils::storage_buffer(device.getDevice(),
                                                   device.getMemoryProperties(),
                                                   buffer_size);
            mDstImage = vulkan_utils::image(device.getDevice(),
                                         device.getMemoryProperties(),
                                         mBufferExtent,
                                         vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                         vulkan_utils::image::kUsage_ReadWrite);
            mDstImageStaging = mDstImage.createStagingBuffer();

            // initialize source memory with random data
            auto srcBufferMap = mSrcBuffer.map<BufferPixelType>();
            test_utils::fill_random_pixels<BufferPixelType>(srcBufferMap.get(),
                                                            srcBufferMap.get() + buffer_length);
        }

        void prepare()
        {
            const std::size_t buffer_length =
                    mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

            // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
            auto srcBufferMap = mSrcBuffer.map<BufferPixelType>();
            auto dstImageMap = mDstImageStaging.map<ImagePixelType>();
            test_utils::copy_pixel_buffer<BufferPixelType, ImagePixelType>(srcBufferMap.get(),
                                                                           srcBufferMap.get() +
                                                                           buffer_length,
                                                                           dstImageMap.get());
            test_utils::invert_pixel_buffer<ImagePixelType>(dstImageMap.get(),
                                                            dstImageMap.get() + buffer_length);
        }

        void run(clspv_utils::kernel& kernel, test_utils::InvocationResult& invocationResult)
        {
            invocationResult.mExecutionTime = invoke(kernel,
                                                     mSrcBuffer,
                                                     mDstImage,
                                                     0,
                                                     mBufferExtent.width,
                                                     pixels::traits<BufferPixelType>::cl_pixel_order,
                                                     pixels::traits<BufferPixelType>::cl_pixel_type,
                                                     false,
                                                     false,
                                                     mBufferExtent.width,
                                                     mBufferExtent.height);
        }

        void checkResults(const clspv_utils::device& device, test_utils::InvocationResult& invocationResult, bool verbose)
        {
            // readback the image data
            vk::UniqueCommandBuffer readbackCommand = vulkan_utils::allocate_command_buffer(
                    device.getDevice(), device.getCommandPool());
            readbackCommand->begin(vk::CommandBufferBeginInfo());
            mDstImageStaging.copyFromImage(*readbackCommand);
            readbackCommand->end();

            vk::CommandBuffer rawCommand = *readbackCommand;
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBufferCount(1)
                    .setPCommandBuffers(&rawCommand);

            device.getComputeQueue().submit(submitInfo, nullptr);
            device.getComputeQueue().waitIdle();

            auto srcBufferMap = mSrcBuffer.map<BufferPixelType>();
            auto dstImageMap = mDstImageStaging.map<ImagePixelType>();
            test_utils::check_results(srcBufferMap.get(),
                                      dstImageMap.get(),
                                      mBufferExtent,
                                      mBufferExtent.width,
                                      verbose,
                                      invocationResult);

        }

        static bool isSupported(clspv_utils::device& device)
        {
            vulkan_utils::image::supportsFormatUse(device.getPhysicalDevice(),
                                                   vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                                   vulkan_utils::image::kUsage_ReadWrite);
        }

        vk::Extent3D                    mBufferExtent;
        vulkan_utils::storage_buffer    mSrcBuffer;
        vulkan_utils::image             mDstImage;
        vulkan_utils::staging_buffer    mDstImageStaging;

    };

    template <typename BufferPixelType, typename ImagePixelType>
    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        if (vulkan_utils::image::supportsFormatUse(device.getPhysicalDevice(),
                                                   vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                                   vulkan_utils::image::kUsage_ReadWrite))
        {
            Test<BufferPixelType, ImagePixelType> t(kernel.getDevice(), args);

            t.prepare();
            t.run(kernel, invocationResult);
            t.checkResults(kernel.getDevice(), invocationResult, verbose);
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

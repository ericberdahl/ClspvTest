//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP
#define CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace copyimagetobuffer_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              dst_offset,
           int                              dst_pitch,
           cl_channel_order                 dst_channel_order,
           cl_channel_type                  dst_channel_type,
           bool                             swap_components,
           int                              width,
           int                              height);

    test_utils::KernelTest::invocation_tests getAllTestVariants();

    template <typename BufferPixelType, typename ImagePixelType>
    struct Test : public test_utils::Test
    {
        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
            mBufferExtent(64, 64, 1)
        {
            auto& device = kernel.getDevice();

            const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
            const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

            // allocate buffers and images
            mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(),
                                                       device.getMemoryProperties(),
                                                       buffer_size);
            mSrcImage = vulkan_utils::image(device.getDevice(),
                                                     device.getMemoryProperties(),
                                                     mBufferExtent,
                                                     vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                                     vulkan_utils::image::kUsage_ReadOnly);
            mSrcImageStaging = mSrcImage.createStagingBuffer();

            // initialize source memory with random data
            auto srcImageMap = mSrcImageStaging.map<ImagePixelType>();
            test_utils::fill_random_pixels<ImagePixelType>(srcImageMap.get(), srcImageMap.get() + buffer_length);
            srcImageMap.reset();

            // complete setup of the image
            mSetupCommand = vulkan_utils::allocate_command_buffer(device.getDevice(),
                                                                                         device.getCommandPool());
            mSetupCommand->begin(vk::CommandBufferBeginInfo());
            mSrcImageStaging.copyToImage(*mSetupCommand);
            mSetupCommand->end();

            vk::CommandBuffer rawCommand = *mSetupCommand;
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBufferCount(1)
                    .setPCommandBuffers(&rawCommand);

            device.getComputeQueue().submit(submitInfo, nullptr);
        }

        virtual void prepare() override
        {
            const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

            // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
            auto srcImageMap = mSrcImageStaging.map<ImagePixelType>();
            auto dstBufferMap = mDstBuffer.map<BufferPixelType>();
            test_utils::copy_pixel_buffer<ImagePixelType, BufferPixelType>(srcImageMap.get(), srcImageMap.get() + buffer_length, dstBufferMap.get());
            test_utils::invert_pixel_buffer<BufferPixelType>(dstBufferMap.get(), dstBufferMap.get() + buffer_length);
        }

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override
        {
            return invoke(kernel,
                          mSrcImage,
                          mDstBuffer,
                          0,
                          mBufferExtent.width,
                          pixels::traits<BufferPixelType>::cl_pixel_order,
                          pixels::traits<BufferPixelType>::cl_pixel_type,
                          false,
                          mBufferExtent.width,
                          mBufferExtent.height);
        }

        virtual test_utils::Evaluation evaluate(bool verbose) override
        {
            auto srcImageMap = mSrcImageStaging.map<ImagePixelType>();
            auto dstBufferMap = mDstBuffer.map<BufferPixelType>();
            return test_utils::check_results(srcImageMap.get(),
                                             dstBufferMap.get(),
                                             mBufferExtent,
                                             mBufferExtent.width,
                                             verbose);
        }

        vk::Extent3D                    mBufferExtent;
        vulkan_utils::storage_buffer    mDstBuffer;
        vulkan_utils::image             mSrcImage;
        vulkan_utils::staging_buffer    mSrcImageStaging;
        vk::UniqueCommandBuffer         mSetupCommand;
    };

    template <typename BufferPixelType, typename ImagePixelType>
    test_utils::InvocationTest getTestVariant()
    {
        test_utils::InvocationTest result;

        std::ostringstream os;
        os << "<src:" << pixels::traits<BufferPixelType>::type_name << " dst:" << pixels::traits<ImagePixelType>::type_name << ">";
        result.mVariation = os.str();

        result.mTestFn = test_utils::run_test<Test<BufferPixelType, ImagePixelType>>;

        return result;
    }
}

#endif //CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP

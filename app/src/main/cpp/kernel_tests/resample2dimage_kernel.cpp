//
// Created by Eric Berdahl on 10/31/17.
//

#include "resample2dimage_kernel.hpp"

#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"

#include <vulkan/vulkan.hpp>

#include <util.hpp>

namespace {
    float normalize_coord(int coord, int range)
    {
        return ((float)coord) / ((float)(range - 1));
    }

    void fill_2d_space(gpu_types::float4* first, gpu_types::float4* last, int width, int height)
    {
        int row = 0;
        int col = 0;

        while (first != last)
        {
            gpu_types::float4 pixel(col,
                                    row,
                                    0.0f,
                                    0.0f);

            *first = pixel;
            ++first;

            ++col;
            if (width == col)
            {
                col= 0;
                ++row;
            }
        }
    }

    float clampf(float value, float lo, float hi)
    {
        if (value < lo) return lo;
        if (value > hi) return hi;
        return value;
    }
}

namespace resample2dimage_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&     kernel,
           vulkan_utils::image&     src_image,
           vulkan_utils::buffer&    dst_buffer,
           vk::Extent3D             extent)
    {
        if (1 != extent.depth)
        {
            throw std::runtime_error("Depth of extent must be 1");
        }

        struct scalar_args {
            int inWidth;            // offset 0
            int inHeight;           // offset 4
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(4 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        vulkan_utils::buffer scalarBuffer = vulkan_utils::createUniformBuffer(kernel.getDevice().getDevice(),
                                                                              kernel.getDevice().getMemoryProperties(),
                                                                              sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = extent.width;
        scalars->inHeight = extent.height;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (extent.width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (extent.height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
        mBufferExtent(64, 64, 1)
    {
        auto& device = kernel.getDevice();

        const int image_height = 3;
        const int image_width = 3;
        const int image_buffer_length = image_width * image_height;
        const gpu_types::float4 image_buffer_data[] = {
                { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 0.5f, 0.0f, 0.0f }, { 0.5f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.5f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f, 0.0f }
        };

        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        mDstBuffer = vulkan_utils::createStorageBuffer(device.getDevice(),
                                                       device.getMemoryProperties(),
                                                       buffer_size);
        mSrcImage = vulkan_utils::image(device.getDevice(),
                                     device.getMemoryProperties(),
                                     vk::Extent3D(image_width, image_height, 1),
                                     vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                     vulkan_utils::image::kUsage_ReadOnly);
        mSrcImageStaging = vulkan_utils::createStagingBuffer(device.getDevice(),
                                                             device.getMemoryProperties(),
                                                             mSrcImage,
                                                             true,
                                                             false);

        // initialize source memory with random data
        auto srcImageMap = mSrcImageStaging.map<ImagePixelType>();
        std::copy(std::begin(image_buffer_data), std::end(image_buffer_data), srcImageMap.get());
        srcImageMap.reset();

        mExpectedDstBuffer.resize(buffer_length);
        for (int row = 0; row < mBufferExtent.height; ++row)
        {
            for (int col = 0; col < mBufferExtent.width; ++col)
            {
                gpu_types::float2 normalizedCoordinate(((float)col + 0.5f) / ((float)mBufferExtent.width),
                                                       ((float)row + 0.5f) / ((float)mBufferExtent.height));

                gpu_types::float2 sampledCoordinate(clampf(normalizedCoordinate.x*image_width - 0.5f, 0.0f, image_width - 1)/(image_width - 1),
                                                    clampf(normalizedCoordinate.y*image_height - 0.5f, 0.0f, image_height - 1)/(image_height - 1));

                auto index = (row * mBufferExtent.width) + col;
                mExpectedDstBuffer[index] = BufferPixelType(sampledCoordinate.x,
                                                             sampledCoordinate.y,
                                                             0.0f,
                                                             0.0f);
            }
        }

        // complete setup of the image
        mSetupCommand = vulkan_utils::allocate_command_buffer(device.getDevice(), device.getCommandPool());
        mSetupCommand->begin(vk::CommandBufferBeginInfo());
        vulkan_utils::copyBufferToImage(*mSetupCommand, mSrcImageStaging, mSrcImage);
        mSetupCommand->end();

        vk::CommandBuffer rawCommand = *mSetupCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        device.getComputeQueue().submit(submitInfo, nullptr);
    }

    void Test::prepare()
    {
        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

        // initialize destination memory to zero
        auto dstBufferMap = mDstBuffer.map<BufferPixelType>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, BufferPixelType(0.0f, 0.0f, 0.0f, 0.0f));
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mSrcImage,
                      mDstBuffer,
                      mBufferExtent);
    }

    test_utils::Evaluation Test::evaluate(bool verbose)
    {
        auto dstBufferMap = mDstBuffer.map<BufferPixelType>();
        return test_utils::check_results(mExpectedDstBuffer.data(),
                                         dstBufferMap.get(),
                                         mBufferExtent,
                                         mBufferExtent.width,
                                         verbose);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        return test_utils::KernelTest::invocation_tests({ test_utils::make_invocation_test<Test>("") });
    }

}

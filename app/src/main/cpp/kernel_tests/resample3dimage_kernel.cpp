//
// Created by Eric Berdahl on 10/31/17.
//

#include "resample3dimage_kernel.hpp"

#include "clspv_utils/kernel.hpp"
#include "gpu_types.hpp"

#include <vulkan/vulkan.hpp>

namespace {
    float clampf(float value, float lo, float hi)
    {
        if (value < lo) return lo;
        if (value > hi) return hi;
        return value;
    }
}

namespace resample3dimage_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              width,
           int                              height,
           int                              depth)
    {
        struct scalar_args {
            int inWidth;            // offset 0
            int inHeight;           // offset 4
            int inDepth;            // offset 8
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(4 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");
        static_assert(8 == offsetof(scalar_args, inDepth), "inDepth offset incorrect");

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().getDevice(),
                                                  kernel.getDevice().getMemoryProperties(),
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars->inDepth = depth;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                (depth + workgroup_sizes.depth - 1) / workgroup_sizes.depth);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
        mBufferExtent(64, 64, 64)
    {
        auto& device = kernel.getDevice();

        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        const vk::Extent3D imageExtent(3, 3, 3);
        const int image_buffer_length = imageExtent.width * imageExtent.height * imageExtent.depth;
        const gpu_types::float4 image_buffer_data[] = {
                { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 0.5f, 0.0f, 0.0f }, { 0.5f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.5f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f, 0.0f },

                { 0.0f, 0.0f, 0.5f, 0.0f }, { 0.5f, 0.0f, 0.5f, 0.0f }, { 1.0f, 0.0f, 0.5f, 0.0f },
                { 0.0f, 0.5f, 0.5f, 0.0f }, { 0.5f, 0.5f, 0.5f, 0.0f }, { 1.0f, 0.5f, 0.5f, 0.0f },
                { 0.0f, 1.0f, 0.5f, 0.0f }, { 0.5f, 1.0f, 0.5f, 0.0f }, { 1.0f, 1.0f, 0.5f, 0.0f },

                { 0.0f, 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 1.0f, 0.0f },
                { 0.0f, 0.5f, 1.0f, 0.0f }, { 0.5f, 0.5f, 1.0f, 0.0f }, { 1.0f, 0.5f, 1.0f, 0.0f },
                { 0.0f, 1.0f, 1.0f, 0.0f }, { 0.5f, 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f },
        };

        // allocate buffers and images
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(),
                                                device.getMemoryProperties(),
                                                buffer_size);
        mSrcImage = vulkan_utils::image(device.getDevice(),
                                     device.getMemoryProperties(),
                                     imageExtent,
                                     vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                     vulkan_utils::image::kUsage_ReadOnly);
        mSrcImageStaging = mSrcImage.createStagingBuffer();

        // initialize source memory with random data
        auto srcImageMap = mSrcImageStaging.map<ImagePixelType>();
        std::copy(std::begin(image_buffer_data), std::end(image_buffer_data), srcImageMap.get());
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

        // compute expected results
        mExpectedDstBuffer.resize(buffer_length);
        for (int row = 0; row < mBufferExtent.height; ++row)
        {
            for (int col = 0; col < mBufferExtent.width; ++col)
            {
                for (int slice = 0; slice < mBufferExtent.depth; ++slice) {
                    gpu_types::float4 normalizedCoordinate(
                            ((float) col + 0.5f) / ((float) mBufferExtent.width),
                            ((float) row + 0.5f) / ((float) mBufferExtent.height),
                            ((float) slice + 0.5f) / ((float) mBufferExtent.depth),
                            0.0f);

                    gpu_types::float4 sampledCoordinate(
                            clampf(normalizedCoordinate.x * imageExtent.width - 0.5f, 0.0f, imageExtent.width - 1) / (imageExtent.width - 1),
                            clampf(normalizedCoordinate.y * imageExtent.height - 0.5f, 0.0f, imageExtent.height - 1) / (imageExtent.height - 1),
                            clampf(normalizedCoordinate.z * imageExtent.depth - 0.5f, 0.0f, imageExtent.depth - 1) / (imageExtent.depth - 1),
                            0.0f);

                    mExpectedDstBuffer[(((slice * mBufferExtent.height) + row) * mBufferExtent.width) + col] = sampledCoordinate;
                }
            }
        }
    }

    void Test::prepare()
    {
        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

        // initialize destination memory to zero
        auto dstBufferMap = mDstBuffer.map<BufferPixelType>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, gpu_types::float4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mSrcImage,
                      mDstBuffer,
                      mBufferExtent.width,
                      mBufferExtent.height,
                      mBufferExtent.depth);
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

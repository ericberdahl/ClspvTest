//
// Created by Eric Berdahl on 10/31/17.
//

#include "resample3dimage_kernel.hpp"

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

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().mDevice,
                                                  kernel.getDevice().mMemoryProperties,
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

        clspv_utils::kernel_invocation invocation = kernel.createInvocation();

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    void test(clspv_utils::kernel &kernel,
              const std::vector<std::string> &args,
              bool verbose,
              test_utils::InvocationResultSet &resultSet)
    {
        typedef gpu_types::float4 BufferPixelType;
        typedef gpu_types::float4 ImagePixelType;

        test_utils::InvocationResult invocationResult;
        invocationResult.mVariation = "<src:";
        invocationResult.mVariation += pixels::traits<ImagePixelType>::type_name;
        invocationResult.mVariation += " dst:";
        invocationResult.mVariation += pixels::traits<BufferPixelType>::type_name;
        invocationResult.mVariation += ">";

        auto &device = kernel.getDevice();

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

        const vk::Extent3D bufferExtent(64, 64, 64);
        const std::size_t buffer_length = bufferExtent.width * bufferExtent.height * bufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::storage_buffer dst_buffer(device.mDevice,
                                                device.mMemoryProperties,
                                                buffer_size);
        vulkan_utils::image srcImage(device.mDevice,
                                     device.mMemoryProperties,
                                     imageExtent,
                                     vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type),
                                     vulkan_utils::image::kUsage_ReadOnly);
        vulkan_utils::staging_buffer srcImageStaging = srcImage.createStagingBuffer();

        // initialize source memory with random data
        auto srcImageMap = srcImageStaging.map<ImagePixelType>();
        std::copy(std::begin(image_buffer_data), std::end(image_buffer_data), srcImageMap.get());

        // initialize destination memory to zero
        auto dstBufferMap = dst_buffer.map<BufferPixelType>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, gpu_types::float4(0.0f, 0.0f, 0.0f, 0.0f));

        std::vector<gpu_types::float4> expectedDstBuffer(buffer_length);
        for (int row = 0; row < bufferExtent.height; ++row)
        {
            for (int col = 0; col < bufferExtent.width; ++col)
            {
                for (int slice = 0; slice < bufferExtent.depth; ++slice) {
                    gpu_types::float4 normalizedCoordinate(
                            ((float) col + 0.5f) / ((float) bufferExtent.width),
                            ((float) row + 0.5f) / ((float) bufferExtent.height),
                            ((float) slice + 0.5f) / ((float) bufferExtent.depth),
                            0.0f);

                    gpu_types::float4 sampledCoordinate(
                            clampf(normalizedCoordinate.x * imageExtent.width - 0.5f, 0.0f, imageExtent.width - 1) / (imageExtent.width - 1),
                            clampf(normalizedCoordinate.y * imageExtent.height - 0.5f, 0.0f, imageExtent.height - 1) / (imageExtent.height - 1),
                            clampf(normalizedCoordinate.z * imageExtent.depth - 0.5f, 0.0f, imageExtent.depth - 1) / (imageExtent.depth - 1),
                            0.0f);

                    expectedDstBuffer[(((slice * bufferExtent.height) + row) * bufferExtent.width) + col] = sampledCoordinate;
                }
            }
        }


        dstBufferMap.reset();
        srcImageMap.reset();

        // complete setup of the image
        vk::UniqueCommandBuffer setupCommand = vulkan_utils::allocate_command_buffer(device.mDevice,
                                                                                     device.mCommandPool);
        setupCommand->begin(vk::CommandBufferBeginInfo());
        srcImageStaging.copyToImage(*setupCommand);
        setupCommand->end();

        vk::CommandBuffer rawCommand = *setupCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        device.mComputeQueue.submit(submitInfo, nullptr);

        invocationResult.mExecutionTime = invoke(kernel,
                                                 srcImage,
                                                 dst_buffer,
                                                 bufferExtent.width,
                                                 bufferExtent.height,
                                                 bufferExtent.depth);

        srcImageMap = srcImageStaging.map<ImagePixelType>();
        dstBufferMap = dst_buffer.map<BufferPixelType>();
        test_utils::check_results(expectedDstBuffer.data(),
                                  dstBufferMap.get(),
                                  bufferExtent,
                                  bufferExtent.width,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

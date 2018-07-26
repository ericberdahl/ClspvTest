//
// Created by Eric Berdahl on 10/31/17.
//

#include "resample2dimage_kernel.hpp"

#include "gpu_types.hpp"

#include <vulkan/vulkan.hpp>

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
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              width,
           int                              height)
    {
        struct scalar_args {
            int inWidth;            // offset 0
            int inHeight;           // offset 4
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(4 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().mDevice,
                                                  kernel.getDevice().mMemoryProperties,
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = width;
        scalars->inHeight = height;
        scalars.reset();

        const vk::Extent2D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent2D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height);

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

        const int image_height = 3;
        const int image_width = 3;
        const int image_buffer_length = image_width * image_height;
        const gpu_types::float4 image_buffer_data[] = {
                { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 0.5f, 0.0f, 0.0f }, { 0.5f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.5f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f, 0.0f }
        };

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::storage_buffer dst_buffer(device.mDevice,
                                                device.mMemoryProperties,
                                                buffer_size);
        vulkan_utils::image srcImage(device.mDevice,
                                     device.mMemoryProperties,
                                     image_width,
                                     image_height,
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
        for (int row = 0; row < buffer_height; ++row)
        {
            for (int col = 0; col < buffer_width; ++col)
            {
                gpu_types::float2 normalizedCoordinate(((float)col + 0.5f) / ((float)buffer_width),
                                                       ((float)row + 0.5f) / ((float)buffer_height));

                gpu_types::float2 sampledCoordinate(clampf(normalizedCoordinate.x*image_width - 0.5f, 0.0f, image_width - 1)/(image_width - 1),
                                                    clampf(normalizedCoordinate.y*image_height - 0.5f, 0.0f, image_height - 1)/(image_height - 1));

                expectedDstBuffer[(row * buffer_width) + col] = gpu_types::float4(sampledCoordinate.x,
                                                                                  sampledCoordinate.y,
                                                                                  normalizedCoordinate.x,
                                                                                  normalizedCoordinate.y);
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
                                                 buffer_width,
                                                 buffer_height);

        srcImageMap = srcImageStaging.map<ImagePixelType>();
        dstBufferMap = dst_buffer.map<BufferPixelType>();
        test_utils::check_results(expectedDstBuffer.data(),
                                  dstBufferMap.get(),
                                  buffer_width, buffer_height,
                                  buffer_height,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

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
           vk::Extent3D                     extent)
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

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().mDevice,
                                                  kernel.getDevice().mMemoryProperties,
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

        clspv_utils::kernel_invocation invocation = kernel.createInvocation();

        invocation.addReadOnlyImageArgument(src_image);
        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::InvocationResult test(clspv_utils::kernel &kernel,
                                      const std::vector<std::string> &args,
                                      bool verbose)
    {
        typedef gpu_types::float4 BufferPixelType;
        typedef gpu_types::float4 ImagePixelType;

        test_utils::InvocationResult invocationResult;

        auto &device = kernel.getDevice();

        const int image_height = 3;
        const int image_width = 3;
        const int image_buffer_length = image_width * image_height;
        const gpu_types::float4 image_buffer_data[] = {
                { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 0.5f, 0.0f, 0.0f }, { 0.5f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.5f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f }, { 0.5f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f, 0.0f }
        };

        const vk::Extent3D bufferExtent(64, 64, 1);
        const std::size_t buffer_length = bufferExtent.width * bufferExtent.height * bufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::storage_buffer dst_buffer(device.mDevice,
                                                device.mMemoryProperties,
                                                buffer_size);
        vulkan_utils::image srcImage(device.mDevice,
                                     device.mMemoryProperties,
                                     vk::Extent3D(image_width, image_height, 1),
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
                gpu_types::float2 normalizedCoordinate(((float)col + 0.5f) / ((float)bufferExtent.width),
                                                       ((float)row + 0.5f) / ((float)bufferExtent.height));

                gpu_types::float2 sampledCoordinate(clampf(normalizedCoordinate.x*image_width - 0.5f, 0.0f, image_width - 1)/(image_width - 1),
                                                    clampf(normalizedCoordinate.y*image_height - 0.5f, 0.0f, image_height - 1)/(image_height - 1));

                auto index = (row * bufferExtent.width) + col;
                expectedDstBuffer[index] = gpu_types::float4(sampledCoordinate.x,
                                                             sampledCoordinate.y,
                                                             0.0f,
                                                             0.0f);
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
                                                 bufferExtent);

        srcImageMap = srcImageStaging.map<ImagePixelType>();
        dstBufferMap = dst_buffer.map<BufferPixelType>();
        test_utils::check_results(expectedDstBuffer.data(),
                                  dstBufferMap.get(),
                                  bufferExtent,
                                  bufferExtent.width,
                                  verbose,
                                  invocationResult);

        return invocationResult;
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        test_utils::InvocationTest t({ "", test });
        return test_utils::KernelTest::invocation_tests({ t });
    }

}

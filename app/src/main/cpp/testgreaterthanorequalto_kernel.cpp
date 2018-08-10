//
// Created by Eric Berdahl on 10/31/17.
//

#include "testgreaterthanorequalto_kernel.hpp"

namespace testgreaterthanorequalto_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           vk::Extent3D                     extent)
    {
        if (1 != extent.depth)
        {
            throw std::runtime_error("Depth must be 1");
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

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose)
    {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        const vk::Extent3D bufferExtent(64, 64, 1);
        const std::size_t buffer_length = bufferExtent.width * bufferExtent.height * bufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // allocate buffers and images
        vulkan_utils::storage_buffer  dstBuffer(device.mDevice, device.mMemoryProperties, buffer_size);

        // initialize destination memory with unexpected value. the kernel should write either 0 or
        // 1. so, initialize thedestination with 2.
        auto dstBufferMap = dstBuffer.map<float>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, 2.0f);
        dstBufferMap.reset();

        // set up expected results of the destination buffer
        int index = 0;
        std::vector<float> expectedResults(buffer_length);
        std::generate(expectedResults.begin(), expectedResults.end(), [&index, bufferExtent]() {
            int x = index % bufferExtent.width;
            int y = index / bufferExtent.width;

            ++index;

            return (x >= 0 && y >= 0 && x < bufferExtent.width && y < bufferExtent.height ? 1.0f : 0.0f);
        });

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dstBuffer,
                                                 bufferExtent);

        dstBufferMap = dstBuffer.map<float>();
        test_utils::check_results(expectedResults.data(), dstBufferMap.get(),
                                  bufferExtent,
                                  bufferExtent.width,
                                  verbose,
                                  invocationResult);

        return invocationResult;
    }

    test_utils::test_kernel_series getAllTestVariants()
    {
        return test_utils::test_kernel_series({ test_utils::test_kernel_fn(test) });
    }

}

//
// Created by Eric Berdahl on 10/31/17.
//

#include "testgreaterthanorequalto_kernel.hpp"

namespace testgreaterthanorequalto_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
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

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::kernel_invocation invocation = kernel.createInvocation();

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    void test_all(clspv_utils::kernel&              kernel,
                  const std::vector<std::string>&   args,
                  bool                              verbose,
                  test_utils::InvocationResultSet&  resultSet)
    {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
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
        std::generate(expectedResults.begin(), expectedResults.end(), [&index, buffer_width, buffer_height]() {
            int x = index % buffer_width;
            int y = index / buffer_width;

            ++index;

            return (x >= 0 && y >= 0 && x < buffer_width && y < buffer_height ? 1.0f : 0.0f);
        });

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dstBuffer,
                                                 buffer_width,
                                                 buffer_height);

        dstBufferMap = dstBuffer.map<float>();
        test_utils::check_results(expectedResults.data(), dstBufferMap.get(),
                                  buffer_width, buffer_height, 1,
                                  buffer_width,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

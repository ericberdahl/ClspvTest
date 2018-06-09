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

        const scalar_args scalars = {
                width,
                height
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(kernel);

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addPodArgument(scalars);

        return invocation.run(num_workgroups);
    }

    void test_all(clspv_utils::kernel&              kernel,
                  const std::vector<std::string>&   args,
                  bool                              verbose,
                  test_utils::InvocationResultSet&  resultSet)
    {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // allocate buffers and images
        vulkan_utils::storage_buffer  dstBuffer(device.mDevice, device.mMemoryProperties, buffer_size);

        // initialize destination memory with unexpected value. the kernel should write either 0 or
        // 1. so, initialize thedestination with 2.
        auto dstBufferMap = dstBuffer.mem.map<float>();
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

        dstBufferMap = dstBuffer.mem.map<float>();
        test_utils::check_results(expectedResults.data(), dstBufferMap.get(),
                                  buffer_width, buffer_height,
                                  buffer_height,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

//
// Created by Eric Berdahl on 10/31/17.
//

#include "testgreaterthanorequalto_kernel.hpp"

namespace testgreaterthanorequalto_kernel {

    clspv_utils::kernel_invocation::execution_time_t
    invoke(const clspv_utils::kernel_module&   module,
           const clspv_utils::kernel&          kernel,
           const sample_info&                  info,
           vk::ArrayProxy<const vk::Sampler>   samplers,
           vk::Buffer                          dst_buffer,
           int                                 width,
           int                                 height)
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

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                                  info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addBufferArgument(dst_buffer);
        invocation.addPodArgument(scalars);

        return invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    void test_all(const clspv_utils::kernel_module&    module,
                  const clspv_utils::kernel&           kernel,
                  const sample_info&                   info,
                  vk::ArrayProxy<const vk::Sampler>    samplers,
                  test_utils::InvocationResultSet&     resultSet)
    {
        test_utils::InvocationResult invocationResult;

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // allocate buffers and images
        vulkan_utils::storage_buffer  dstBuffer(info, buffer_size);

        // initialize destination memory with unexpected value. the kernel should write either 0 or
        // 1. so, initialize thedestination with 2.
        vulkan_utils::fillDeviceMemory(dstBuffer.mem, buffer_length, 2.0f);

        // set up expected results of the destination buffer
        int index = 0;
        std::vector<float> expectedResults(buffer_length);
        std::generate(expectedResults.begin(), expectedResults.end(), [&index, buffer_width, buffer_height]() {
            int x = index % buffer_width;
            int y = index / buffer_width;

            ++index;

            return (x >= 0 && y >= 0 && x < buffer_width && y < buffer_height ? 1.0f : 0.0f);
        });

        invocationResult.mExecutionTime = invoke(module, kernel,
                                                 info,
                                                 samplers,
                                                 *dstBuffer.buf,
                                                 buffer_width,
                                                 buffer_height);

        test_utils::check_results<float, float>(expectedResults.data(), dstBuffer.mem,
                                                buffer_width, buffer_height,
                                                buffer_height,
                                                invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

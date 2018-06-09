//
// Created by Eric Berdahl on 10/31/17.
//

#include "strangeshuffle_kernel.hpp"

namespace strangeshuffle_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    index_buffer,
           vulkan_utils::storage_buffer&    source_buffer,
           vulkan_utils::storage_buffer&    destination_buffer,
           std::size_t                      num_elements)
    {
        if (0 != (num_elements % 2)) {
            throw std::runtime_error("num_elements must be even");
        }

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(((num_elements/2) + workgroup_sizes.x - 1)/workgroup_sizes.x);

        clspv_utils::kernel_invocation invocation(kernel);

        invocation.addStorageBufferArgument(index_buffer);
        invocation.addStorageBufferArgument(source_buffer);
        invocation.addStorageBufferArgument(destination_buffer);
        invocation.addLocalArraySizeArgument(2 * workgroup_sizes.x);
        return invocation.run(num_workgroups);
    }

    void test(clspv_utils::kernel&              kernel,
              const std::vector<std::string>&   args,
              bool                              verbose,
              test_utils::InvocationResultSet&  resultSet) {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        int buffer_width = 4096;

        // allocate source and destination buffers
        const std::size_t pixel_buffer_size = buffer_width * sizeof(gpu_types::float4);
        vulkan_utils::storage_buffer src_buffer(device.mDevice, device.mMemoryProperties, pixel_buffer_size);
        vulkan_utils::storage_buffer dst_buffer(device.mDevice, device.mMemoryProperties, pixel_buffer_size);

        // allocate index buffer
        const std::size_t index_buffer_size = buffer_width * sizeof(int32_t);
        vulkan_utils::storage_buffer index_buffer(device.mDevice, device.mMemoryProperties, index_buffer_size);

        test_utils::fill_random_pixels<gpu_types::float4>(src_buffer.mem, buffer_width);
        test_utils::fill_random_pixels<gpu_types::float4>(dst_buffer.mem, buffer_width);

        auto mappedIndices = index_buffer.mem.map<int32_t>();
        std::iota(mappedIndices.get(), mappedIndices.get() + buffer_width, 0);
        mappedIndices.reset();

        invocationResult.mExecutionTime = invoke(kernel,
                                                 index_buffer,
                                                 src_buffer,
                                                 dst_buffer,
                                                 buffer_width);

        test_utils::check_results<gpu_types::float4,gpu_types::float4>(src_buffer.mem,
                                                                       dst_buffer.mem,
                                                                       buffer_width, 1,
                                                                       buffer_width,
                                                                       verbose,
                                                                       invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

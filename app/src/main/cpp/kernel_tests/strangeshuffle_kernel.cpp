//
// Created by Eric Berdahl on 10/31/17.
//

#include "strangeshuffle_kernel.hpp"

#include "clspv_utils/kernel.hpp"

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

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(((num_elements/2) + workgroup_sizes.width - 1)/workgroup_sizes.width,
                                          1,
                                          1);

        clspv_utils::kernel_invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(index_buffer);
        invocation.addStorageBufferArgument(source_buffer);
        invocation.addStorageBufferArgument(destination_buffer);
        invocation.addLocalArraySizeArgument(2 * workgroup_sizes.width);
        return invocation.run(num_workgroups);
    }

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose) {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        int buffer_width = 4096;

        // allocate source and destination buffers
        const std::size_t pixel_buffer_size = buffer_width * sizeof(gpu_types::float4);
        vulkan_utils::storage_buffer src_buffer(device.getDevice(), device.getMemoryProperties(), pixel_buffer_size);
        vulkan_utils::storage_buffer dst_buffer(device.getDevice(), device.getMemoryProperties(), pixel_buffer_size);

        // allocate index buffer
        const std::size_t index_buffer_size = buffer_width * sizeof(int32_t);
        vulkan_utils::storage_buffer index_buffer(device.getDevice(), device.getMemoryProperties(), index_buffer_size);

        auto srcBufferMap = src_buffer.map<gpu_types::float4>();
        test_utils::fill_random_pixels<gpu_types::float4>(srcBufferMap.get(), srcBufferMap.get() + buffer_width);
        srcBufferMap.reset();

        auto dstBufferMap = dst_buffer.map<gpu_types::float4>();
        test_utils::fill_random_pixels<gpu_types::float4>(dstBufferMap.get(), dstBufferMap.get() + buffer_width);
        dstBufferMap.reset();

        auto mappedIndices = index_buffer.map<int32_t>();
        std::iota(mappedIndices.get(), mappedIndices.get() + buffer_width, 0);
        mappedIndices.reset();

        invocationResult.mExecutionTime = invoke(kernel,
                                                 index_buffer,
                                                 src_buffer,
                                                 dst_buffer,
                                                 buffer_width);

        srcBufferMap = src_buffer.map<gpu_types::float4>();
        dstBufferMap = dst_buffer.map<gpu_types::float4>();
        test_utils::check_results(srcBufferMap.get(),
                                  dstBufferMap.get(),
                                  vk::Extent3D(buffer_width, 1, 1),
                                  buffer_width,
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

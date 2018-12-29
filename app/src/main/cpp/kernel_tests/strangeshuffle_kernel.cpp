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

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(index_buffer);
        invocation.addStorageBufferArgument(source_buffer);
        invocation.addStorageBufferArgument(destination_buffer);
        invocation.addLocalArraySizeArgument(2 * workgroup_sizes.width);
        return invocation.run(num_workgroups);
    }

    Test::Test(const clspv_utils::device& device, const std::vector<std::string>& args) :
        mBufferWidth(4096)
    {
        // allocate source and destination buffers
        const std::size_t pixel_buffer_size = mBufferWidth * sizeof(gpu_types::float4);
        mSrcBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), pixel_buffer_size);
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), pixel_buffer_size);

        // allocate index buffer
        const std::size_t index_buffer_size = mBufferWidth * sizeof(int32_t);
        mIndexBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), index_buffer_size);

        auto srcBufferMap = mSrcBuffer.map<gpu_types::float4>();
        test_utils::fill_random_pixels<gpu_types::float4>(srcBufferMap.get(), srcBufferMap.get() + mBufferWidth);

        auto mappedIndices = mIndexBuffer.map<int32_t>();
        std::iota(mappedIndices.get(), mappedIndices.get() + mBufferWidth, 0);
    }

    void Test::prepare()
    {
        auto dstBufferMap = mDstBuffer.map<gpu_types::float4>();
        test_utils::fill_random_pixels<gpu_types::float4>(dstBufferMap.get(), dstBufferMap.get() + mBufferWidth);
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mIndexBuffer,
                      mSrcBuffer,
                      mDstBuffer,
                      mBufferWidth);
    }

    test_utils::Evaluation Test::checkResults(bool verbose)
    {
        auto srcBufferMap = mSrcBuffer.map<gpu_types::float4>();
        auto dstBufferMap = mDstBuffer.map<gpu_types::float4>();
        return test_utils::check_results(srcBufferMap.get(),
                                         dstBufferMap.get(),
                                         vk::Extent3D(mBufferWidth, 1, 1),
                                         mBufferWidth,
                                         verbose);
    }


    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose) {
        test_utils::InvocationResult invocationResult;

        Test t(kernel.getDevice(), args);

        t.prepare();
        invocationResult.mExecutionTime = t.run(kernel);
        invocationResult.mEvaluation = t.checkResults(verbose);

        return invocationResult;
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        test_utils::InvocationTest t({ "", test });
        return test_utils::KernelTest::invocation_tests({ t });
    }
}

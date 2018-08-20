//
// Created by Eric Berdahl on 10/31/17.
//

#include "fillarraystruct_kernel.hpp"

namespace {

    const unsigned int kWrapperArraySize = 18;

    typedef struct {
        float arr[kWrapperArraySize];
    } FloatArrayWrapper;

    static_assert(sizeof(FloatArrayWrapper) == kWrapperArraySize*sizeof(float), "bad size for FloatArrayWrapper");
}

namespace fillarraystruct_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    destination_buffer,
           unsigned int                     num_elements)
    {
        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (num_elements + workgroup_sizes.width - 1) / workgroup_sizes.width,
                1,
                1);

        clspv_utils::kernel_invocation invocation = kernel.createInvocation();

        invocation.addStorageBufferArgument(destination_buffer);
        return invocation.run(num_workgroups);
    }

    test_utils::InvocationResult test(clspv_utils::kernel&              kernel,
                                      const std::vector<std::string>&   args,
                                      bool                              verbose) {
        test_utils::InvocationResult invocationResult;
        auto& device = kernel.getDevice();

        const int buffer_width = 32;
        const int num_floats_in_struct = sizeof(FloatArrayWrapper) / sizeof(float);

        // allocate destination buffer
        const std::size_t buffer_size = buffer_width * sizeof(FloatArrayWrapper);
        const int num_floats_in_buffer = num_floats_in_struct * buffer_width;
        vulkan_utils::storage_buffer dst_buffer(device.mDevice, device.mMemoryProperties, buffer_size);

        auto dstBufferMap = dst_buffer.map<float>();
        test_utils::fill_random_pixels<float>(dstBufferMap.get(), dstBufferMap.get() + num_floats_in_buffer);
        dstBufferMap.reset();

        auto dstFloatArrayMap = dst_buffer.map<FloatArrayWrapper>();
        std::vector<FloatArrayWrapper> expectedResults(dstFloatArrayMap.get(), dstFloatArrayMap.get() + buffer_width);
        for (unsigned int wrapperIndex = 0; wrapperIndex < buffer_width; ++wrapperIndex) {
            for (unsigned int elementIndex = 0; elementIndex < kWrapperArraySize; ++elementIndex) {
                expectedResults[wrapperIndex].arr[elementIndex] = sizeof(FloatArrayWrapper) * 10000.0f
                                          + wrapperIndex * 100.0f
                                          + (float) elementIndex;

            }
        }
        dstFloatArrayMap.reset();
        assert(expectedResults.size() == buffer_width);

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dst_buffer,
                                                 buffer_width);

        dstBufferMap = dst_buffer.map<float>();
        test_utils::check_results(reinterpret_cast<float*>(expectedResults.data()),
                                  dstBufferMap.get(),
                                  vk::Extent3D(num_floats_in_struct, buffer_width, 1),
                                  num_floats_in_struct,
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

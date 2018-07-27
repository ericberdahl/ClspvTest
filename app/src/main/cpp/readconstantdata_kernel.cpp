//
// Created by Eric Berdahl on 10/31/17.
//

#include "readconstantdata_kernel.hpp"

namespace readconstantdata_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              width)
    {
        struct scalar_args {
            int inWidth;            // offset 0
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().mDevice,
                                                  kernel.getDevice().mMemoryProperties,
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = width;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                1,
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

        const int buffer_height = 1;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // number of elements in the constant data array (in the kernel itself)
        const std::size_t constant_data_length = 10;

        // allocate buffers and images
        vulkan_utils::storage_buffer  dstBuffer(device.mDevice, device.mMemoryProperties, buffer_size);

        // initialize destination memory with random data
        auto dstBufferMap = dstBuffer.map<float>();
        test_utils::fill_random_pixels<float>(dstBufferMap.get(), dstBufferMap.get() + buffer_length);
        dstBufferMap.reset();

        // set up expected results of the destination buffer
        int index = 0;
        std::vector<float> expectedResults(buffer_length);
        std::generate(expectedResults.begin(), expectedResults.end(), [&index, buffer_length, constant_data_length]() {
            float result = std::pow(2.0f, index);
            if (index >= std::min(buffer_length, constant_data_length)) {
                result = -1.0f;
            }

            ++index;

            return result;
        });

        invocationResult.mExecutionTime = invoke(kernel,
                                                 dstBuffer,
                                                 buffer_width);

        dstBufferMap = dstBuffer.map<float>();
        test_utils::check_results(expectedResults.data(),
                                  dstBufferMap.get(),
                                  buffer_width, buffer_height, 1,
                                  buffer_width,
                                  verbose,
                                  invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }
}

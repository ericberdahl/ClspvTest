//
// Created by Eric Berdahl on 10/31/17.
//

#include "readconstantdata_kernel.hpp"

#include "clspv_utils/kernel.hpp"

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

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().getDevice(),
                                                  kernel.getDevice().getMemoryProperties(),
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = width;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                1,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
        mBufferExtent(64, 1, 1)
    {
        auto& device = kernel.getDevice();

        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // number of elements in the constant data array (in the kernel itself)
        const std::size_t constant_data_length = 12;

        // allocate buffers and images
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), buffer_size);

        // set up expected results of the destination buffer
        int index = 0;
        mExpectedResults.resize(buffer_length);
        std::generate(mExpectedResults.begin(), mExpectedResults.end(), [&index, buffer_length, constant_data_length]() {
            float result = std::pow(2.0f, index);
            if (index >= std::min(buffer_length, constant_data_length)) {
                result = -1.0f;
            }

            ++index;

            return result;
        });
    }

    void Test::prepare()
    {
        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

        // initialize destination memory with random data
        auto dstBufferMap = mDstBuffer.map<float>();
        test_utils::fill_random_pixels<float>(dstBufferMap.get(), dstBufferMap.get() + buffer_length);
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mDstBuffer,
                      mBufferExtent.width);
    }

    test_utils::Evaluation Test::evaluate(bool verbose)
    {
        auto dstBufferMap = mDstBuffer.map<float>();
        return test_utils::check_results(mExpectedResults.data(),
                                         dstBufferMap.get(),
                                         mBufferExtent,
                                         mBufferExtent.width,
                                         verbose);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        test_utils::InvocationTest t({ "", test_utils::run_test<Test> });
        return test_utils::KernelTest::invocation_tests({ t });
    }

}

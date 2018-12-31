//
// Created by Eric Berdahl on 10/31/17.
//

#include "fillarraystruct_kernel.hpp"

#include "clspv_utils/kernel.hpp"

namespace {
    using namespace fillarraystruct_kernel;

    const int num_floats_in_struct = sizeof(Test::FloatArrayWrapper) / sizeof(float);

    static_assert(sizeof(Test::FloatArrayWrapper) == Test::kWrapperArraySize*sizeof(float), "bad size for FloatArrayWrapper");
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

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(destination_buffer);
        return invocation.run(num_workgroups);
    }

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
        mBufferWidth(32)
    {
        auto& device = kernel.getDevice();

        // allocate destination buffer
        const std::size_t buffer_size = mBufferWidth * sizeof(FloatArrayWrapper);
        const int num_floats_in_buffer = num_floats_in_struct * mBufferWidth;
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), buffer_size);

        mExpectedResults.resize(mBufferWidth);
    }

    void Test::prepare()
    {
        const int num_floats_in_buffer = num_floats_in_struct * mBufferWidth;

        auto dstBufferMap = mDstBuffer.map<float>();
        test_utils::fill_random_pixels<float>(dstBufferMap.get(), dstBufferMap.get() + num_floats_in_buffer);
        dstBufferMap.reset();

        auto dstFloatArrayMap = mDstBuffer.map<FloatArrayWrapper>();
        mExpectedResults.assign(dstFloatArrayMap.get(), dstFloatArrayMap.get() + mBufferWidth);
        for (unsigned int wrapperIndex = 0; wrapperIndex < mBufferWidth; ++wrapperIndex) {
            for (unsigned int elementIndex = 0; elementIndex < kWrapperArraySize; ++elementIndex) {
                mExpectedResults[wrapperIndex].arr[elementIndex] = sizeof(FloatArrayWrapper) * 10000.0f
                                                                  + wrapperIndex * 100.0f
                                                                  + (float) elementIndex;

            }
        }
        dstFloatArrayMap.reset();

        assert(mExpectedResults.size() == mBufferWidth);

    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mDstBuffer,
                      mBufferWidth);
    }

    test_utils::Evaluation Test::evaluate(bool verbose)
    {
        auto dstBufferMap = mDstBuffer.map<float>();
        return test_utils::check_results(reinterpret_cast<float*>(mExpectedResults.data()),
                                         dstBufferMap.get(),
                                         vk::Extent3D(num_floats_in_struct, mBufferWidth, 1),
                                         num_floats_in_struct,
                                         verbose);
    }

    test_utils::KernelTest::invocation_tests getAllTestVariants()
    {
        test_utils::InvocationTest t({ "", test_utils::run_test<Test> });
        return test_utils::KernelTest::invocation_tests({ t });
    }
}

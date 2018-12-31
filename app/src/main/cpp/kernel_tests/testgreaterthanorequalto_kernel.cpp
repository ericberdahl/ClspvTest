//
// Created by Eric Berdahl on 10/31/17.
//

#include "testgreaterthanorequalto_kernel.hpp"

#include "clspv_utils/kernel.hpp"

namespace testgreaterthanorequalto_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::storage_buffer&    dst_buffer,
           vk::Extent3D                     extent)
    {
        if (1 != extent.depth)
        {
            throw std::runtime_error("Depth must be 1");
        }

        struct scalar_args {
            int inWidth;            // offset 0
            int inHeight;           // offset 4
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(4 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        vulkan_utils::uniform_buffer scalarBuffer(kernel.getDevice().getDevice(),
                                                  kernel.getDevice().getMemoryProperties(),
                                                  sizeof(scalar_args));
        auto scalars = scalarBuffer.map<scalar_args>();
        scalars->inWidth = extent.width;
        scalars->inHeight = extent.height;
        scalars.reset();

        const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
        const vk::Extent3D num_workgroups(
                (extent.width + workgroup_sizes.width - 1) / workgroup_sizes.width,
                (extent.height + workgroup_sizes.height - 1) / workgroup_sizes.height,
                1);

        clspv_utils::invocation invocation(kernel.createInvocationReq());

        invocation.addStorageBufferArgument(dst_buffer);
        invocation.addUniformBufferArgument(scalarBuffer);

        return invocation.run(num_workgroups);
    }

    Test::Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args) :
            mBufferExtent(64, 64, 1)
    {
        auto& device = kernel.getDevice();

        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // allocate buffers and images
        mDstBuffer = vulkan_utils::storage_buffer(device.getDevice(), device.getMemoryProperties(), buffer_size);

        // set up expected results of the destination buffer
        int index = 0;
        mExpectedResults.resize(buffer_length);
        std::generate(mExpectedResults.begin(), mExpectedResults.end(), [&index, this]() {
            int x = index % this->mBufferExtent.width;
            int y = index / this->mBufferExtent.width;

            ++index;

            return (x >= 0 && y >= 0 && x < this->mBufferExtent.width && y < this->mBufferExtent.height ? 1.0f : 0.0f);
        });
    }

    void Test::prepare()
    {
        const std::size_t buffer_length = mBufferExtent.width * mBufferExtent.height * mBufferExtent.depth;

        // initialize destination memory with unexpected value. the kernel should write either 0 or
        // 1. so, initialize the destination with 2.
        auto dstBufferMap = mDstBuffer.map<float>();
        std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, 2.0f);
        dstBufferMap.reset();
    }

    clspv_utils::execution_time_t Test::run(clspv_utils::kernel& kernel)
    {
        return invoke(kernel,
                      mDstBuffer,
                      mBufferExtent);
    }

    test_utils::Evaluation Test::evaluate(bool verbose)
    {
        auto dstBufferMap = mDstBuffer.map<float>();
        return test_utils::check_results(mExpectedResults.data(), dstBufferMap.get(),
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

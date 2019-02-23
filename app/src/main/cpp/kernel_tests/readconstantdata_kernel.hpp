//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_READCONSTANTDATA_KERNEL_HPP
#define CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace readconstantdata_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&     kernel,
           vulkan_utils::buffer&    dst_buffer,
           int                      width);

    struct Test : public test_utils::Test
    {
        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args);

        virtual void prepare() override;

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override;

        virtual test_utils::Evaluation evaluate(bool verbose) override;

        vk::Extent3D            mBufferExtent;
        vulkan_utils::buffer    mDstBuffer;
        std::vector<float>      mExpectedResults;
    };

    test_utils::KernelTest::invocation_tests getAllTestVariants();
}

#endif // CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

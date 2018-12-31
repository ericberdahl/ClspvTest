//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_RESAMPLE3DIMAGE_KERNEL_HPP
#define CLSPVTEST_RESAMPLE3DIMAGE_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vector>


namespace resample3dimage_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel &kernel,
           vulkan_utils::image &src_image,
           vulkan_utils::storage_buffer &dst_buffer,
           int width,
           int height,
           int depth);

    struct Test : public test_utils::Test
    {
        typedef gpu_types::float4 BufferPixelType;
        typedef gpu_types::float4 ImagePixelType;

        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args);

        virtual void prepare() override;

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override;

        virtual test_utils::Evaluation evaluate(bool verbose) override;

        vk::Extent3D                    mBufferExtent;
        vulkan_utils::image             mSrcImage;
        vulkan_utils::staging_buffer    mSrcImageStaging;
        vulkan_utils::storage_buffer    mDstBuffer;
        std::vector<BufferPixelType>    mExpectedDstBuffer;
        vk::UniqueCommandBuffer         mSetupCommand;
    };

    test_utils::KernelTest::invocation_tests getAllTestVariants();

}

#endif //CLSPVTEST_RESAMPLE3DIMAGE_KERNEL_HPP

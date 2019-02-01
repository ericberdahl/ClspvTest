//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_GENERIC_KERNEL_HPP
#define CLSPVTEST_GENERIC_KERNEL_HPP

#include "clspv_utils/clspv_utils_fwd.hpp"
#include "test_utils.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace generic_kernel {

    struct Test : public test_utils::Test
    {
        typedef std::vector<vulkan_utils::storage_buffer>   storage_list;
        typedef std::vector<vulkan_utils::uniform_buffer>   uniform_list;
        typedef std::vector<std::size_t>                    local_size_list;

        enum arg_kind
        {
            kind_storageBuffer,
            kind_uniformBuffer,
            kind_LocalArraySize
        };

        Test(clspv_utils::kernel& kernel, const std::vector<std::string>& args);

        virtual std::string getParameterString() const override;

        virtual void prepare() override;

        virtual clspv_utils::execution_time_t run(clspv_utils::kernel& kernel) override;

        virtual test_utils::Evaluation evaluate(bool verbose) override;

        std::string             mParameterString;

        storage_list            mStorageBuffers;
        uniform_list            mUniformBuffers;
        local_size_list         mLocalArraySizes;
        std::vector<arg_kind>   mArgOrder;

        vk::Extent3D        mNumWorkgroups;
    };

    test_utils::KernelTest::invocation_tests getAllTestVariants();
}

#endif //CLSPVTEST_GENERIC_KERNEL_HPP

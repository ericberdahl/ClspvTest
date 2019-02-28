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
        typedef std::vector<vulkan_utils::buffer>           storage_list;
        typedef std::vector<vulkan_utils::buffer>           uniform_list;
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

        // TODO unify storage_list and uniform_list into buffer_list

        storage_list            mStorageBuffers;
        uniform_list            mUniformBuffers;
        local_size_list         mLocalArraySizes;
        std::vector<arg_kind>   mArgOrder;

        vk::Extent3D        mNumWorkgroups;
    };

    class formatter
    {
    public:
        formatter(const char*           entryPoint,
                  const char*           label,
                  unsigned int          numIterations,
                  const vk::Extent3D&   workgroupSizes,
                  const vk::Extent3D&   numWorkgroups);

        void    addStorageBufferArgument(vulkan_utils::buffer& buffer);
        void    addUniformBufferArgument(vulkan_utils::buffer& buffer);
        void    addReadOnlyImageArgument(vulkan_utils::image& image);
        void    addWriteOnlyImageArgument(vulkan_utils::image& image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        std::string getString() const { return mStream.str(); }

    private:
        std::ostringstream  mStream;
    };

    test_utils::KernelTest::invocation_tests getAllTestVariants();
}

#endif //CLSPVTEST_GENERIC_KERNEL_HPP

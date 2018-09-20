//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_INVOCATION_HPP
#define CLSPVUTILS_KERNEL_INVOCATION_HPP

#include "clspv_utils_fwd.hpp"

#include <chrono>
#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulkan_utils/vulkan_utils.hpp"

#include "arg_spec.hpp"
#include "device.hpp"

namespace clspv_utils {

    struct execution_time_t {
        struct vulkan_timestamps_t {
            uint64_t start          = 0;
            uint64_t host_barrier   = 0;
            uint64_t execution      = 0;
            uint64_t gpu_barrier    = 0;
        };

        execution_time_t();

        std::chrono::duration<double>   cpu_duration;
        vulkan_timestamps_t             timestamps;
    };

    class kernel_invocation {
    public:
        typedef vk::ArrayProxy<const arg_spec_t> arg_list_proxy_t;

                    kernel_invocation();

        explicit    kernel_invocation(kernel&           kernel,
                                      device            device,
                                      vk::DescriptorSet argumentDescSet,
                                      arg_list_proxy_t  argList);

                    kernel_invocation(kernel_invocation&& other);

                    ~kernel_invocation();

        void    addStorageBufferArgument(vulkan_utils::storage_buffer& buffer);
        void    addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer);
        void    addReadOnlyImageArgument(vulkan_utils::image& image);
        void    addWriteOnlyImageArgument(vulkan_utils::image& image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        execution_time_t    run(const vk::Extent3D& num_workgroups);

        void    swap(kernel_invocation& other);

    private:
        void    bindCommand();
        void    updatePipeline();
        void    fillCommandBuffer(const vk::Extent3D&    num_workgroups);
        void    updateDescriptorSets();
        void    submitCommand();

        // Sanity check that the nth argument (specified by ordinal) has the indicated
        // spvmap type. Throw an exception if false. Return the binding number if true.
        std::uint32_t   validateArgType(std::size_t ordinal, vk::DescriptorType kind) const;
        std::uint32_t   validateArgType(std::size_t ordinal, arg_spec_t::kind_t kind) const;

        std::size_t countArguments() const;

    private:
        enum QueryIndex {
            kQueryIndex_FirstIndex = 0,
            kQueryIndex_StartOfExecution = 0,
            kQueryIndex_PostHostBarrier = 1,
            kQueryIndex_PostExecution = 2,
            kQueryIndex_PostGPUBarrier= 3,
            kQueryIndex_Count = 4
        };

    private:
        kernel*                                 mKernel = nullptr;
        device                                  mDevice;
        arg_list_proxy_t                        mArgList;
        vk::UniqueCommandBuffer                 mCommand;
        vk::UniqueQueryPool                     mQueryPool;

        vk::DescriptorSet                       mArgumentDescriptorSet;

        std::vector<vk::BufferMemoryBarrier>    mBufferMemoryBarriers;
        std::vector<vk::ImageMemoryBarrier>     mImageMemoryBarriers;

        std::vector<vk::DescriptorImageInfo>    mImageArgumentInfo;
        std::vector<vk::DescriptorBufferInfo>   mBufferArgumentInfo;

        std::vector<vk::WriteDescriptorSet>     mArgumentDescriptorWrites;
        std::vector<int32_t>                    mSpecConstantArguments;
    };

    inline void swap(kernel_invocation & lhs, kernel_invocation & rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_INVOCATION_HPP

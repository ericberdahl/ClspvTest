//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_INVOCATION_HPP
#define CLSPVUTILS_KERNEL_INVOCATION_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "interface.hpp"
#include "invocation_req.hpp"

#include <chrono>
#include <memory>

#include <vulkan/vulkan.hpp>

#include "vulkan_utils/vulkan_utils.hpp"

// TODO invocation should dispatch, not run. taking a command buffer argument that the client owns
// TODO invocation should return execution time via a getter, not the result of run

namespace clspv_utils {

    struct execution_time_t {
        struct vulkan_timestamps {
            uint64_t start          = 0;
            uint64_t host_barrier   = 0;
            uint64_t execution      = 0;
            uint64_t gpu_barrier    = 0;
        };

        execution_time_t();

        std::chrono::duration<double>   cpu_duration;
        vulkan_timestamps               timestamps;
    };

    class invocation {
    public:
                    invocation();

        explicit    invocation(invocation_req_t  req);

                    invocation(invocation&& other);

                    ~invocation();

        void    addStorageBufferArgument(vulkan_utils::buffer& buffer);
        void    addUniformBufferArgument(vulkan_utils::buffer& buffer);
        void    addReadOnlyImageArgument(vulkan_utils::image& image);
        void    addWriteOnlyImageArgument(vulkan_utils::image& image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        execution_time_t    run(const vk::Extent3D& num_workgroups);

        void    swap(invocation& other);

    private:
        void    fillCommandBuffer(const vk::Extent3D&    num_workgroups);
        void    updateDescriptorSets();
        void    submitCommand();

        // Sanity check that the nth argument (specified by ordinal) has the indicated
        // spvmap type. Throw an exception if false. Return the binding number if true.
        std::uint32_t   validateArgType(std::size_t ordinal, vk::DescriptorType kind) const;
        std::uint32_t   validateArgType(std::size_t ordinal, arg_spec_t::kind kind) const;

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
        invocation_req_t                    mReq;
        vk::UniqueCommandBuffer             mCommand;
        vk::UniqueQueryPool                 mQueryPool;

        vector<vk::BufferMemoryBarrier>     mBufferMemoryBarriers;
        vector<vk::ImageMemoryBarrier>      mImageMemoryBarriers;

        vector<vk::DescriptorImageInfo>     mImageArgumentInfo;
        vector<vk::DescriptorBufferInfo>    mBufferArgumentInfo;

        vector<vk::WriteDescriptorSet>      mArgumentDescriptorWrites;
        vector<std::uint32_t>               mSpecConstantArguments;
    };

    inline void swap(invocation & lhs, invocation & rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_INVOCATION_HPP

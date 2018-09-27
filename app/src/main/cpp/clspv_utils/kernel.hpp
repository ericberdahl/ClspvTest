//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_HPP
#define CLSPVUTILS_KERNEL_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "interface.hpp"
#include "kernel_req.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    class kernel {
    public:
        kernel();

        kernel(kernel_req_t         layout,
               const vk::Extent3D&  workgroup_sizes);

        kernel(kernel&& other);

        ~kernel();

        kernel&             operator=(kernel&& other);

        kernel_invocation   createInvocation();
        void                bindCommand(vk::CommandBuffer command) const;

        string              getEntryPoint() const { return mLayout.mKernelSpec.mName; }
        vk::Extent3D        getWorkgroupSize() const { return mWorkgroupSizes; }

        const device&       getDevice() { return mLayout.mDevice; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

        void                swap(kernel& other);

    private:
        kernel_req_t                    mLayout;
        vk::UniqueDescriptorSetLayout   mArgumentsLayout;
        vk::UniqueDescriptorSet         mArgumentsDescriptor;
        vk::UniquePipelineLayout        mPipelineLayout;
        vk::UniquePipeline              mPipeline;
        vk::Extent3D                    mWorkgroupSizes;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_HPP

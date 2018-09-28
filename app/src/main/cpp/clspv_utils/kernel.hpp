//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_HPP
#define CLSPVUTILS_KERNEL_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "interface.hpp"
#include "invocation_req.hpp"
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

        string              getEntryPoint() const { return mReq.mKernelSpec.mName; }
        vk::Extent3D        getWorkgroupSize() const { return vk::Extent3D(mSpecConstants[0], mSpecConstants[1], mSpecConstants[2]); }

        const device&       getDevice() { return mReq.mDevice; }

        vk::Pipeline        updatePipeline(vk::ArrayProxy<uint32_t> otherSpecConstants);

        void                swap(kernel& other);

        invocation_req_t    createInvocationReq();

    private:
        typedef vector<std::uint32_t>   spec_constant_list;

    private:
        kernel_req_t                    mReq;
        vk::UniqueDescriptorSetLayout   mArgumentsLayout;
        vk::UniqueDescriptorSet         mArgumentsDescriptor;
        vk::UniquePipelineLayout        mPipelineLayout;
        vk::UniquePipeline              mPipeline;
        spec_constant_list              mSpecConstants;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_HPP

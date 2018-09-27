//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_HPP
#define CLSPVUTILS_KERNEL_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "interface.hpp"
#include "kernel_layout.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    class kernel {
    public:
        typedef vk::ArrayProxy<const arg_spec_t> arg_list_proxy_t;

        kernel();

        kernel(device               dev,
               kernel_layout_t      layout,
               vk::ShaderModule     shaderModule,
               vk::PipelineCache    pipelineCache,
               string               entryPoint,
               const vk::Extent3D&  workgroup_sizes,
               arg_list_proxy_t     args);

        kernel(kernel&& other);

        ~kernel();

        kernel&             operator=(kernel&& other);

        kernel_invocation   createInvocation();
        void                bindCommand(vk::CommandBuffer command) const;

        string              getEntryPoint() const { return mEntryPoint; }
        vk::Extent3D        getWorkgroupSize() const { return mWorkgroupSizes; }

        const device&       getDevice() { return mDevice; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

        void                swap(kernel& other);

    private:
        device              mDevice;
        vk::ShaderModule    mShaderModule;
        string              mEntryPoint;
        vk::Extent3D        mWorkgroupSizes;
        kernel_layout_t     mLayout;
        vk::PipelineCache   mPipelineCache;
        vk::UniquePipeline  mPipeline;
        arg_list_proxy_t    mArgList;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_HPP

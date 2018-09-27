//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_REQ_HPP
#define CLSPVUTILS_KERNEL_REQ_HPP

#include "clspv_utils_fwd.hpp"

#include "device.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    struct kernel_req_t {
        device                          mDevice;
        kernel_spec_t                   mKernelSpec;
        vk::ShaderModule                mShaderModule;
        vk::PipelineCache               mPipelineCache;
        vk::DescriptorSet               mLiteralSamplerDescriptor;
        vk::DescriptorSetLayout         mLiteralSamplerLayout;
    };
}

#endif //CLSPVUTILS_KERNEL_REQ_HPP

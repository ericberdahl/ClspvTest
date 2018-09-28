//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_INVOCATION_REQ_HPP
#define CLSPVUTILS_INVOCATION_REQ_HPP

#include "clspv_utils_fwd.hpp"

#include "device.hpp"

#include <vulkan/vulkan.hpp>

#include <functional>

namespace clspv_utils {

    struct invocation_req_t {
        typedef std::function<vk::Pipeline (vk::ArrayProxy<std::uint32_t>)> get_pipeline_fn;

        device              mDevice;
        kernel_spec_t       mKernelSpec;

        vk::PipelineLayout  mPipelineLayout;
        get_pipeline_fn     mGetPipelineFn;

        vk::DescriptorSet   mLiteralSamplerDescriptor;
        vk::DescriptorSet   mArgumentsDescriptor;
    };
}

#endif //CLSPVUTILS_INVOCATION_REQ_HPP

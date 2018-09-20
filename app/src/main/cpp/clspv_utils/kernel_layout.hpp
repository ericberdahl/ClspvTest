//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_LAYOUT_HPP
#define CLSPVUTILS_KERNEL_LAYOUT_HPP

#include "clspv_utils_fwd.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    struct kernel_layout_t {
        vk::DescriptorSet               mLiteralSamplerDescriptor;

        vk::UniqueDescriptorSetLayout   mArgumentDescriptorLayout;
        vk::UniqueDescriptorSet         mArgumentsDescriptor;

        vk::UniquePipelineLayout        mPipelineLayout;
    };
}

#endif //CLSPVUTILS_KERNEL_LAYOUT_HPP

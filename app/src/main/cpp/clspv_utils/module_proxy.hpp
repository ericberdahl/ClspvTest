//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_MODULE_PROXY_HPP
#define CLSPVUTILS_MODULE_PROXY_HPP

#include "clspv_utils_fwd.hpp"

#include "kernel_interface.hpp" // TODO: break dependency on kernel_interface
#include "sampler_spec.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    struct module_proxy_t {
        vk::ArrayProxy<const sampler_spec_t>    mSamplers;
        vk::ArrayProxy<const kernel_spec_t>     mKernels;
    };
}

#endif // CLSPVUTILS_MODULE_PROXY_HPP

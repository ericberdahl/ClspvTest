//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_INTERFACE_HPP
#define CLSPVUTILS_KERNEL_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include "arg_spec.hpp"
#include "clspv_utils_interop.hpp"
#include "sampler_spec.hpp"

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    struct kernel_spec_t {
        typedef vector<arg_spec_t>  arg_list;

        string      mName;
        arg_list    mArguments;
    };

    vk::UniqueDescriptorSetLayout createKernelArgumentDescriptorLayout(const kernel_spec_t::arg_list&   arguments,
                                                                       const device&                    inDevice);

    int     getKernelArgumentDescriptorSet(const kernel_spec_t::arg_list& arguments);

    void    validateKernelArg(const arg_spec_t &arg);

    void    validateKernelSpec(const kernel_spec_t& spec);

    vk::DescriptorType  getDescriptorType(arg_spec_t::kind_t argKind);

    // Sort the args such that pods are grouped together at the end of the sequence, and that
    // the non-pod and pod groups are each individually sorted by increasing ordinal
    void    standardizeKernelArgumentOrder(kernel_spec_t::arg_list& arguments);
}

#endif //CLSPVUTILS_KERNEL_INTERFACE_HPP

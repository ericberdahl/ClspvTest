//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_MODULE_INTERFACE_HPP
#define CLSPVUTILS_MODULE_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"

#include <iosfwd>

namespace clspv_utils {

    struct module_spec_t {
        typedef vector<sampler_spec_t>  sampler_list;
        typedef vector<kernel_spec_t>   kernel_list;

        sampler_list  mSamplers;
        kernel_list   mKernels;
    };

    module_spec_t           createModuleSpec(std::istream& spvmapStream);

    const kernel_spec_t*    findKernelSpec(const string&                        name,
                                           const module_spec_t::kernel_list&    kernels);

    kernel_spec_t*          findKernelSpec(const string&                name,
                                           module_spec_t::kernel_list&  kernels);

    int                     getSamplersDescriptorSet(const module_spec_t::sampler_list& spec);

    vector<string>          getEntryPointNames(const module_spec_t::kernel_list& specs);

    void                    validateModuleSpec(const module_spec_t& spec);
}

#endif //CLSPVUTILS_MODULE_INTERFACE_HPP

//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_MODULE_INTERFACE_HPP
#define CLSPVUTILS_MODULE_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"

#include <iosfwd>

namespace clspv_utils {

    class module_interface {
    public:
        typedef vector<sampler_spec_t>     sampler_list_t;
        typedef vector<kernel_interface>   kernel_list_t;

                                module_interface();

        explicit                module_interface(std::istream& spvmapStream);

        const kernel_interface* findKernelInterface(const string& entryPoint) const;

        vector<string>          getEntryPoints() const;

        module_proxy_t          createModuleProxy() const;

    private:
        void    addLiteralSampler(sampler_spec_t sampler);

    private:
        sampler_list_t  mSamplers;
        kernel_list_t   mKernels;
    };
}

#endif //CLSPVUTILS_MODULE_INTERFACE_HPP

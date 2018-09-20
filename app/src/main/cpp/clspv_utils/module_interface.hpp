//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_MODULE_INTERFACE_HPP
#define CLSPVUTILS_MODULE_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include <string>
#include <vector>

namespace clspv_utils {

    class module_interface {
    public:
        typedef std::vector<sampler_spec_t>     sampler_list_t;
        typedef std::vector<kernel_interface>   kernel_list_t;

                                        module_interface();

        explicit                        module_interface(const std::string& moduleName);

        const kernel_interface*         findKernelInterface(const std::string& entryPoint) const;

        std::vector<std::string>        getEntryPoints() const;

        int                             getLiteralSamplersDescriptorSet() const;

        kernel_module                   load(device dev) const;

    private:
        void    addLiteralSampler(sampler_spec_t sampler);

    private:
        std::string     mName;
        sampler_list_t  mSamplers;
        kernel_list_t   mKernels;
    };
}

#endif //CLSPVUTILS_MODULE_INTERFACE_HPP

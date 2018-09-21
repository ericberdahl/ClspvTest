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

    class kernel_interface {
    public:
        typedef vector<arg_spec_t>                      arg_list_t;
        typedef vk::ArrayProxy<const arg_spec_t>        arg_list_proxy_t;
        typedef vk::ArrayProxy<const sampler_spec_t>    sampler_list_proxy_t;

        kernel_interface();

        kernel_interface(string                 entryPoint,
                         sampler_list_proxy_t   samplers,
                         arg_list_t             arguments);

        int                             getArgDescriptorSet() const;
        const string&                   getEntryPoint() const { return mName; }

        sampler_list_proxy_t            getLiteralSamplers() const { return mLiteralSamplers; }
        int                             getLiteralSamplersDescriptorSet() const;

        arg_list_proxy_t                getArguments() const { return mArguments; }

    private:
        void        validate() const;

    private:
        string                  mName;
        sampler_list_proxy_t    mLiteralSamplers;
        arg_list_t              mArguments;
    };

    vk::UniqueDescriptorSetLayout createArgumentDescriptorLayout(const kernel_interface& kernel, const device& inDevice);
}

#endif //CLSPVUTILS_KERNEL_INTERFACE_HPP

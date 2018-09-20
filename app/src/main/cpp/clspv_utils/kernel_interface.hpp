//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_INTERFACE_HPP
#define CLSPVUTILS_KERNEL_INTERFACE_HPP

#include "clspv_utils_fwd.hpp"

#include "arg_spec.hpp"
#include "sampler_spec.hpp"

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    class kernel_interface {
    public:
        typedef std::vector<arg_spec_t>                 arg_list_t;
        typedef vk::ArrayProxy<const sampler_spec_t>    sampler_list_proxy_t;

        kernel_interface();

        kernel_interface(std::string           entryPoint,
                         sampler_list_proxy_t  samplers,
                         arg_list_t            arguments);

        int                         getArgDescriptorSet() const;
        const std::string&          getEntryPoint() const { return mName; }
        vk::UniqueDescriptorSetLayout createArgDescriptorLayout(const device& dev) const;

        const sampler_list_proxy_t& getLiteralSamplers() const { return mLiteralSamplers; }
        int                         getLiteralSamplersDescriptorSet() const;

    private:
        void        validate() const;

    private:
        std::string             mName;
        sampler_list_proxy_t    mLiteralSamplers;
    public:
        arg_list_t              mArgSpecs;  // TODO: make mArgSpecs private
    };
}

#endif //CLSPVUTILS_KERNEL_INTERFACE_HPP

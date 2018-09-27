//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_MODULE_HPP
#define CLSPVUTILS_MODULE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "interface.hpp"

#include <vulkan/vulkan.hpp>

#include <iosfwd>

namespace clspv_utils {

    class module {
    public:
                            module();

                            module(module&& other);

                            module(std::istream& spvModuleStream,
                                   device        dev,
                                   module_spec_t spec);

                            ~module();

        module&             operator=(module&& other);

        void                swap(module& other);

        bool                isLoaded() const { return (bool)(*mShaderModule); }

        vector<string>      getEntryPoints() const;

        kernel_req_t        createKernelReq(const string &entryPoint) const;

    private:
        device                  mDevice;
        module_spec_t           mModuleSpec;

        vk::DescriptorSetLayout mLiteralSamplerDescriptorLayout;
        vk::DescriptorSet       mLiteralSamplerDescriptor;
        vk::UniqueShaderModule  mShaderModule;
        vk::UniquePipelineCache mPipelineCache;
    };

    inline void swap(module& lhs, module& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_MODULE_HPP

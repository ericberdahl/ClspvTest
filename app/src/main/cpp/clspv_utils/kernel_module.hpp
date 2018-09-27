//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_MODULE_HPP
#define CLSPVUTILS_KERNEL_MODULE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "module_interface.hpp"

#include <vulkan/vulkan.hpp>

#include <iosfwd>

namespace clspv_utils {

    class kernel_module {
    public:
        typedef vk::ArrayProxy<const kernel_spec_t>  kernel_list_proxy_t;

                            kernel_module();

                            kernel_module(kernel_module&& other);

                            kernel_module(const string& moduleName,
                                          std::istream& spvModuleStream,
                                          device        dev,
                                          module_spec_t spec);

                            ~kernel_module();

        kernel_module&      operator=(kernel_module&& other);

        void                swap(kernel_module& other);

        kernel              createKernel(const string&          entryPoint,
                                         const vk::Extent3D&    workgroup_sizes);

        bool                isLoaded() const { return (bool)getShaderModule(); }

        string              getName() const { return mName; }
        vector<string>      getEntryPoints() const;
        vk::ShaderModule    getShaderModule() const { return *mShaderModule; }

    private:
        kernel_layout_t     createKernelLayout(const string& entryPoint) const;

    private:
        string                  mName;
        device                  mDevice;
        module_spec_t           mModuleSpec;

        vk::DescriptorSetLayout mLiteralSamplerDescriptorLayout;
        vk::DescriptorSet       mLiteralSamplerDescriptor;
        vk::UniqueShaderModule  mShaderModule;
        vk::UniquePipelineCache mPipelineCache;
    };

    inline void swap(kernel_module& lhs, kernel_module& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_MODULE_HPP

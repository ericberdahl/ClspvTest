//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_KERNEL_MODULE_HPP
#define CLSPVUTILS_KERNEL_MODULE_HPP

#include "clspv_utils_fwd.hpp"

#include "device.hpp"

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace clspv_utils {

    class kernel_module {
    public:
        typedef vk::ArrayProxy<const kernel_interface>  kernel_list_proxy_t;

                                    kernel_module();

                                    kernel_module(kernel_module&& other);

                                    kernel_module(const std::string&        moduleName,
                                                  device                    dev,
                                                  vk::DescriptorSet         literalSamplerDescriptor,
                                                  vk::DescriptorSetLayout   literalSamplerDescriptorLayout,
                                                  kernel_list_proxy_t       kernelInterfaces);

                                    ~kernel_module();

        kernel_module&              operator=(kernel_module&& other);

        void                        swap(kernel_module& other);

        kernel                      createKernel(const std::string&     entryPoint,
                                                 const vk::Extent3D&    workgroup_sizes);

        bool                        isLoaded() const { return (bool)getShaderModule(); }

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;
        vk::ShaderModule            getShaderModule() const { return *mShaderModule; }

    private:
        kernel_layout_t             createKernelLayout(const std::string& entryPoint) const;

    private:
        std::string                     mName;
        device                          mDevice;
        kernel_list_proxy_t             mKernelInterfaces;

        vk::DescriptorSetLayout         mLiteralSamplerDescriptorLayout;
        vk::DescriptorSet               mLiteralSamplerDescriptor;
        vk::UniqueShaderModule          mShaderModule;
        vk::UniquePipelineCache         mPipelineCache;
    };

    inline void swap(kernel_module& lhs, kernel_module& rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPVUTILS_KERNEL_MODULE_HPP

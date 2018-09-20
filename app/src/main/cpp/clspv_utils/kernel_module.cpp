//
// Created by Eric Berdahl on 10/22/17.
//

#include "kernel_module.hpp"

#include "kernel.hpp"
#include "kernel_interface.hpp"
#include "kernel_layout.hpp"

#include "file_utils.hpp"

#include <functional>
#include <memory>


namespace {
    using namespace clspv_utils;

    vk::UniqueShaderModule create_shader(vk::Device device, const std::string& spvFilename) {
        std::vector<std::uint32_t> spvModule;
        file_utils::read_file_contents(spvFilename, spvModule);

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.setCodeSize(spvModule.size() * sizeof(decltype(spvModule)::value_type))
                .setPCode(spvModule.data());

        return device.createShaderModuleUnique(shaderModuleCreateInfo);
    }

    /* TODO opportunity for sharing */
    vk::UniqueDescriptorSet allocate_descriptor_set(const device&           inDevice,
                                                    vk::DescriptorSetLayout layout)
    {
        vk::DescriptorSetAllocateInfo createInfo;
        createInfo.setDescriptorPool(inDevice.getDescriptorPool())
                .setDescriptorSetCount(1)
                .setPSetLayouts(&layout);

        return std::move(inDevice.getDevice().allocateDescriptorSetsUnique(createInfo)[0]);
    }

    vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                      device,
                                                    vk::ArrayProxy<const vk::DescriptorSetLayout>   layouts)
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayoutCount(layouts.size())
                .setPSetLayouts(layouts.data());

        return device.createPipelineLayoutUnique(createInfo);
    }

    const kernel_interface* find_kernel_interface(const std::string&                        name,
                                                  vk::ArrayProxy<const kernel_interface>    kernels)
    {
        auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                   [&name](const kernel_interface &iter) {
                                       return iter.getEntryPoint() == name;
                                   });

        return (kernel == kernels.end() ? nullptr : &(*kernel));
    }

    std::vector<std::string> get_entry_points(vk::ArrayProxy<const kernel_interface> kernels)
    {
        std::vector<std::string> result;

        std::transform(kernels.begin(), kernels.end(),
                       std::back_inserter(result),
                       [](const kernel_interface& k) { return k.getEntryPoint(); });

        return result;
    }

} // anonymous namespace

namespace clspv_utils {

    kernel_module::kernel_module()
            : mKernelInterfaces(nullptr)
    {
    }

    kernel_module::kernel_module(kernel_module&& other)
            : kernel_module()
    {
        swap(other);
    }

    kernel_module::kernel_module(const std::string&         moduleName,
                                 device                     inDevice,
                                 vk::DescriptorSet          literalSamplerDescriptor,
                                 vk::DescriptorSetLayout    literalSamplerDescriptorLayout,
                                 kernel_list_proxy_t        kernelInterfaces)
            : mName(moduleName),
              mDevice(inDevice),
              mKernelInterfaces(kernelInterfaces),
              mLiteralSamplerDescriptor(literalSamplerDescriptor),
              mLiteralSamplerDescriptorLayout(literalSamplerDescriptorLayout)
    {
        const std::string spvFilename = mName + ".spv";
        mShaderModule = create_shader(mDevice.getDevice(), spvFilename.c_str());
        mPipelineCache = mDevice.getDevice().createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
    }

    kernel_module::~kernel_module()
    {
    }

    kernel_module& kernel_module::operator=(kernel_module&& other)
    {
        swap(other);
        return *this;
    }

    void kernel_module::swap(kernel_module& other)
    {
        using std::swap;

        swap(mName, other.mName);
        swap(mDevice, other.mDevice);
        swap(mKernelInterfaces, other.mKernelInterfaces);
        swap(mLiteralSamplerDescriptorLayout, other.mLiteralSamplerDescriptorLayout);
        swap(mLiteralSamplerDescriptor, other.mLiteralSamplerDescriptor);
        swap(mShaderModule, other.mShaderModule);
        swap(mPipelineCache, other.mPipelineCache);
    }

    std::vector<std::string> kernel_module::getEntryPoints() const
    {
        return get_entry_points(mKernelInterfaces);
    }

    kernel_layout_t kernel_module::createKernelLayout(const std::string& entryPoint) const {
        if (!isLoaded()) {
            throw std::runtime_error("cannot create layout for unloaded module");
        }

        kernel_layout_t result;

        const auto kernelInterface = find_kernel_interface(entryPoint, mKernelInterfaces);
        if (!kernelInterface) {
            throw std::runtime_error("cannot create kernel layout for unknown entry point");
        }

        if (-1 != kernelInterface->getArgDescriptorSet()) {
            result.mArgumentDescriptorLayout = kernelInterface->createArgDescriptorLayout(mDevice);

            result.mArgumentsDescriptor = allocate_descriptor_set(mDevice,
                                                                  *result.mArgumentDescriptorLayout);
        }

        result.mLiteralSamplerDescriptor = mLiteralSamplerDescriptor;

        std::vector<vk::DescriptorSetLayout> layouts;
        if (mLiteralSamplerDescriptorLayout) layouts.push_back(mLiteralSamplerDescriptorLayout);
        if (result.mArgumentDescriptorLayout) layouts.push_back(*result.mArgumentDescriptorLayout);
        result.mPipelineLayout = create_pipeline_layout(mDevice.getDevice(), layouts);

        return result;
    }

    kernel kernel_module::createKernel(const std::string&   entryPoint,
                                       const vk::Extent3D&  workgroup_sizes)
    {
        return kernel(mDevice,
                      createKernelLayout(entryPoint),
                      *mShaderModule,
                      *mPipelineCache,
                      entryPoint,
                      workgroup_sizes,
                      find_kernel_interface(entryPoint, mKernelInterfaces)->mArgSpecs);
    }

} // namespace clspv_utils

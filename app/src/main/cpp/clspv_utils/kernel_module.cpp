//
// Created by Eric Berdahl on 10/22/17.
//

#include "kernel_module.hpp"

#include "kernel.hpp"
#include "kernel_interface.hpp"
#include "kernel_layout.hpp"
#include "module_proxy.hpp"

#include <istream>
#include <functional>
#include <memory>

namespace {
    using namespace clspv_utils;

    vk::UniqueShaderModule create_shader(vk::Device     device,
                                         std::istream&  in)
    {
        const auto savePos = in.tellg();
        in.seekg(0, std::ios_base::end);
        const auto num_bytes = in.tellg();
        in.seekg(savePos, std::ios_base::beg);

        const auto num_words = (num_bytes + std::streamoff(sizeof(std::uint32_t) - 1)) / sizeof(std::uint32_t);
        vector<std::uint32_t> spvModule(num_words);
        if (num_bytes != (spvModule.size() * sizeof(std::uint32_t)))
        {
            fail_runtime_error("spv module size is not multiple of uint32_t word size");
        }

        in.read(reinterpret_cast<char*>(spvModule.data()), num_bytes);

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

    const kernel_interface* find_kernel_interface(const string&                             name,
                                                  vk::ArrayProxy<const kernel_interface>    kernels)
    {
        auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                   [&name](const kernel_interface &iter) {
                                       return iter.getEntryPoint() == name;
                                   });

        return (kernel == kernels.end() ? nullptr : &(*kernel));
    }

    vector<string> get_entry_points(vk::ArrayProxy<const kernel_interface> kernels)
    {
        vector<string> result;

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

    kernel_module::kernel_module(const string&          moduleName,
                                 std::istream&          spvmoduleStream,
                                 device                 inDevice,
                                 const module_proxy_t&  proxy)
            : mName(moduleName),
              mDevice(inDevice),
              mKernelInterfaces(proxy.mKernels),
              mLiteralSamplerDescriptor(),
              mLiteralSamplerDescriptorLayout()
    {
        const auto literalSamplerDescriptorGroup = inDevice.getCachedSamplerDescriptorGroup(proxy.mSamplers);
        mLiteralSamplerDescriptor = literalSamplerDescriptorGroup.descriptor;
        mLiteralSamplerDescriptorLayout = literalSamplerDescriptorGroup.layout;

        mShaderModule = create_shader(mDevice.getDevice(), spvmoduleStream);
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

    vector<string> kernel_module::getEntryPoints() const
    {
        return get_entry_points(mKernelInterfaces);
    }

    kernel_layout_t kernel_module::createKernelLayout(const string& entryPoint) const {
        if (!isLoaded()) {
            fail_runtime_error("cannot create layout for unloaded module");
        }

        kernel_layout_t result;

        const auto kernelInterface = find_kernel_interface(entryPoint, mKernelInterfaces);
        if (!kernelInterface) {
            fail_runtime_error("cannot create kernel layout for unknown entry point");
        }

        if (-1 != kernelInterface->getArgDescriptorSet()) {
            result.mArgumentDescriptorLayout = createArgumentDescriptorLayout(*kernelInterface, mDevice);

            result.mArgumentsDescriptor = allocate_descriptor_set(mDevice,
                                                                  *result.mArgumentDescriptorLayout);
        }

        result.mLiteralSamplerDescriptor = mLiteralSamplerDescriptor;

        vector<vk::DescriptorSetLayout> layouts;
        if (mLiteralSamplerDescriptorLayout) layouts.push_back(mLiteralSamplerDescriptorLayout);
        if (result.mArgumentDescriptorLayout) layouts.push_back(*result.mArgumentDescriptorLayout);
        result.mPipelineLayout = create_pipeline_layout(mDevice.getDevice(), layouts);

        return result;
    }

    kernel kernel_module::createKernel(const string&        entryPoint,
                                       const vk::Extent3D&  workgroup_sizes)
    {
        return kernel(mDevice,
                      createKernelLayout(entryPoint),
                      *mShaderModule,
                      *mPipelineCache,
                      entryPoint,
                      workgroup_sizes,
                      find_kernel_interface(entryPoint, mKernelInterfaces)->getArguments());
    }

} // namespace clspv_utils

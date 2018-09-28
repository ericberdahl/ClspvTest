//
// Created by Eric Berdahl on 10/22/17.
//

#include "module.hpp"

#include "interface.hpp"
#include "kernel_req.hpp"

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


} // anonymous namespace

namespace clspv_utils {

    module::module()
    {
    }

    module::module(module&& other)
            : module()
    {
        swap(other);
    }

    module::module(std::istream&  spvmoduleStream,
                   device         inDevice,
                   module_spec_t  spec)
            : mDevice(inDevice),
              mModuleSpec(spec),
              mLiteralSamplerDescriptor(),
              mLiteralSamplerDescriptorLayout()
    {
        const auto literalSamplerDescriptorGroup = mDevice.getCachedSamplerDescriptorGroup(mModuleSpec.mSamplers);
        mLiteralSamplerDescriptor = literalSamplerDescriptorGroup.descriptor;
        mLiteralSamplerDescriptorLayout = literalSamplerDescriptorGroup.layout;

        mShaderModule = create_shader(mDevice.getDevice(), spvmoduleStream);
        mPipelineCache = mDevice.getDevice().createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
    }

    module::~module()
    {
    }

    module& module::operator=(module&& other)
    {
        swap(other);
        return *this;
    }

    void module::swap(module& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mModuleSpec, other.mModuleSpec);
        swap(mLiteralSamplerDescriptorLayout, other.mLiteralSamplerDescriptorLayout);
        swap(mLiteralSamplerDescriptor, other.mLiteralSamplerDescriptor);
        swap(mShaderModule, other.mShaderModule);
        swap(mPipelineCache, other.mPipelineCache);
    }

    vector<string> module::getEntryPoints() const
    {
        return getEntryPointNames(mModuleSpec.mKernels);
    }

    kernel_req_t module::createKernelReq(const string &entryPoint) const {
        if (!isLoaded()) {
            fail_runtime_error("cannot create layout for unloaded module");
        }

        const auto kernelSpec = findKernelSpec(entryPoint, mModuleSpec.mKernels);
        if (!kernelSpec) {
            fail_runtime_error("cannot create kernel layout for unknown entry point");
        }

        kernel_req_t result;
        result.mDevice = mDevice;
        result.mKernelSpec = *kernelSpec;
        result.mShaderModule = *mShaderModule;
        result.mPipelineCache = *mPipelineCache;
        result.mLiteralSamplerDescriptor = mLiteralSamplerDescriptor;
        result.mLiteralSamplerLayout = mLiteralSamplerDescriptorLayout;

        return result;
    }

} // namespace clspv_utils

//
// Created by Eric Berdahl on 10/22/17.
//

#include "kernel.hpp"

#include "kernel_invocation.hpp"

namespace clspv_utils {

    kernel::kernel()
            : mArgList(nullptr)
    {
    }

    kernel::kernel(device               inDevice,
                   kernel_layout_t      layout,
                   vk::ShaderModule     shaderModule,
                   vk::PipelineCache    pipelineCache,
                   string               entryPoint,
                   const vk::Extent3D&  workgroup_sizes,
                   arg_list_proxy_t     args) :
            mDevice(inDevice),
            mShaderModule(shaderModule),
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mLayout(std::move(layout)),
            mPipelineCache(pipelineCache),
            mPipeline(),
            mArgList(args)
    {
        updatePipeline(nullptr);
    }

    kernel::~kernel() {
    }

    kernel::kernel(kernel &&other)
            : kernel()
    {
        swap(other);
    }

    kernel& kernel::operator=(kernel&& other)
    {
        swap(other);
        return *this;
    }

    void kernel::swap(kernel& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mShaderModule, other.mShaderModule);
        swap(mEntryPoint, other.mEntryPoint);
        swap(mWorkgroupSizes, other.mWorkgroupSizes);
        swap(mLayout, other.mLayout);
        swap(mPipelineCache, other.mPipelineCache);
        swap(mPipeline, other.mPipeline);
        swap(mArgList, other.mArgList);
    }

    kernel_invocation kernel::createInvocation()
    {
        return kernel_invocation(*this,
                                 mDevice,
                                 *mLayout.mArgumentsDescriptor,
                                 mArgList);
    }

    void kernel::updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants) {
        // TODO: refactor pipelines so invocations that use spec constants don't create them twice, and are still efficient
        vector<std::uint32_t> specConstants = {
                mWorkgroupSizes.width,
                mWorkgroupSizes.height,
                mWorkgroupSizes.depth
        };
        typedef decltype(specConstants)::value_type spec_constant_t;
        std::copy(otherSpecConstants.begin(), otherSpecConstants.end(), std::back_inserter(specConstants));

        vector<vk::SpecializationMapEntry> specializationEntries;
        uint32_t index = 0;
        std::generate_n(std::back_inserter(specializationEntries),
                        specConstants.size(),
                        [&index] () {
                            const uint32_t current = index++;
                            return vk::SpecializationMapEntry(current, current * sizeof(spec_constant_t), sizeof(spec_constant_t));
                        });
        vk::SpecializationInfo specializationInfo;
        specializationInfo.setMapEntryCount(specConstants.size())
                .setPMapEntries(specializationEntries.data())
                .setDataSize(specConstants.size() * sizeof(spec_constant_t))
                .setPData(specConstants.data());

        vk::ComputePipelineCreateInfo createInfo;
        createInfo.setLayout(*mLayout.mPipelineLayout);
        createInfo.stage.setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(mShaderModule)
                .setPName(mEntryPoint.c_str())
                .setPSpecializationInfo(&specializationInfo);

        mPipeline = mDevice.getDevice().createComputePipelineUnique(mPipelineCache, createInfo);
    }

    void kernel::bindCommand(vk::CommandBuffer command) const {
        // TODO: Refactor bindCommand to move into kernel_invocation
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *mPipeline);

        vk::DescriptorSet descriptors[] = { mLayout.mLiteralSamplerDescriptor, *mLayout.mArgumentsDescriptor };
        std::uint32_t numDescriptors = (descriptors[0] ? 2 : 1);
        if (1 == numDescriptors) descriptors[0] = descriptors[1];

        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *mLayout.mPipelineLayout,
                                   0,
                                   { numDescriptors, descriptors },
                                   nullptr);
    }

} // namespace clspv_utils

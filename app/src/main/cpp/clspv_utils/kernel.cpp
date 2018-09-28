//
// Created by Eric Berdahl on 10/22/17.
//

#include "kernel.hpp"

#include "kernel_invocation.hpp"

namespace {

    vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                      device,
                                                    vk::ArrayProxy<const vk::DescriptorSetLayout>   layouts)
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayoutCount(layouts.size())
                .setPSetLayouts(layouts.data());

        return device.createPipelineLayoutUnique(createInfo);
    }

}

namespace clspv_utils {

    kernel::kernel()
    {
    }

    kernel::kernel(kernel_req_t         layout,
                   const vk::Extent3D&  workgroup_sizes) :
            mReq(std::move(layout)),
            mWorkgroupSizes(workgroup_sizes)
    {
        if (-1 != getKernelArgumentDescriptorSet(mReq.mKernelSpec.mArguments)) {
            mArgumentsLayout = createKernelArgumentDescriptorLayout(mReq.mKernelSpec.mArguments, mReq.mDevice.getDevice());

            mArgumentsDescriptor = allocateDescriptorSet(mReq.mDevice, *mArgumentsLayout);
        }

        vector<vk::DescriptorSetLayout> layouts;
        if (mReq.mLiteralSamplerLayout) layouts.push_back(mReq.mLiteralSamplerLayout);
        if (mArgumentsLayout) layouts.push_back(*mArgumentsLayout);
        mPipelineLayout = create_pipeline_layout(mReq.mDevice.getDevice(), layouts);

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

        swap(mReq, other.mReq);
        swap(mArgumentsLayout, other.mArgumentsLayout);
        swap(mArgumentsDescriptor, other.mArgumentsDescriptor);
        swap(mPipelineLayout, other.mPipelineLayout);
        swap(mPipeline, other.mPipeline);
        swap(mWorkgroupSizes, other.mWorkgroupSizes);
    }

    kernel_invocation kernel::createInvocation()
    {
        return kernel_invocation(*this,
                                 mReq.mDevice,
                                 *mArgumentsDescriptor,
                                 mReq.mKernelSpec.mArguments);
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
        createInfo.setLayout(*mPipelineLayout);
        createInfo.stage.setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(mReq.mShaderModule)
                .setPName(mReq.mKernelSpec.mName.c_str())
                .setPSpecializationInfo(&specializationInfo);

        mPipeline = mReq.mDevice.getDevice().createComputePipelineUnique(mReq.mPipelineCache, createInfo);
    }

    void kernel::bindCommand(vk::CommandBuffer command) const {
        // TODO: Refactor bindCommand to move into kernel_invocation
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *mPipeline);

        vk::DescriptorSet descriptors[] = { mReq.mLiteralSamplerDescriptor, *mArgumentsDescriptor };
        std::uint32_t numDescriptors = (descriptors[0] ? 2 : 1);
        if (1 == numDescriptors) descriptors[0] = descriptors[1];

        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *mPipelineLayout,
                                   0,
                                   { numDescriptors, descriptors },
                                   nullptr);
    }

} // namespace clspv_utils

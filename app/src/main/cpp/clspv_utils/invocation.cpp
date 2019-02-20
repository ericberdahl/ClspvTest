//
// Created by Eric Berdahl on 10/22/17.
//

#include "invocation.hpp"

#include "interface.hpp"

#include <cassert>
#include <memory>


namespace clspv_utils {

    execution_time_t::execution_time_t() :
            cpu_duration(0),
            timestamps()
    {
    }

    invocation::invocation()
    {
        // this space intentionally left blank
    }

    invocation::invocation(invocation_req_t req)
            : mReq(std::move(req))
    {
        mCommand = vulkan_utils::allocate_command_buffer(mReq.mDevice.getDevice(), mReq.mDevice.getCommandPool());

        vk::QueryPoolCreateInfo poolCreateInfo;
        poolCreateInfo.setQueryType(vk::QueryType::eTimestamp)
                .setQueryCount(kQueryIndex_Count);

        mQueryPool = mReq.mDevice.getDevice().createQueryPoolUnique(poolCreateInfo);
    }

    invocation::invocation(invocation&& other)
            : invocation()
    {
        swap(other);
    }

    invocation::~invocation() {
    }

    void invocation::swap(invocation& other)
    {
        using std::swap;

        swap(mReq, other.mReq);
        swap(mCommand, other.mCommand);
        swap(mQueryPool, other.mQueryPool);

        swap(mSpecConstantArguments, other.mSpecConstantArguments);
        swap(mBufferMemoryBarriers, other.mBufferMemoryBarriers);
        swap(mImageMemoryBarriers, other.mImageMemoryBarriers);

        swap(mImageArgumentInfo, other.mImageArgumentInfo);
        swap(mBufferArgumentInfo, other.mBufferArgumentInfo);
        swap(mArgumentDescriptorWrites, other.mArgumentDescriptorWrites);
    }

    std::size_t invocation::countArguments() const {
        return mArgumentDescriptorWrites.size() + mSpecConstantArguments.size();
    }

    std::uint32_t invocation::validateArgType(std::size_t        ordinal,
                                                     vk::DescriptorType kind) const
    {
        if (ordinal >= mReq.mKernelSpec.mArguments.size()) {
            fail_runtime_error("adding too many arguments to kernel invocation");
        }

        auto& ka = mReq.mKernelSpec.mArguments.data()[ordinal];
        if (getDescriptorType(ka.mKind) != kind) {
            fail_runtime_error("adding incompatible argument to kernel invocation");
        }

        return ka.mBinding;
    }

    std::uint32_t invocation::validateArgType(std::size_t        ordinal,
                                                     arg_spec_t::kind   kind) const
    {
        if (ordinal >= mReq.mKernelSpec.mArguments.size()) {
            fail_runtime_error("adding too many arguments to kernel invocation");
        }

        auto& ka = mReq.mKernelSpec.mArguments.data()[ordinal];
        if (ka.mKind != kind) {
            fail_runtime_error("adding incompatible argument to kernel invocation");
        }

        return ka.mBinding;
    }

    void invocation::addStorageBufferArgument(vulkan_utils::storage_buffer& buffer) {
        mBufferMemoryBarriers.push_back(buffer.prepareForComputeRead());
        mBufferMemoryBarriers.push_back(buffer.prepareForComputeWrite());
        mBufferArgumentInfo.push_back(buffer.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eStorageBuffer))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer) {
        mBufferMemoryBarriers.push_back(buffer.prepareForComputeRead());
        mBufferArgumentInfo.push_back(buffer.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eUniformBuffer))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addSamplerArgument(vk::Sampler samp) {
        vk::DescriptorImageInfo samplerInfo;
        samplerInfo.setSampler(samp);
        mImageArgumentInfo.push_back(samplerInfo);

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eSampler))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampler);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addReadOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eShaderReadOnlyOptimal));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eSampledImage))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampledImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addWriteOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eGeneral));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eStorageImage))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addLocalArraySizeArgument(unsigned int numElements) {
        validateArgType(countArguments(), arg_spec_t::kind_local);
        mSpecConstantArguments.push_back(numElements);
    }

    void invocation::updateDescriptorSets() {
        //
        // Set up to create the descriptor set write structures for arguments.
        // We will iterate the param lists in the same order,
        // picking up image and buffer infos in order.
        //

        vector<vk::WriteDescriptorSet> writeSets;

        auto nextImage = mImageArgumentInfo.begin();
        auto nextBuffer = mBufferArgumentInfo.begin();

        for (auto& a : mArgumentDescriptorWrites) {
            switch (a.descriptorType) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage:
                case vk::DescriptorType::eSampler:
                    a.setPImageInfo(&(*nextImage));
                    ++nextImage;
                    break;

                case vk::DescriptorType::eUniformBuffer:
                case vk::DescriptorType::eStorageBuffer:
                    a.setPBufferInfo(&(*nextBuffer));
                    ++nextBuffer;
                    break;

                default:
                    assert(0 && "unkown argument type");
            }
        }

        writeSets.insert(writeSets.end(), mArgumentDescriptorWrites.begin(), mArgumentDescriptorWrites.end());

        //
        // Do the actual descriptor set updates
        //
        mReq.mDevice.getDevice().updateDescriptorSets(writeSets, nullptr);
    }

    void invocation::fillCommandBuffer(const vk::Extent3D& num_workgroups)
    {
        auto pipeline = mReq.mGetPipelineFn(mSpecConstantArguments);

        mCommand->begin(vk::CommandBufferBeginInfo());

        mCommand->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

        vk::DescriptorSet descriptors[] = { mReq.mLiteralSamplerDescriptor, mReq.mArgumentsDescriptor };
        std::uint32_t numDescriptors = (descriptors[0] ? 2 : 1);
        if (1 == numDescriptors) descriptors[0] = descriptors[1];

        mCommand->bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     mReq.mPipelineLayout,
                                     0,
                                     { numDescriptors, descriptors },
                                     nullptr);

        mCommand->resetQueryPool(*mQueryPool, kQueryIndex_FirstIndex, kQueryIndex_Count);

        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_StartOfExecution);
        mCommand->pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  vk::DependencyFlags(),
                                  nullptr,    // memory barriers
                                  mBufferMemoryBarriers,    // buffer memory barriers
                                  mImageMemoryBarriers);    // image memory barriers
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostHostBarrier);
        mCommand->dispatch(num_workgroups.width, num_workgroups.height, num_workgroups.depth);
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostExecution);
        mCommand->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer,
                                  vk::DependencyFlags(),
                                  { { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead} },    // memory barriers
                                  nullptr,    // buffer memory barriers
                                  nullptr);    // image memory barriers
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostGPUBarrier);

        mCommand->end();
    }

    void invocation::submitCommand() {
        vk::CommandBuffer rawCommand = *mCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        mReq.mDevice.getComputeQueue().submit(submitInfo, nullptr);

    }

    execution_time_t invocation::run(const vk::Extent3D& num_workgroups) {
        updateDescriptorSets();
        fillCommandBuffer(num_workgroups);

        auto start = std::chrono::high_resolution_clock::now();
        submitCommand();
        mReq.mDevice.getComputeQueue().waitIdle();
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t timestamps[kQueryIndex_Count];
        mReq.mDevice.getDevice().getQueryPoolResults(*mQueryPool,
                                                     kQueryIndex_FirstIndex,
                                                     kQueryIndex_Count,
                                                     sizeof(uint64_t),
                                                     timestamps,
                                                     sizeof(uint64_t),
                                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

        execution_time_t result;
        result.cpu_duration = end - start;
        result.timestamps.start = timestamps[kQueryIndex_StartOfExecution];
        result.timestamps.host_barrier = timestamps[kQueryIndex_PostHostBarrier];
        result.timestamps.execution = timestamps[kQueryIndex_PostExecution];
        result.timestamps.gpu_barrier = timestamps[kQueryIndex_PostGPUBarrier];
        return result;
    }

} // namespace clspv_utils

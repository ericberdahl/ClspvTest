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
        vk::QueryPoolCreateInfo poolCreateInfo;
        poolCreateInfo.setQueryType(vk::QueryType::eTimestamp)
                .setQueryCount(kTimestamp_count);

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

    void invocation::addStorageBufferArgument(vulkan_utils::buffer& buffer) {
        if (!(buffer.getUsage() & vk::BufferUsageFlagBits::eStorageBuffer)) {
            fail_runtime_error("buffer is not configured as a storage buffer");
        }

        mBufferMemoryBarriers.push_back(buffer.prepareForShaderRead());
        mBufferMemoryBarriers.push_back(buffer.prepareForShaderWrite());
        mBufferArgumentInfo.push_back(buffer.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mReq.mArgumentsDescriptor)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eStorageBuffer))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void invocation::addUniformBufferArgument(vulkan_utils::buffer& buffer) {
        if (!(buffer.getUsage() & vk::BufferUsageFlagBits::eUniformBuffer)) {
            fail_runtime_error("buffer is not configured as a uniform buffer");
        }

        mBufferMemoryBarriers.push_back(buffer.prepareForShaderRead());
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

        //
        // Do the actual descriptor set updates
        //
        mReq.mDevice.getDevice().updateDescriptorSets(mArgumentDescriptorWrites, nullptr);
    }

    void invocation::fillCommandBuffer(vk::CommandBuffer commandBuffer, const vk::Extent3D& num_workgroups)
    {
        auto pipeline = mReq.mGetPipelineFn(mSpecConstantArguments);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

        vk::DescriptorSet descriptors[] = { mReq.mLiteralSamplerDescriptor, mReq.mArgumentsDescriptor };
        std::uint32_t numDescriptors = (descriptors[0] ? 2 : 1);
        if (1 == numDescriptors) descriptors[0] = descriptors[1];

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                         mReq.mPipelineLayout,
                                         0,
                                         { numDescriptors, descriptors },
                                         nullptr);

        commandBuffer.resetQueryPool(*mQueryPool, kTimestamp_first, kTimestamp_count);

        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                     *mQueryPool,
                                     kTimestamp_startOfExecution);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      vk::DependencyFlags(),
                                      nullptr,    // memory barriers
                                      mBufferMemoryBarriers,    // buffer memory barriers
                                      mImageMemoryBarriers);    // image memory barriers

        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                     *mQueryPool,
                                     kTimestamp_postHostBarrier);

        commandBuffer.dispatch(num_workgroups.width, num_workgroups.height, num_workgroups.depth);

        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                     *mQueryPool,
                                     kTimestamp_postExecution);
    }

    void invocation::submitCommand(vk::CommandBuffer commandBuffer) {
        vk::CommandBuffer rawCommand = commandBuffer;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        mReq.mDevice.getComputeQueue().submit(submitInfo, nullptr);

    }

    execution_time_t invocation::run(const vk::Extent3D& num_workgroups) {
        const auto commandBuffer = vulkan_utils::allocate_command_buffer(mReq.mDevice.getDevice(), mReq.mDevice.getCommandPool());

        commandBuffer->begin(vk::CommandBufferBeginInfo());
        dispatch(*commandBuffer, num_workgroups);
        commandBuffer->end();

        auto start = std::chrono::high_resolution_clock::now();
        submitCommand(*commandBuffer);
        mReq.mDevice.getComputeQueue().waitIdle();
        auto end = std::chrono::high_resolution_clock::now();

        execution_time_t result = getExecutionTime();
        result.cpu_duration = end - start;
        return result;
    }

    void invocation::dispatch(vk::CommandBuffer commandBuffer, const vk::Extent3D& numWorkgroups)
    {
        updateDescriptorSets();
        fillCommandBuffer(commandBuffer, numWorkgroups);
    }

    execution_time_t invocation::getExecutionTime()
    {
        uint64_t timestamps[kTimestamp_count];
        mReq.mDevice.getDevice().getQueryPoolResults(*mQueryPool,
                                                     kTimestamp_first,
                                                     kTimestamp_count,
                                                     sizeof(uint64_t),
                                                     timestamps,
                                                     sizeof(uint64_t),
                                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

        execution_time_t result;
        result.timestamps.start = timestamps[kTimestamp_startOfExecution];
        result.timestamps.host_barrier = timestamps[kTimestamp_postHostBarrier];
        result.timestamps.execution = timestamps[kTimestamp_postExecution];
        return result;
    }

} // namespace clspv_utils

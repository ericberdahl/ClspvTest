//
// Created by Eric Berdahl on 10/22/17.
//

#include "device.hpp"

#include "sampler_spec.hpp"

#include <cassert>


namespace {
    using namespace clspv_utils;

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


    //
    // boost_* code heavily borrowed from Boost 1.65.0
    // Ideally, we'd just use boost directly (that's what it's for, after all). However, it's a lot
    // to pick up, just for these functions and for what is intened to be a simple library.
    //

    template <typename SizeT>
    void boost_hash_combine_impl(SizeT& seed, SizeT value)
    {
        seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }

    std::uint32_t boost_functional_hash_rotl32(std::uint32_t x, unsigned int r) {
        return (x << r) | (x >> (32 - r));
    }

    void boost_hash_combine_impl(std::uint32_t& h1, std::uint32_t k1)
    {
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;

        k1 *= c1;
        k1 = boost_functional_hash_rotl32(k1,15);
        k1 *= c2;

        h1 ^= k1;
        h1 = boost_functional_hash_rotl32(h1,13);
        h1 = h1*5+0xe6546b64;
    }

    void boost_hash_combine_impl(std::uint64_t& h, std::uint64_t k)
    {
        const std::uint64_t m = 0xc6a4a7935bd1e995ULL;
        const int r = 47;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;

        // Completely arbitrary number, to prevent 0's
        // from hashing to 0.
        h += 0xe6546b64;
    }

    std::size_t compute_hash(const vk::ArrayProxy<const sampler_spec_t>& samplers)
    {
        std::size_t result = 0;

        for (auto& s : samplers)
        {
            boost_hash_combine_impl(result, std::hash<int>{}(s.opencl_flags));
        }

        return result;
    }

} // anonymous namespace

namespace clspv_utils {

    device::device(vk::PhysicalDevice                   physicalDevice,
                   vk::Device                           device,
                   vk::DescriptorPool                   descriptorPool,
                   vk::CommandPool                      commandPool,
                   vk::Queue                            computeQueue)
            : mPhysicalDevice(physicalDevice),
              mDevice(device),
              mMemoryProperties(physicalDevice.getMemoryProperties()),
              mDescriptorPool(descriptorPool),
              mCommandPool(commandPool),
              mComputeQueue(computeQueue),
              mSamplerCache(new sampler_cache_t),
              mSamplerDescriptorCache(new descriptor_cache_t)
    {
    }

    vk::Sampler device::getCachedSampler(int opencl_flags)
    {
        assert(mSamplerCache);

        if (!mSamplerCache->count(opencl_flags)) {
            (*mSamplerCache)[opencl_flags] = createCompatibleSampler(mDevice, opencl_flags);
        }
        return *(*mSamplerCache)[opencl_flags];
    }

    vk::UniqueDescriptorSetLayout device::createSamplerDescriptorLayout(const sampler_list_proxy_t& samplers) const
    {
        vk::UniqueDescriptorSetLayout samplerDescriptorLayout;

        if (!samplers.empty()) {

            vector<vk::DescriptorSetLayoutBinding> bindingSet;

            vk::DescriptorSetLayoutBinding binding;
            binding.setStageFlags(vk::ShaderStageFlagBits::eCompute)
                    .setDescriptorCount(1);

            for (auto& s : samplers) {
                binding.descriptorType = vk::DescriptorType::eSampler;
                binding.binding = s.binding;
                bindingSet.push_back(binding);
            }

            vk::DescriptorSetLayoutCreateInfo createInfo;
            createInfo.setBindingCount(bindingSet.size())
                    .setPBindings(bindingSet.size() ? bindingSet.data() : nullptr);

            samplerDescriptorLayout = mDevice.createDescriptorSetLayoutUnique(createInfo);
        }

        return samplerDescriptorLayout;
    }

    vk::UniqueDescriptorSet device::createSamplerDescriptor(const sampler_list_proxy_t& samplers,
                                                            vk::DescriptorSetLayout     layout)
    {
        vk::UniqueDescriptorSet samplerDescriptor;

        if (layout) {
            samplerDescriptor = allocate_descriptor_set(*this, layout);

            vector<vk::DescriptorImageInfo> literalSamplerInfo;
            vector<vk::WriteDescriptorSet> literalSamplerDescriptorWrites;

            literalSamplerInfo.reserve(samplers.size());
            literalSamplerDescriptorWrites.reserve(samplers.size());

            for (auto s : samplers) {
                vk::DescriptorImageInfo samplerInfo;
                samplerInfo.setSampler(getCachedSampler(s.opencl_flags));
                literalSamplerInfo.push_back(samplerInfo);

                vk::WriteDescriptorSet literalSamplerSet;
                literalSamplerSet.setDstSet(*samplerDescriptor)
                        .setDstBinding(s.binding)
                        .setDescriptorCount(1)
                        .setDescriptorType(vk::DescriptorType::eSampler)
                        .setPImageInfo(&literalSamplerInfo.back());
                literalSamplerDescriptorWrites.push_back(literalSamplerSet);
            }

            mDevice.updateDescriptorSets(literalSamplerDescriptorWrites, nullptr);
        }

        return samplerDescriptor;
    }

    device::descriptor_group_t device::getCachedSamplerDescriptorGroup(const sampler_list_proxy_t& samplers)
    {
        assert(mSamplerDescriptorCache);

        const std::size_t hash = compute_hash(samplers);

        if (0 == mSamplerDescriptorCache->count(hash))
        {
            unique_descriptor_group_t unique_group;
            unique_group.layout = createSamplerDescriptorLayout(samplers);
            unique_group.descriptor = createSamplerDescriptor(samplers, *unique_group.layout);

            (*mSamplerDescriptorCache)[hash] = std::move(unique_group);
        }

        const auto found = mSamplerDescriptorCache->find(hash);
        assert(found != mSamplerDescriptorCache->end());

        descriptor_group_t result;
        result.layout = *found->second.layout;
        result.descriptor = *found->second.descriptor;
        return result;
    }

} // namespace clspv_utils

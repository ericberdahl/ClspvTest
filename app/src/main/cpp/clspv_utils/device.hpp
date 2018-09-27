//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_DEVICE_HPP
#define CLSPVUTILS_DEVICE_HPP

#include "clspv_utils_fwd.hpp"

#include "clspv_utils_interop.hpp"
#include "interface.hpp"

#include <vulkan/vulkan.hpp>

#include <memory>

namespace clspv_utils {

    class device {
    public:
        struct descriptor_group_t
        {
            vk::DescriptorSet       descriptor;
            vk::DescriptorSetLayout layout;
        };

        typedef vk::ArrayProxy<const sampler_spec_t> sampler_list_proxy_t;

        device() {}

        device(vk::PhysicalDevice   physicalDevice,
               vk::Device           device,
               vk::DescriptorPool   descriptorPool,
               vk::CommandPool      commandPool,
               vk::Queue            computeQueue);

        vk::PhysicalDevice  getPhysicalDevice() const { return mPhysicalDevice; }
        vk::Device          getDevice() const { return mDevice; }
        vk::DescriptorPool  getDescriptorPool() const { return mDescriptorPool; }
        vk::CommandPool     getCommandPool() const { return mCommandPool; }
        vk::Queue           getComputeQueue() const { return mComputeQueue; }

        const vk::PhysicalDeviceMemoryProperties&   getMemoryProperties() const { return mMemoryProperties; }

        vk::Sampler                     getCachedSampler(int opencl_flags);

        vk::UniqueDescriptorSetLayout   createSamplerDescriptorLayout(const sampler_list_proxy_t& samplers) const;

        vk::UniqueDescriptorSet         createSamplerDescriptor(const sampler_list_proxy_t& samplers,
                                                                vk::DescriptorSetLayout layout);

        descriptor_group_t              getCachedSamplerDescriptorGroup(const sampler_list_proxy_t& samplers);

    private:
        struct unique_descriptor_group_t
        {
            vk::UniqueDescriptorSet       descriptor;
            vk::UniqueDescriptorSetLayout layout;
        };

        typedef map<std::size_t,unique_descriptor_group_t> descriptor_cache_t;
        typedef map<int,vk::UniqueSampler> sampler_cache_t;

    private:
        vk::PhysicalDevice                  mPhysicalDevice;
        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::DescriptorPool                  mDescriptorPool;
        vk::CommandPool                     mCommandPool;
        vk::Queue                           mComputeQueue;

        shared_ptr<descriptor_cache_t>      mSamplerDescriptorCache;
        shared_ptr<sampler_cache_t>         mSamplerCache;
    };
}

#endif //CLSPVUTILS_DEVICE_HPP

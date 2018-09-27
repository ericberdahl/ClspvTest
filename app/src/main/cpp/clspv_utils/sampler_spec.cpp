//
// Created by Eric Berdahl on 10/22/17.
//

#include "interface.hpp"

#include "clspv_utils_interop.hpp"
#include "opencl_types.hpp"

#include <algorithm>
#include <functional>
#include <memory>

namespace {
    using namespace clspv_utils;

    const auto kCLAddressMode_VkAddressMode_Map = {
            std::make_pair(CLK_ADDRESS_NONE,            vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP_TO_EDGE,   vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP,           vk::SamplerAddressMode::eClampToBorder),
            std::make_pair(CLK_ADDRESS_REPEAT,          vk::SamplerAddressMode::eRepeat),
            std::make_pair(CLK_ADDRESS_MIRRORED_REPEAT, vk::SamplerAddressMode::eMirroredRepeat)
    };

} // anonymous namespace

namespace clspv_utils {

    vk::SamplerAddressMode getSamplerAddressMode(int opencl_flags) {
        opencl_flags &= CLK_ADDRESS_MASK;

        auto found = std::find_if(std::begin(kCLAddressMode_VkAddressMode_Map),
                                  std::end(kCLAddressMode_VkAddressMode_Map),
                                  [&opencl_flags](decltype(kCLAddressMode_VkAddressMode_Map)::const_reference am) {
                                      return (am.first == opencl_flags);
                                  });

        return (found == std::end(kCLAddressMode_VkAddressMode_Map) ? vk::SamplerAddressMode::eClampToEdge : found->second);
    }


    vk::Filter getSamplerFilter(int opencl_flags) {
        return ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ? vk::Filter::eLinear
                                                                      : vk::Filter::eNearest);
    }

    vk::Bool32 isSamplerUnnormalizedCoordinates(int opencl_flags) {
        return ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_TRUE : VK_FALSE);
    }

    void validateSampler(const sampler_spec_t& spec, int requiredDescriptorSet) {
        if (spec.mOpenclFlags == 0) {
            fail_runtime_error("sampler missing OpenCL flags");
        }
        if (spec.mDescriptorSet < 0) {
            fail_runtime_error("sampler missing descriptorSet");
        }
        if (spec.mBinding < 0) {
            fail_runtime_error("sampler missing binding");
        }

        // all samplers, are documented to share descriptor set 0
        if (spec.mDescriptorSet != 0) {
            fail_runtime_error("all clspv literal samplers must use descriptor set 0");
        }

        if (!isSamplerSupported(spec.mOpenclFlags)) {
            fail_runtime_error("sampler is not representable in Vulkan");
        }

        // All literal samplers for a module need to be in the same descriptor set
        if (requiredDescriptorSet > 0 && spec.mDescriptorSet != requiredDescriptorSet) {
            fail_runtime_error("sampler is not in required descriptor_set");
        }
    }

    bool isSamplerSupported(int opencl_flags)
    {
        const vk::Bool32 unnormalizedCoordinates    = isSamplerUnnormalizedCoordinates(opencl_flags);
        const vk::SamplerAddressMode addressMode    = getSamplerAddressMode(opencl_flags);

        return (!unnormalizedCoordinates || addressMode == vk::SamplerAddressMode::eClampToEdge || addressMode == vk::SamplerAddressMode::eClampToBorder);
    }

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags) {
        if (!isSamplerSupported(opencl_flags)) {
            fail_runtime_error("This OpenCL sampler cannot be represented in Vulkan");
        }

        const vk::Filter filter                     = getSamplerFilter(opencl_flags);
        const vk::Bool32 unnormalizedCoordinates    = isSamplerUnnormalizedCoordinates(opencl_flags);
        const vk::SamplerAddressMode addressMode    = getSamplerAddressMode(opencl_flags);

        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.setMagFilter(filter)
                .setMinFilter(filter)
                .setMipmapMode(vk::SamplerMipmapMode::eNearest)
                .setAddressModeU(addressMode)
                .setAddressModeV(addressMode)
                .setAddressModeW(addressMode)
                .setAnisotropyEnable(VK_FALSE)
                .setCompareEnable(VK_FALSE)
                .setUnnormalizedCoordinates(unnormalizedCoordinates);

        return device.createSamplerUnique(samplerCreateInfo);
    }

} // namespace clspv_utils

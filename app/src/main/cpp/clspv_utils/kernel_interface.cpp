//
// Created by Eric Berdahl on 10/22/17.
//

#include "kernel_interface.hpp"

#include "clspv_utils_interop.hpp"
#include "device.hpp"
#include "kernel.hpp"
#include "kernel_invocation.hpp"
#include "kernel_module.hpp"
#include "sampler_spec.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>

namespace {
    using namespace clspv_utils;

    const auto kArgKind_DescriptorType_Map = {
            std::make_pair(arg_spec_t::kind_pod_ubo, vk::DescriptorType::eUniformBuffer),
            std::make_pair(arg_spec_t::kind_pod, vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_buffer, vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_ro_image, vk::DescriptorType::eSampledImage),
            std::make_pair(arg_spec_t::kind_wo_image, vk::DescriptorType::eStorageImage),
            std::make_pair(arg_spec_t::kind_sampler, vk::DescriptorType::eSampler)
    };

    /* TODO opportunity for sharing */
    vk::DescriptorType find_descriptor_type(arg_spec_t::kind_t argKind) {
        auto found = std::find_if(std::begin(kArgKind_DescriptorType_Map),
                                  std::end(kArgKind_DescriptorType_Map),
                                  [argKind](decltype(kArgKind_DescriptorType_Map)::const_reference p) {
                                      return argKind == p.first;
                                  });
        if (found == std::end(kArgKind_DescriptorType_Map)) {
            fail_runtime_error("unknown argKind encountered");
        }
        return found->second;
    }

    void validate_kernel_arg(const arg_spec_t& arg) {
        if (arg.kind == arg_spec_t::kind_unknown) {
            fail_runtime_error("kernel argument kind unknown");
        }
        if (arg.ordinal < 0) {
            fail_runtime_error("kernel argument missing ordinal");
        }

        if (arg.kind == arg_spec_t::kind_local) {
            if (arg.spec_constant < 0) {
                fail_runtime_error("local kernel argument missing spec constant");
            }
        }
        else {
            if (arg.descriptor_set < 0) {
                fail_runtime_error("kernel argument missing descriptorSet");
            }
            if (arg.binding < 0) {
                fail_runtime_error("kernel argument missing binding");
            }
            if (arg.offset < 0) {
                fail_runtime_error("kernel argument missing offset");
            }
        }
    }

} // anonymous namespace

namespace clspv_utils {

    kernel_interface::kernel_interface()
            : mLiteralSamplers(nullptr)
    {
    }

    kernel_interface::kernel_interface(string               entryPoint,
                                       sampler_list_proxy_t samplers,
                                       arg_list_t           arguments)
            : kernel_interface()
    {
        mName = entryPoint;
        mArgSpecs = std::move(arguments);
        mLiteralSamplers = samplers;

        // Sort the args such that pods are grouped together at the end of the sequence, and that
        // the non-pod and pod groups are each individually sorted by increasing ordinal
        std::sort(mArgSpecs.begin(), mArgSpecs.end(), [](const arg_spec_t& lhs, const arg_spec_t& rhs) {
            auto isPod = [](arg_spec_t::kind_t kind) {
                return (kind == arg_spec_t::kind_pod || kind == arg_spec_t::kind_pod_ubo);
            };

            const auto lhs_is_pod = isPod(lhs.kind);
            const auto rhs_is_pod = isPod(rhs.kind);

            return (lhs_is_pod == rhs_is_pod ? lhs.ordinal < rhs.ordinal : !lhs_is_pod);
        });

        validate();
    }

    void kernel_interface::validate() const
    {
        if (mName.empty()) {
            fail_runtime_error("kernel has no name");
        }

        const int arg_ds = getArgDescriptorSet();
        for (auto& ka : mArgSpecs) {
            // All arguments for a given kernel that are passed in a descriptor set need to be in
            // the same descriptor set
            if (ka.kind != arg_spec_t::kind_local && ka.descriptor_set != arg_ds) {
                fail_runtime_error("kernel arg descriptor_sets don't match");
            }

            validate_kernel_arg(ka);
        }

        const int sampler_ds = getLiteralSamplersDescriptorSet();
        for (auto& ls : mLiteralSamplers) {
            // All literal samplers for a given kernel need to be in the same descriptor set
            if (ls.descriptor_set != sampler_ds) {
                fail_runtime_error("literal sampler descriptor_sets don't match");
            }

            ls.validate();
        }

        // TODO: mArgSpec entries are in increasing binding, and pod/pod_ubo's come after non-pod/non-pod_ubo's
        // TODO: there cannot be both pod and pod_ubo arguments for a given kernel
        // TODO: if there is a pod or pod_ubo argument, its descriptor set must be larger than other descriptor sets
    }

    int kernel_interface::getArgDescriptorSet() const {
        auto found = std::find_if(mArgSpecs.begin(), mArgSpecs.end(), [](const arg_spec_t& as) {
            return (-1 != as.descriptor_set);
        });
        return (found == mArgSpecs.end() ? -1 : found->descriptor_set);
    }

    int kernel_interface::getLiteralSamplersDescriptorSet() const {
        auto found = std::find_if(mLiteralSamplers.begin(), mLiteralSamplers.end(), [](const sampler_spec_t& ss) {
            return (-1 != ss.descriptor_set);
        });
        return (found == mLiteralSamplers.end() ? -1 : found->descriptor_set);
    }

    vk::UniqueDescriptorSetLayout kernel_interface::createArgDescriptorLayout(const device& inDevice) const
    {
        assert(getArgDescriptorSet() == (getLiteralSamplers().empty() ? 0 : 1));

        vector<vk::DescriptorSetLayoutBinding> bindingSet;

        vk::DescriptorSetLayoutBinding binding;
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute)
                .setDescriptorCount(1);

        for (auto &ka : mArgSpecs) {
            // ignore any argument not in offset 0
            if (0 != ka.offset) continue;

            binding.descriptorType = find_descriptor_type(ka.kind);
            binding.binding = ka.binding;

            bindingSet.push_back(binding);
        }

        vk::DescriptorSetLayoutCreateInfo createInfo;
        createInfo.setBindingCount(bindingSet.size())
                .setPBindings(bindingSet.size() ? bindingSet.data() : nullptr);

        return inDevice.getDevice().createDescriptorSetLayoutUnique(createInfo);
    }

} // namespace clspv_utils

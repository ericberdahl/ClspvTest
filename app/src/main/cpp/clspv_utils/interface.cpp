//
// Created by Eric Berdahl on 10/22/17.
//

#include "interface.hpp"

#include "clspv_utils_interop.hpp"
#include "opencl_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <istream>
#include <limits>
#include <memory>

#include <sstream> // std::istringstream

namespace {
    using namespace clspv_utils;

    typedef std::pair<string, string> key_value_t;

    const auto kCLAddressMode_VkAddressMode_Map = {
            std::make_pair(CLK_ADDRESS_NONE,            vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP_TO_EDGE,   vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP,           vk::SamplerAddressMode::eClampToBorder),
            std::make_pair(CLK_ADDRESS_REPEAT,          vk::SamplerAddressMode::eRepeat),
            std::make_pair(CLK_ADDRESS_MIRRORED_REPEAT, vk::SamplerAddressMode::eMirroredRepeat)
    };

    const auto kArgKind_DescriptorType_Map = {
            std::make_pair(arg_spec_t::kind_pod_ubo,    vk::DescriptorType::eUniformBuffer),
            std::make_pair(arg_spec_t::kind_pod,        vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_buffer,     vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_buffer_ubo, vk::DescriptorType::eUniformBuffer),
            std::make_pair(arg_spec_t::kind_combined_image_sampler,   vk::DescriptorType::eCombinedImageSampler),
            std::make_pair(arg_spec_t::kind_ro_image,   vk::DescriptorType::eSampledImage),
            std::make_pair(arg_spec_t::kind_wo_image,   vk::DescriptorType::eStorageImage),
            std::make_pair(arg_spec_t::kind_sampler,    vk::DescriptorType::eSampler)
    };

    const auto kSpvMapArgType_ArgKind_Map = {
            std::make_pair("pod",        arg_spec_t::kind_pod),
            std::make_pair("pod_ubo",    arg_spec_t::kind_pod_ubo),
            std::make_pair("buffer",     arg_spec_t::kind_buffer),
            std::make_pair("buffer_ubo", arg_spec_t::kind_buffer_ubo),
            std::make_pair("combined_image_sampler",   arg_spec_t::kind_combined_image_sampler),
            std::make_pair("ro_image",   arg_spec_t::kind_ro_image),
            std::make_pair("wo_image",   arg_spec_t::kind_wo_image),
            std::make_pair("sampler",    arg_spec_t::kind_sampler),
            std::make_pair("local",      arg_spec_t::kind_local)
    };

    arg_spec_t::kind find_arg_kind(const string &argType) {
        auto found = std::find_if(std::begin(kSpvMapArgType_ArgKind_Map),
                                  std::end(kSpvMapArgType_ArgKind_Map),
                                  [&argType](decltype(kSpvMapArgType_ArgKind_Map)::const_reference p) {
                                      return argType == p.first;
                                  });
        if (found == std::end(kSpvMapArgType_ArgKind_Map)) {
            fail_runtime_error("unknown argType encountered");
        }
        return found->second;
    }

    vector<std::uint8_t> hexToBytes(string hexString) {
        vector<std::uint8_t> result;

        if (0 != (hexString.length() % 2)) {
            fail_runtime_error("spvmap constant hex string must have even number of characters");
        }

        result.reserve(hexString.length() / 2);

        for (string::size_type i = 0; i < hexString.length(); i +=2) {
            const string    byteString = hexString.substr(i, 2);
            const int       byteInt = std::stoi(byteString, nullptr, 16);

            assert(0 <= byteInt);
            assert(byteInt <= std::numeric_limits<std::uint8_t>::max());

            result.push_back(static_cast<std::uint8_t>(byteInt));
        }

        return result;
    }

    string read_csv_field(std::istream& in) {
        string result;

        if (in.good()) {
            const bool is_quoted = (in.peek() == '"');

            if (is_quoted) {
                in.ignore(std::numeric_limits<std::streamsize>::max(), '"');
            }

            std::getline(in, result, is_quoted ? '"' : ',');

            if (is_quoted) {
                in.ignore(std::numeric_limits<std::streamsize>::max(), ',');
            }
        }

        return result;
    }

    key_value_t read_key_value_pair(std::istream& in) {
        return std::make_pair(read_csv_field(in), read_csv_field(in));
    };

    constant_spec_t parse_spvmap_constant(std::istream& in) {
        constant_spec_t result;

        while (!in.eof()) {
            const auto tag = read_key_value_pair(in);

            if ("descriptorSet" == tag.first) {
                result.mDescriptorSet = std::stoi(tag.second);
            } else if ("binding" == tag.first) {
                result.mBinding = std::stoi(tag.second);
            } else if ("hexbytes" == tag.first) {
                result.mBytes = hexToBytes(tag.second);
            }
        }

        return result;
    }

    sampler_spec_t parse_spvmap_sampler(std::istream& in) {
        sampler_spec_t result;

        const auto clFlagString = read_csv_field(in);
        result.mOpenclFlags = std::stoi(clFlagString);

        while (!in.eof()) {
            const auto tag = read_key_value_pair(in);

            if ("descriptorSet" == tag.first) {
                result.mDescriptorSet = std::stoi(tag.second);
            } else if ("binding" == tag.first) {
                result.mBinding = std::stoi(tag.second);
            }
        }

        return result;
    }

    arg_spec_t parse_spvmap_kernel_arg(std::istream& in) {
        arg_spec_t result;

        while (!in.eof()) {
            const auto tag = read_key_value_pair(in);

            if ("argOrdinal" == tag.first) {
                result.mOrdinal = std::stoi(tag.second);
            } else if ("descriptorSet" == tag.first) {
                result.mDescriptorSet = std::stoi(tag.second);
            } else if ("binding" == tag.first) {
                result.mBinding = std::stoi(tag.second);
            } else if ("offset" == tag.first) {
                result.mOffset = std::stoi(tag.second);
            } else if ("argKind" == tag.first) {
                result.mKind = find_arg_kind(tag.second);
            } else if ("arrayElemSize" == tag.first) {
                // arrayElemSize is ignored by clspvtest
            } else if ("arrayNumElemSpecId" == tag.first) {
                result.mSpecConstant = std::stoi(tag.second);
            } else if ("argSize" == tag.first) {
                result.mArgSize = std::stoi(tag.second);
            }

        }

        return result;
    }

} // anonymous namespace

namespace clspv_utils {

    /***********************************************************************************************
     * module_spect_t functions
     **********************************************************************************************/

    module_spec_t createModuleSpec(std::istream& in)
    {
        module_spec_t result;

        kernel_spec_t* recentKernel = nullptr;

        string line;
        while (!in.eof()) {
            std::getline(in, line);

            std::istringstream in_line(line);
            const auto tag = read_csv_field(in_line);
            if ("sampler" == tag) {
                result.mSamplers.push_back(parse_spvmap_sampler(in_line));
            } else if ("constant" == tag) {
                result.mConstants.push_back(parse_spvmap_constant(in_line));
            } else if ("kernel" == tag) {
                const auto kernelName = read_csv_field(in_line);
                if (!recentKernel || recentKernel->mName != kernelName)
                {
                    recentKernel = findKernelSpec(kernelName, result.mKernels);
                    if (!recentKernel) {
                        result.mKernels.push_back(kernel_spec_t{ kernelName, kernel_spec_t::arg_list() });
                        recentKernel = &result.mKernels.back();
                    }
                }
                assert(recentKernel);

                recentKernel->mArguments.push_back(parse_spvmap_kernel_arg(in_line));
            }
        }

        // Ensure that the literal samplers are sorted by increasing binding number. This will be
        // important if the sequence is later used to determine whether a cached sampler descriptor
        // set can be re-used for this module.
        std::sort(result.mSamplers.begin(), result.mSamplers.end(), [](const sampler_spec_t& lhs, const sampler_spec_t& rhs) {
            return lhs.mBinding < rhs.mBinding;
        });

        for (auto& k : result.mKernels) {
            standardizeKernelArgumentOrder(k.mArguments);
        }

        validateModule(result);

        return result;
    }

    /***********************************************************************************************
     * module_spec_t::kernel_list functions
     **********************************************************************************************/

    const kernel_spec_t* findKernelSpec(const string&                       name,
                                        const module_spec_t::kernel_list&   kernels)
    {
        auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                   [&name](const kernel_spec_t &iter) {
                                       return iter.mName == name;
                                   });

        return (kernel == kernels.end() ? nullptr : &(*kernel));
    }

    kernel_spec_t* findKernelSpec(const string&                 name,
                                  module_spec_t::kernel_list&   kernels)
    {
        return const_cast<kernel_spec_t*>(findKernelSpec(name, const_cast<const module_spec_t::kernel_list&>(kernels)));
    }

    vector<string> getEntryPointNames(const module_spec_t::kernel_list& specs)
    {
        vector<string> result;

        std::transform(specs.begin(), specs.end(),
                       std::back_inserter(result),
                       [](const kernel_spec_t& k) { return k.mName; });

        return result;
    }

    /***********************************************************************************************
     * module_spec_t::sampler_list functions
     **********************************************************************************************/

    int getSamplersDescriptorSet(const module_spec_t::sampler_list& samplers) {
        auto found = std::find_if(samplers.begin(), samplers.end(),
                                  [](const sampler_spec_t &ss) {
                                      return (-1 != ss.mDescriptorSet);
                                  });
        return (found == samplers.end() ? -1 : found->mDescriptorSet);
    }

    /***********************************************************************************************
     * kernel_spec_t::arg_list functions
     **********************************************************************************************/

    void standardizeKernelArgumentOrder(kernel_spec_t::arg_list& arguments)
    {
        std::sort(arguments.begin(), arguments.end(), [](const arg_spec_t& lhs, const arg_spec_t& rhs) {
            auto isPod = [](arg_spec_t::kind kind) {
                return (kind == arg_spec_t::kind_pod || kind == arg_spec_t::kind_pod_ubo);
            };

            const auto lhs_is_pod = isPod(lhs.mKind);
            const auto rhs_is_pod = isPod(rhs.mKind);

            return (lhs_is_pod == rhs_is_pod ? lhs.mOrdinal < rhs.mOrdinal : !lhs_is_pod);
        });
    }

    int getKernelArgumentDescriptorSet(const kernel_spec_t::arg_list& arguments) {
        auto found = std::find_if(arguments.begin(), arguments.end(), [](const arg_spec_t& as) {
            return (-1 != as.mDescriptorSet);
        });
        return (found == arguments.end() ? -1 : found->mDescriptorSet);
    }

    vk::UniqueDescriptorSetLayout createKernelArgumentDescriptorLayout(const kernel_spec_t::arg_list& arguments,
                                                                       vk::Device inDevice)
    {
        vector<vk::DescriptorSetLayoutBinding> bindingSet;

        vk::DescriptorSetLayoutBinding binding;
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute)
                .setDescriptorCount(1);

        for (auto &ka : arguments) {
            // ignore any argument not in offset 0
            if (0 != ka.mOffset) continue;

            binding.descriptorType = getDescriptorType(ka.mKind);
            binding.binding = ka.mBinding;

            bindingSet.push_back(binding);
        }

        vk::DescriptorSetLayoutCreateInfo createInfo;
        createInfo.setBindingCount(bindingSet.size())
                .setPBindings(bindingSet.size() ? bindingSet.data() : nullptr);

        return inDevice.createDescriptorSetLayoutUnique(createInfo);
    }

    /***********************************************************************************************
     * arg_spec_t::kind functions
     **********************************************************************************************/

    vk::DescriptorType getDescriptorType(arg_spec_t::kind argKind) {
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

    /***********************************************************************************************
     * OpenCL sampler flags functions
     **********************************************************************************************/

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

    /***********************************************************************************************
     * Validation functions
     **********************************************************************************************/

    void validateModule(const module_spec_t& spec)
    {
        const int sampler_ds = getSamplersDescriptorSet(spec.mSamplers);
        for (auto& ls : spec.mSamplers) {
            // All literal samplers for a module need to be in the same descriptor set
            validateSampler(ls, sampler_ds);
        }

        // If there are literal samplers, the kernel arguments are in descriptor set 1, otherwise
        // they are in descriptor set 0
        const int kernel_ds = (sampler_ds > 0 ? 1 : 0);
        for (auto& k : spec.mKernels) {
            validateKernel(k, kernel_ds);
        }

    }

    void validateKernel(const kernel_spec_t& spec, int requiredDescriptorSet)
    {
        if (spec.mName.empty()) {
            fail_runtime_error("kernel has no name");
        }

        const int arg_ds = getKernelArgumentDescriptorSet(spec.mArguments);
        if (requiredDescriptorSet > 0 && arg_ds != requiredDescriptorSet) {
            fail_runtime_error("kernel's arguments are in incorrect descriptor set");
        }

        for (auto& ka : spec.mArguments) {
            // All arguments for a given kernel that are passed in a descriptor set need to be in
            // the same descriptor set
            if (ka.mKind != arg_spec_t::kind_local && ka.mDescriptorSet != arg_ds) {
                fail_runtime_error("kernel arg descriptor_sets don't match");
            }

            validateKernelArg(ka);
        }

        // TODO: mArguments entries are in increasing binding, and pod/pod_ubo's come after non-pod/non-pod_ubo's
        // TODO: there cannot be both pod and pod_ubo arguments for a given kernel
        // TODO: if there is a pod or pod_ubo argument, its binding must be larger than other descriptor sets
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

    void validateKernelArg(const arg_spec_t &arg) {
        if (arg.mKind == arg_spec_t::kind_unknown) {
            fail_runtime_error("kernel argument kind unknown");
        }
        if (arg.mOrdinal < 0) {
            fail_runtime_error("kernel argument missing ordinal");
        }

        if (arg.mKind == arg_spec_t::kind_local) {
            if (arg.mSpecConstant < 0) {
                fail_runtime_error("local kernel argument missing spec constant");
            }
        }
        else {
            if (arg.mDescriptorSet < 0) {
                fail_runtime_error("kernel argument missing descriptorSet");
            }
            if (arg.mBinding < 0) {
                fail_runtime_error("kernel argument missing binding");
            }
            if (arg.mOffset < 0) {
                fail_runtime_error("kernel argument missing offset");
            }
        }
    }

} // namespace clspv_utils

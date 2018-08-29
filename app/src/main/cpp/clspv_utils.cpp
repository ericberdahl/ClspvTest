//
// Created by Eric Berdahl on 10/22/17.
//

#include "clspv_utils.hpp"

#include "getline_crlf_savvy.hpp"
#include "opencl_types.hpp"
#include "util.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>


namespace {
    using namespace clspv_utils;

    typedef std::unique_ptr<std::FILE, decltype(&std::fclose)>  UniqueFILE;

    UniqueFILE fopen_unique(const char* filename, const char* mode) {
        return UniqueFILE(AndroidFopen(filename, mode), &std::fclose);
    }

    //
    // get_data_hack works around a deficiency in std::string, prior to C++17, in which
    // std::string::data() only returns a const char*.
    //
    template <typename Container>
    void* get_data_hack(Container& c) { return c.data(); }

    template <>
    void* get_data_hack(std::string& c) { return const_cast<char*>(c.data()); }

    template <typename Container>
    void read_file_contents(const std::string& filename, Container& fileContents) {
        const std::size_t wordSize = sizeof(typename Container::value_type);

        UniqueFILE pFile = fopen_unique(filename.c_str(), "rb");
        if (!pFile) {
            throw std::runtime_error("can't open file: " + filename);
        }

        std::fseek(pFile.get(), 0, SEEK_END);

        const auto num_bytes = std::ftell(pFile.get());
        if (0 != (num_bytes % wordSize)) {
            throw std::runtime_error("file size of " + filename + " inappropriate for requested type");
        }

        const auto num_words = (num_bytes + wordSize - 1) / wordSize;
        fileContents.resize(num_words);
        assert(num_bytes == (fileContents.size() * wordSize));

        std::fseek(pFile.get(), 0, SEEK_SET);
        std::fread(get_data_hack(fileContents), 1, num_bytes, pFile.get());
    }

    const auto kArgKind_DescriptorType_Map = {
            std::make_pair(arg_spec_t::kind_pod_ubo, vk::DescriptorType::eUniformBuffer),
            std::make_pair(arg_spec_t::kind_pod, vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_buffer, vk::DescriptorType::eStorageBuffer),
            std::make_pair(arg_spec_t::kind_ro_image, vk::DescriptorType::eSampledImage),
            std::make_pair(arg_spec_t::kind_wo_image, vk::DescriptorType::eStorageImage),
            std::make_pair(arg_spec_t::kind_sampler, vk::DescriptorType::eSampler)
    };

    vk::DescriptorType find_descriptor_type(arg_spec_t::kind_t argKind) {
        auto found = std::find_if(std::begin(kArgKind_DescriptorType_Map),
                                  std::end(kArgKind_DescriptorType_Map),
                                  [argKind](decltype(kArgKind_DescriptorType_Map)::const_reference p) {
                                      return argKind == p.first;
                                  });
        if (found == std::end(kArgKind_DescriptorType_Map)) {
            throw std::runtime_error("unknown argKind encountered");
        }
        return found->second;
    }

    const auto kCLAddressMode_VkAddressMode_Map = {
            std::make_pair(CLK_ADDRESS_NONE, vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP_TO_EDGE, vk::SamplerAddressMode::eClampToEdge),
            std::make_pair(CLK_ADDRESS_CLAMP, vk::SamplerAddressMode::eClampToBorder),
            std::make_pair(CLK_ADDRESS_REPEAT, vk::SamplerAddressMode::eRepeat),
            std::make_pair(CLK_ADDRESS_MIRRORED_REPEAT, vk::SamplerAddressMode::eMirroredRepeat)
    };

    vk::SamplerAddressMode find_address_mode(int opencl_flags) {
        opencl_flags &= CLK_ADDRESS_MASK;

        auto found = std::find_if(std::begin(kCLAddressMode_VkAddressMode_Map),
                                  std::end(kCLAddressMode_VkAddressMode_Map),
                                  [&opencl_flags](decltype(kCLAddressMode_VkAddressMode_Map)::const_reference am) {
                                      return (am.first == opencl_flags);
                                  });

        return (found == std::end(kCLAddressMode_VkAddressMode_Map) ? vk::SamplerAddressMode::eClampToEdge : found->second);
    }


    const auto kSpvMapArgType_ArgKind_Map = {
            std::make_pair("pod", arg_spec_t::kind_pod),
            std::make_pair("pod_ubo", arg_spec_t::kind_pod_ubo),
            std::make_pair("buffer", arg_spec_t::kind_buffer),
            std::make_pair("ro_image", arg_spec_t::kind_ro_image),
            std::make_pair("wo_image", arg_spec_t::kind_wo_image),
            std::make_pair("sampler", arg_spec_t::kind_sampler),
            std::make_pair("local", arg_spec_t::kind_local)
    };

    arg_spec_t::kind_t find_arg_kind(const std::string &argType) {
        auto found = std::find_if(std::begin(kSpvMapArgType_ArgKind_Map),
                                  std::end(kSpvMapArgType_ArgKind_Map),
                                  [&argType](decltype(kSpvMapArgType_ArgKind_Map)::const_reference p) {
                                      return argType == p.first;
                                  });
        if (found == std::end(kSpvMapArgType_ArgKind_Map)) {
            throw std::runtime_error("unknown argType encountered");
        }
        return found->second;
    }

    vk::Filter get_vk_filter(int opencl_flags) {
        return ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ? vk::Filter::eLinear
                                                                      : vk::Filter::eNearest);
    }

    vk::Bool32 get_vk_unnormalized_coordinates(int opencl_flags) {
        return ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_TRUE : VK_FALSE);
    }

    std::string read_csv_field(std::istream& in) {
        std::string result;

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

    typedef std::pair<std::string,std::string> key_value_t;

    key_value_t read_key_value_pair(std::istream& in) {
        return std::make_pair(read_csv_field(in), read_csv_field(in));
    };

    vk::UniqueShaderModule create_shader(vk::Device device, const std::string& spvFilename) {
        std::vector<std::uint32_t> spvModule;
        read_file_contents(spvFilename, spvModule);

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.setCodeSize(spvModule.size() * sizeof(decltype(spvModule)::value_type))
                .setPCode(spvModule.data());

        return device.createShaderModuleUnique(shaderModuleCreateInfo);
    }

    vk::UniqueDescriptorSet allocate_descriptor_set(const device&           inDevice,
                                                    vk::DescriptorSetLayout layout)
    {
        vk::DescriptorSetAllocateInfo createInfo;
        createInfo.setDescriptorPool(inDevice.getDescriptorPool())
                .setDescriptorSetCount(1)
                .setPSetLayouts(&layout);

        return std::move(inDevice.getDevice().allocateDescriptorSetsUnique(createInfo)[0]);
    }

    vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                      device,
                                                    vk::ArrayProxy<const vk::DescriptorSetLayout>   layouts)
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayoutCount(layouts.size())
                .setPSetLayouts(layouts.data());

        return device.createPipelineLayoutUnique(createInfo);
    }

    void validate_sampler(const sampler_spec_t& sampler) {
        const auto fail = [](const char* message) {
            throw std::runtime_error(message);
        };

        if (sampler.opencl_flags == 0) {
            fail("sampler missing OpenCL flags");
        }
        if (sampler.descriptor_set < 0) {
            fail("sampler missing descriptorSet");
        }
        if (sampler.binding < 0) {
            fail("sampler missing binding");
        }

        // all samplers, are documented to share descriptor set 0
        if (sampler.descriptor_set != 0) {
            fail("all clspv literal samplers must use descriptor set 0");
        }

        if (!isSamplerSupported(sampler.opencl_flags)) {
            fail("sampler is not representable in Vulkan");
        }
    }

    void validate_kernel_arg(const arg_spec_t& arg) {
        const auto fail = [](const char* message) {
            throw std::runtime_error(message);
        };

        if (arg.kind == arg_spec_t::kind_unknown) {
            fail("kernel argument kind unknown");
        }
        if (arg.ordinal < 0) {
            fail("kernel argument missing ordinal");
        }

        if (arg.kind == arg_spec_t::kind_local) {
            if (arg.spec_constant < 0) {
                fail("local kernel argument missing spec constant");
            }
        }
        else {
            if (arg.descriptor_set < 0) {
                fail("kernel argument missing descriptorSet");
            }
            if (arg.binding < 0) {
                fail("kernel argument missing binding");
            }
            if (arg.offset < 0) {
                fail("kernel argument missing offset");
            }
        }
    }

    sampler_spec_t parse_spvmap_sampler(key_value_t tag, std::istream& in) {
        sampler_spec_t result;

        result.opencl_flags = std::atoi(tag.second.c_str());

        while (!in.eof()) {
            tag = read_key_value_pair(in);

            if ("descriptorSet" == tag.first) {
                result.descriptor_set = std::atoi(tag.second.c_str());
            } else if ("binding" == tag.first) {
                result.binding = std::atoi(tag.second.c_str());
            }
        }

        return result;
    }

    arg_spec_t parse_spvmap_kernel_arg(key_value_t tag, std::istream& in) {
        arg_spec_t result;

        while (!in.eof()) {
            tag = read_key_value_pair(in);

            if ("argOrdinal" == tag.first) {
                result.ordinal = std::atoi(tag.second.c_str());
            } else if ("descriptorSet" == tag.first) {
                result.descriptor_set = std::atoi(tag.second.c_str());
            } else if ("binding" == tag.first) {
                result.binding = std::atoi(tag.second.c_str());
            } else if ("offset" == tag.first) {
                result.offset = std::atoi(tag.second.c_str());
            } else if ("argKind" == tag.first) {
                result.kind = find_arg_kind(tag.second);
            } else if ("arrayElemSize" == tag.first) {
                // arrayElemSize is ignored by clspvtest
            } else if ("arrayNumElemSpecId" == tag.first) {
                result.spec_constant = std::atoi(tag.second.c_str());
            }

        }

        return result;
    }

    const kernel_interface* find_kernel_interface(const std::string&                        name,
                                                  vk::ArrayProxy<const kernel_interface>    kernels)
    {
        auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                   [&name](const kernel_interface &iter) {
                                       return iter.getEntryPoint() == name;
                                   });

        return (kernel == kernels.end() ? nullptr : &(*kernel));
    }

    std::vector<std::string> get_entry_points(vk::ArrayProxy<const kernel_interface> kernels)
    {
        std::vector<std::string> result;

        std::transform(kernels.begin(), kernels.end(),
                       std::back_inserter(result),
                       [](const kernel_interface& k) { return k.getEntryPoint(); });

        return result;
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

    kernel_interface::kernel_interface()
            : mLiteralSamplers(nullptr)
    {
    }

    kernel_interface::kernel_interface(std::string            entryPoint,
                                       sampler_list_proxy_t   samplers,
                                       arg_list_t             arguments)
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
        const auto fail = [](const char* message) {
            throw std::runtime_error(message);
        };

        if (mName.empty()) {
            fail("kernel has no name");
        }

        const int arg_ds = getArgDescriptorSet();
        for (auto& ka : mArgSpecs) {
            // All arguments for a given kernel that are passed in a descriptor set need to be in
            // the same descriptor set
            if (ka.kind != arg_spec_t::kind_local && ka.descriptor_set != arg_ds) {
                fail("kernel arg descriptor_sets don't match");
            }

            validate_kernel_arg(ka);
        }

        const int sampler_ds = getLiteralSamplersDescriptorSet();
        for (auto& ls : mLiteralSamplers) {
            // All literal samplers for a given kernel need to be in the same descriptor set
            if (ls.descriptor_set != sampler_ds) {
                fail("literal sampler descriptor_sets don't match");
            }

            validate_sampler(ls);
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

        std::vector<vk::DescriptorSetLayoutBinding> bindingSet;

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

    module_interface::module_interface()
    {
    }

    module_interface::module_interface(const std::string& moduleName)
            : module_interface()
    {
        mName = moduleName;

        std::string buffer;
        read_file_contents(moduleName + ".spvmap", buffer);

        std::map<std::string, kernel_interface::arg_list_t> kernel_args;

        std::string line;
        std::istringstream in(buffer);
        while (!in.eof()) {
            // spvmap files may have been generated on a system which uses different line ending
            // conventions than the system on which the consumer runs. Safer to fetch lines
            // using a function which recognizes multiple line endings.
            crlf_savvy::getline(in, line);

            std::istringstream in_line(line);
            auto tag = read_key_value_pair(in_line);
            if ("sampler" == tag.first) {
                addLiteralSampler(parse_spvmap_sampler(tag, in_line));
            } else if ("kernel" == tag.first) {
                kernel_args[tag.second].push_back(parse_spvmap_kernel_arg(tag, in_line));
            }
        }

        // Ensure that the literal samplers are sorted by increasing binding number. This will be
        // important if the sequence is later used to determine whether a cached sampler descriptor
        // set can be re-used for this module.
        std::sort(mSamplers.begin(), mSamplers.end(), [](const sampler_spec_t& lhs, const sampler_spec_t& rhs) {
            return lhs.binding < rhs.binding;
        });

        for (auto& k : kernel_args) {
            mKernels.push_back(kernel_interface(k.first, mSamplers, k.second));
        }
    }

    void module_interface::addLiteralSampler(clspv_utils::sampler_spec_t sampler) {
        validate_sampler(sampler);
        mSamplers.push_back(sampler);
    }

    const kernel_interface* module_interface::findKernelInterface(const std::string& name) const {
        return find_kernel_interface(name, mKernels);
    }

    int module_interface::getLiteralSamplersDescriptorSet() const {
        auto found = std::find_if(mSamplers.begin(), mSamplers.end(), [](const sampler_spec_t& ss) {
            return (-1 != ss.descriptor_set);
        });
        return (found == mSamplers.end() ? -1 : found->descriptor_set);
    }

    std::vector<std::string> module_interface::getEntryPoints() const
    {
        return get_entry_points(mKernels);
    }

    kernel_module module_interface::load(device inDevice) const
    {
        const int ds = getLiteralSamplersDescriptorSet();
        assert( (mSamplers.empty() && -1 == ds) || (!mSamplers.empty() && 0 == ds) );

        //
        // Create literal sampler descriptor set
        //

        const auto literalSamplerDescriptorGroup = inDevice.getCachedSamplerDescriptorGroup(mSamplers);

        return kernel_module(mName,
                             inDevice,
                             literalSamplerDescriptorGroup.descriptor,
                             literalSamplerDescriptorGroup.layout,
                             mKernels);
    }

    bool isSamplerSupported(int opencl_flags)
    {
        const vk::Bool32 unnormalizedCoordinates    = get_vk_unnormalized_coordinates(opencl_flags);
        const vk::SamplerAddressMode addressMode    = find_address_mode(opencl_flags);

        return (!unnormalizedCoordinates || addressMode == vk::SamplerAddressMode::eClampToEdge || addressMode == vk::SamplerAddressMode::eClampToBorder);
    }

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags) {
        if (!isSamplerSupported(opencl_flags)) {
            throw std::runtime_error("This OpenCL sampler cannot be represented in Vulkan");
        }

        const vk::Filter filter                     = get_vk_filter(opencl_flags);
        const vk::Bool32 unnormalizedCoordinates    = get_vk_unnormalized_coordinates(opencl_flags);
        const vk::SamplerAddressMode addressMode    = find_address_mode(opencl_flags);

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

    execution_time_t::execution_time_t() :
            cpu_duration(0),
            timestamps()
    {
    }

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

            std::vector<vk::DescriptorSetLayoutBinding> bindingSet;

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

            std::vector<vk::DescriptorImageInfo> literalSamplerInfo;
            std::vector<vk::WriteDescriptorSet> literalSamplerDescriptorWrites;

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

    kernel_module::kernel_module()
            : mKernelInterfaces(nullptr)
    {
    }

    kernel_module::kernel_module(kernel_module&& other)
            : kernel_module()
    {
        swap(other);
    }

    kernel_module::kernel_module(const std::string&         moduleName,
                                 device                     inDevice,
                                 vk::DescriptorSet          literalSamplerDescriptor,
                                 vk::DescriptorSetLayout    literalSamplerDescriptorLayout,
                                 kernel_list_proxy_t        kernelInterfaces)
            : mName(moduleName),
              mDevice(inDevice),
              mKernelInterfaces(kernelInterfaces),
              mLiteralSamplerDescriptor(literalSamplerDescriptor),
              mLiteralSamplerDescriptorLayout(literalSamplerDescriptorLayout)
    {
        const std::string spvFilename = mName + ".spv";
        mShaderModule = create_shader(mDevice.getDevice(), spvFilename.c_str());
        mPipelineCache = mDevice.getDevice().createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
    }

    kernel_module::~kernel_module()
    {
    }

    kernel_module& kernel_module::operator=(kernel_module&& other)
    {
        swap(other);
        return *this;
    }

    void kernel_module::swap(kernel_module& other)
    {
        using std::swap;

        swap(mName, other.mName);
        swap(mDevice, other.mDevice);
        swap(mKernelInterfaces, other.mKernelInterfaces);
        swap(mLiteralSamplerDescriptorLayout, other.mLiteralSamplerDescriptorLayout);
        swap(mLiteralSamplerDescriptor, other.mLiteralSamplerDescriptor);
        swap(mShaderModule, other.mShaderModule);
        swap(mPipelineCache, other.mPipelineCache);
    }

    std::vector<std::string> kernel_module::getEntryPoints() const
    {
        return get_entry_points(mKernelInterfaces);
    }

    kernel_layout_t kernel_module::createKernelLayout(const std::string& entryPoint) const {
        if (!isLoaded()) {
            throw std::runtime_error("cannot create layout for unloaded module");
        }

        kernel_layout_t result;

        const auto kernelInterface = find_kernel_interface(entryPoint, mKernelInterfaces);
        if (!kernelInterface) {
            throw std::runtime_error("cannot create kernel layout for unknown entry point");
        }

        if (-1 != kernelInterface->getArgDescriptorSet()) {
            result.mArgumentDescriptorLayout = kernelInterface->createArgDescriptorLayout(mDevice);

            result.mArgumentsDescriptor = allocate_descriptor_set(mDevice,
                                                                  *result.mArgumentDescriptorLayout);
        }

        result.mLiteralSamplerDescriptor = mLiteralSamplerDescriptor;

        std::vector<vk::DescriptorSetLayout> layouts;
        if (mLiteralSamplerDescriptorLayout) layouts.push_back(mLiteralSamplerDescriptorLayout);
        if (result.mArgumentDescriptorLayout) layouts.push_back(*result.mArgumentDescriptorLayout);
        result.mPipelineLayout = create_pipeline_layout(mDevice.getDevice(), layouts);

        return result;
    }

    kernel kernel_module::createKernel(const std::string&   entryPoint,
                                       const vk::Extent3D&  workgroup_sizes)
    {
        return kernel(mDevice,
                      createKernelLayout(entryPoint),
                      *mShaderModule,
                      *mPipelineCache,
                      entryPoint,
                      workgroup_sizes,
                      find_kernel_interface(entryPoint, mKernelInterfaces)->mArgSpecs);
    }

    kernel::kernel()
            : mArgList(nullptr)
    {
    }

    kernel::kernel(device               inDevice,
                   kernel_layout_t      layout,
                   vk::ShaderModule     shaderModule,
                   vk::PipelineCache    pipelineCache,
                   std::string          entryPoint,
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
        std::vector<std::uint32_t> specConstants = {
                mWorkgroupSizes.width,
                mWorkgroupSizes.height,
                mWorkgroupSizes.depth
        };
        typedef decltype(specConstants)::value_type spec_constant_t;
        std::copy(otherSpecConstants.begin(), otherSpecConstants.end(), std::back_inserter(specConstants));

        std::vector<vk::SpecializationMapEntry> specializationEntries;
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

    kernel_invocation::kernel_invocation()
            : mArgList(nullptr)
    {
        // this space intentionally left blank
    }

    kernel_invocation::kernel_invocation(kernel&            kernel,
                                         device             inDevice,
                                         vk::DescriptorSet  argumentDescSet,
                                         arg_list_proxy_t   argList)
            : kernel_invocation()
    {
        mKernel = &kernel;
        mDevice = inDevice;
        mArgList = argList;
        mArgumentDescriptorSet = argumentDescSet;

        mCommand = vulkan_utils::allocate_command_buffer(mDevice.getDevice(), mDevice.getCommandPool());

        vk::QueryPoolCreateInfo poolCreateInfo;
        poolCreateInfo.setQueryType(vk::QueryType::eTimestamp)
                .setQueryCount(kQueryIndex_Count);

        mQueryPool = mDevice.getDevice().createQueryPoolUnique(poolCreateInfo);
    }

    kernel_invocation::kernel_invocation(kernel_invocation&& other)
            : kernel_invocation()
    {
        swap(other);
    }

    kernel_invocation::~kernel_invocation() {
    }

    void kernel_invocation::swap(kernel_invocation& other)
    {
        using std::swap;

        swap(mKernel, other.mKernel);
        swap(mDevice, other.mDevice);
        swap(mArgList, other.mArgList);
        swap(mCommand, other.mCommand);
        swap(mQueryPool, other.mQueryPool);

        swap(mArgumentDescriptorSet, other.mArgumentDescriptorSet);

        swap(mSpecConstantArguments, other.mSpecConstantArguments);
        swap(mBufferMemoryBarriers, other.mBufferMemoryBarriers);
        swap(mImageMemoryBarriers, other.mImageMemoryBarriers);

        swap(mImageArgumentInfo, other.mImageArgumentInfo);
        swap(mBufferArgumentInfo, other.mBufferArgumentInfo);
        swap(mArgumentDescriptorWrites, other.mArgumentDescriptorWrites);
    }

    std::size_t kernel_invocation::countArguments() const {
        return mArgumentDescriptorWrites.size() + mSpecConstantArguments.size();
    }

    std::uint32_t kernel_invocation::validateArgType(std::size_t        ordinal,
                                                     vk::DescriptorType kind) const
    {
        const auto fail = [](const char* message) {
            throw std::runtime_error(message);
        };

        if (ordinal >= mArgList.size()) {
            fail("adding too many arguments to kernel invocation");
        }

        auto& ka = mArgList.data()[ordinal];
        if (find_descriptor_type(ka.kind) != kind) {
            fail("adding incompatible argument to kernel invocation");
        }

        return ka.binding;
    }

    std::uint32_t kernel_invocation::validateArgType(std::size_t        ordinal,
                                                     arg_spec_t::kind_t kind) const
    {
        const auto fail = [](const char* message) {
            throw std::runtime_error(message);
        };

        if (ordinal >= mArgList.size()) {
            fail("adding too many arguments to kernel invocation");
        }

        auto& ka = mArgList.data()[ordinal];
        if (ka.kind != kind) {
            fail("adding incompatible argument to kernel invocation");
        }

        return ka.binding;
    }

    void kernel_invocation::addStorageBufferArgument(vulkan_utils::storage_buffer& buffer) {
        mBufferMemoryBarriers.push_back(buffer.prepareForRead());
        mBufferMemoryBarriers.push_back(buffer.prepareForWrite());
        mBufferArgumentInfo.push_back(buffer.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eStorageBuffer))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer) {
        mBufferMemoryBarriers.push_back(buffer.prepareForRead());
        mBufferArgumentInfo.push_back(buffer.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eUniformBuffer))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addSamplerArgument(vk::Sampler samp) {
        vk::DescriptorImageInfo samplerInfo;
        samplerInfo.setSampler(samp);
        mImageArgumentInfo.push_back(samplerInfo);

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eSampler))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampler);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addReadOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eShaderReadOnlyOptimal));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eSampledImage))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampledImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addWriteOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eGeneral));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(validateArgType(countArguments(), vk::DescriptorType::eStorageImage))
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addLocalArraySizeArgument(unsigned int numElements) {
        validateArgType(countArguments(), arg_spec_t::kind_t::kind_local);
        mSpecConstantArguments.push_back(numElements);
    }

    void kernel_invocation::updateDescriptorSets() {
        //
        // Set up to create the descriptor set write structures for arguments.
        // We will iterate the param lists in the same order,
        // picking up image and buffer infos in order.
        //

        std::vector<vk::WriteDescriptorSet> writeSets;

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
        mDevice.getDevice().updateDescriptorSets(writeSets, nullptr);
    }

    void kernel_invocation::bindCommand()
    {
        mKernel->bindCommand(*mCommand);
    }

    void kernel_invocation::fillCommandBuffer(const vk::Extent3D& num_workgroups)
    {
        mCommand->begin(vk::CommandBufferBeginInfo());

        bindCommand();

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

    void kernel_invocation::submitCommand() {
        vk::CommandBuffer rawCommand = *mCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        mDevice.getComputeQueue().submit(submitInfo, nullptr);

    }

    void kernel_invocation::updatePipeline()
    {
        mKernel->updatePipeline(mSpecConstantArguments);
    }

    execution_time_t kernel_invocation::run(const vk::Extent3D& num_workgroups) {
        // HACK re-create the pipeline if the invocation includes spec constant arguments.
        // TODO factor the pipeline recreation better, possibly along with an overhaul of kernel
        // management
        if (!mSpecConstantArguments.empty()) {
            updatePipeline();
        }

        updateDescriptorSets();
        fillCommandBuffer(num_workgroups);

        auto start = std::chrono::high_resolution_clock::now();
        submitCommand();
        mDevice.getComputeQueue().waitIdle();
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t timestamps[kQueryIndex_Count];
        mDevice.getDevice().getQueryPoolResults(*mQueryPool,
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

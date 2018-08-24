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
#include <iostream>
#include <limits>
#include <memory>


namespace {
    using namespace clspv_utils;

    int sampler_descriptor_set(const details::spv_map& spv_map) {
        return (spv_map.samplers.empty() ? -1 : spv_map.samplers[0].descriptor_set);
    }

    int kernel_descriptor_set(const details::spv_map::kernel& kernel) {
        return (kernel.args.empty() ? -1 : kernel.args[0].descriptor_set);
    }

    const auto kArgKind_DescriptorType_Map = {
            std::make_pair(details::spv_map::arg::kind_pod_ubo, vk::DescriptorType::eUniformBuffer),
            std::make_pair(details::spv_map::arg::kind_pod, vk::DescriptorType::eStorageBuffer),
            std::make_pair(details::spv_map::arg::kind_buffer, vk::DescriptorType::eStorageBuffer),
            std::make_pair(details::spv_map::arg::kind_ro_image, vk::DescriptorType::eSampledImage),
            std::make_pair(details::spv_map::arg::kind_wo_image, vk::DescriptorType::eStorageImage),
            std::make_pair(details::spv_map::arg::kind_sampler, vk::DescriptorType::eSampler)
    };

    vk::DescriptorType find_descriptor_type(details::spv_map::arg::kind_t argKind) {
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
            std::make_pair("pod", details::spv_map::arg::kind_pod),
            std::make_pair("pod_ubo", details::spv_map::arg::kind_pod_ubo),
            std::make_pair("buffer", details::spv_map::arg::kind_buffer),
            std::make_pair("ro_image", details::spv_map::arg::kind_ro_image),
            std::make_pair("wo_image", details::spv_map::arg::kind_wo_image),
            std::make_pair("sampler", details::spv_map::arg::kind_sampler),
            std::make_pair("local", details::spv_map::arg::kind_local)
    };

    details::spv_map::arg::kind_t find_arg_kind(const std::string &argType) {
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

    details::spv_map create_spv_map(const char *spvmapFilename) {
        // Read the spvmap file into a string buffer
        std::unique_ptr<std::FILE, decltype(&std::fclose)> spvmap_file(AndroidFopen(spvmapFilename, "rb"),
                                                                       &std::fclose);
        assert(spvmap_file);

        std::fseek(spvmap_file.get(), 0, SEEK_END);
        std::string buffer(std::ftell(spvmap_file.get()), ' ');
        std::fseek(spvmap_file.get(), 0, SEEK_SET);
        std::fread(&buffer.front(), 1, buffer.length(), spvmap_file.get());

        spvmap_file.reset();

        // parse the spvmap file contents
        std::istringstream in(buffer);
        return details::spv_map::parse(in);
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
        std::unique_ptr<std::FILE, decltype(&std::fclose)> spv_file(AndroidFopen(spvFilename.c_str(), "rb"),
                                                                    &std::fclose);
        if (!spv_file) {
            throw std::runtime_error("can't open file: " + spvFilename);
        }

        std::fseek(spv_file.get(), 0, SEEK_END);
        // Use vector of uint32_t to ensure alignment is satisfied.
        const auto num_bytes = std::ftell(spv_file.get());
        if (0 != (num_bytes % sizeof(uint32_t))) {
            throw std::runtime_error("file size of " + spvFilename + " inappropriate for spv file");
        }
        const auto num_words = (num_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
        std::vector<uint32_t> spvModule(num_words);
        assert(num_bytes == (spvModule.size() * sizeof(uint32_t)));

        std::fseek(spv_file.get(), 0, SEEK_SET);
        std::fread(spvModule.data(), 1, num_bytes, spv_file.get());

        spv_file.reset();

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.setCodeSize(num_bytes)
                .setPCode(spvModule.data());

        return device.createShaderModuleUnique(shaderModuleCreateInfo);
    }

    vk::UniqueDescriptorSet allocate_descriptor_set(vk::Device              device,
                                                    vk::DescriptorPool      pool,
                                                    vk::DescriptorSetLayout layout)
    {
        vk::DescriptorSetAllocateInfo createInfo;
        createInfo.setDescriptorPool(pool)
                .setDescriptorSetCount(1)
                .setPSetLayouts(&layout);

        return std::move(device.allocateDescriptorSetsUnique(createInfo)[0]);
    }

    vk::UniqueDescriptorSetLayout create_descriptor_set_layout(
            vk::Device                                  device,
            vk::ArrayProxy<const vk::DescriptorType>    descriptorTypes)
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindingSet;

        vk::DescriptorSetLayoutBinding binding;
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute)
                .setDescriptorCount(1)
                .setBinding(0);

        for (auto type : descriptorTypes) {
            binding.descriptorType = type;
            bindingSet.push_back(binding);

            ++binding.binding;
        }

        vk::DescriptorSetLayoutCreateInfo createInfo;
        createInfo.setBindingCount(bindingSet.size())
                .setPBindings(bindingSet.size() ? bindingSet.data() : nullptr);

        return device.createDescriptorSetLayoutUnique(createInfo);
    }

    vk::UniqueDescriptorSetLayout create_literalsampler_descriptor_layout(vk::Device device,
                                                                         const details::spv_map& spvMap) {
        vk::UniqueDescriptorSetLayout result;

        if (!spvMap.samplers.empty()) {
            assert(0 == spvMap.samplers_desc_set);

            const std::vector<vk::DescriptorType> descriptorTypes(spvMap.samplers.size(),
                                                                  vk::DescriptorType::eSampler);
            result = create_descriptor_set_layout(device, descriptorTypes);
        }

        return result;
    }

    vk::UniqueDescriptorSetLayout create_arg_descriptor_layout(vk::Device               device,
                                                               const details::spv_map&  spvMap,
                                                               const std::string&       entryPoint) {
        const auto kernel = spvMap.findKernel(entryPoint);
        if (!kernel) {
            throw std::runtime_error("entryPoint not found; cannot create descriptor layout");
        }

        std::vector<vk::DescriptorType> descriptorTypes;

        assert(kernel->descriptor_set == (spvMap.samplers.empty() ? 0 : 1));

        // If the caller has asked only for a pipeline layout for a single entry point,
        // create empty descriptor layouts for all argument descriptors other than the
        // one used by the requested entry point.
        for (auto &ka : kernel->args) {
            // ignore any argument not in offset 0
            if (0 != ka.offset) continue;

            descriptorTypes.push_back(find_descriptor_type(ka.kind));
        }

        return create_descriptor_set_layout(device, descriptorTypes);
    }

    vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                      device,
                                                    vk::ArrayProxy<const vk::DescriptorSetLayout>   layouts)
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayoutCount(layouts.size())
                .setPSetLayouts(layouts.data());

        return device.createPipelineLayoutUnique(createInfo);
    }

    std::vector<std::string> validate_sampler(const details::spv_map::sampler& sampler) {
        std::vector<std::string> result;

        if (sampler.opencl_flags == 0) {
            result.push_back("sampler missing OpenCL flags");
        }
        if (sampler.descriptor_set < 0) {
            result.push_back("sampler missing descriptorSet");
        }
        if (sampler.binding < 0) {
            result.push_back("sampler missing binding");
        }

        return result;
    }

    std::vector<std::string> validate_kernel_arg(const details::spv_map::arg& arg) {
        std::vector<std::string> result;

        if (arg.kind == details::spv_map::arg::kind_unknown) {
            result.push_back("kernel argument kind unknown");
        }
        if (arg.ordinal < 0) {
            result.push_back("kernel argument missing ordinal");
        }

        if (arg.kind == details::spv_map::arg::kind_local) {
            if (arg.spec_constant < 0) {
                result.push_back("kernel argument missing spec constant");
            }
        }
        else {
            if (arg.descriptor_set < 0) {
                result.push_back("kernel argument missing descriptorSet");
            }
            if (arg.binding < 0) {
                result.push_back("kernel argument missing binding");
            }
            if (arg.offset < 0) {
                result.push_back("kernel argument missing offset");
            }
        }

        return result;
    }

    std::vector<std::string> validate_kernel(const details::spv_map::kernel& kernel) {
        std::vector<std::string> result;
        std::vector<std::string> tempErrors;

        if (kernel.name.empty()) {
            result.push_back("kernel has no name");
        }

        const int arg_ds = kernel_descriptor_set(kernel);
        for (auto& ka : kernel.args) {
            tempErrors = validate_kernel_arg(ka);
            result.insert(result.end(), tempErrors.begin(), tempErrors.end());
            tempErrors.clear();

            if (ka.kind != details::spv_map::arg::kind_local && ka.descriptor_set != arg_ds) {
                result.push_back("kernel arg descriptor_sets don't match");
            }
        }

        return result;
    }

    std::vector<std::string> validate_spvmap(const details::spv_map& spvmap) {
        std::vector<std::string> result;
        std::vector<std::string> tempErrors;

        for (auto& k : spvmap.kernels) {
            tempErrors = validate_kernel(k);
            result.insert(result.end(), tempErrors.begin(), tempErrors.end());
            tempErrors.clear();
        }

        const int sampler_ds = sampler_descriptor_set(spvmap);
        for (auto& s : spvmap.samplers) {
            tempErrors = validate_sampler(s);
            result.insert(result.end(), tempErrors.begin(), tempErrors.end());
            tempErrors.clear();

            if (s.descriptor_set != sampler_ds) {
                result.push_back("sampler descriptor_sets don't match");
            }
        }

        return result;
    }

    details::spv_map::sampler parse_spvmap_sampler(key_value_t tag, std::istream& in) {
        details::spv_map::sampler result;

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

    details::spv_map::arg parse_spvmap_kernel_arg(key_value_t tag, std::istream& in) {
            details::spv_map::arg result;

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

    vk::Sampler getCachedSampler(device_t& device, const details::spv_map::sampler& s) {
        if (!device.mSamplerCache) {
            device.mSamplerCache.reset(new device_t::sampler_cache_t);
        }

        if (!device.mSamplerCache->count(s.opencl_flags)) {
            (*device.mSamplerCache)[s.opencl_flags] = createCompatibleSampler(device.mDevice, s.opencl_flags);
        }
        return *(*device.mSamplerCache)[s.opencl_flags];
    }
} // anonymous namespace

namespace clspv_utils {

    namespace details {

        spv_map spv_map::parse(std::istream &in) {
            spv_map result;

            while (!in.eof()) {
                // read one line
                std::string line;

                // spvmap files may have been generated on a system which uses different line ending
                // conventions than the system on which the consumer runs. Safer to fetch lines
                // using a function which recognizes multiple line endings.
                crlf_savvy::getline(in, line);

                std::istringstream in_line(line);
                auto tag = read_key_value_pair(in_line);
                if ("sampler" == tag.first) {
                    auto sampler = parse_spvmap_sampler(tag, in_line);

                    // all samplers, if any, are documented to share descriptor set 0
                    assert(sampler.descriptor_set == 0);

                    if (-1 == result.samplers_desc_set) {
                        result.samplers_desc_set = sampler.descriptor_set;
                    }

                    result.samplers.push_back(sampler);
                } else if ("kernel" == tag.first) {
                    auto kernel_arg = parse_spvmap_kernel_arg(tag, in_line);

                    auto kernel = result.findKernel(tag.second);
                    if (!kernel) {
                        result.kernels.push_back(spv_map::kernel());
                        kernel = &result.kernels.back();
                        kernel->name = tag.second;
                    }
                    assert(kernel);

                    if (-1 == kernel->descriptor_set && -1 != kernel_arg.descriptor_set) {
                        kernel->descriptor_set = kernel_arg.descriptor_set;
                    }

                    if (kernel->args.size() <= kernel_arg.ordinal) {
                        kernel->args.resize(kernel_arg.ordinal + 1, spv_map::arg());
                    }
                    kernel->args[kernel_arg.ordinal] = kernel_arg;
                }
            }

            auto validationErrors = validate_spvmap(result);
            if (!validationErrors.empty()) {
                std::ostringstream os;
                for (auto& s : validationErrors) {
                    os << s << std::endl;
                }
                throw std::runtime_error(os.str());
            }

            return result;
        }

        spv_map::kernel* spv_map::findKernel(const std::string& name) {
            return const_cast<kernel*>(const_cast<const spv_map*>(this)->findKernel(name));
        }

        const spv_map::kernel* spv_map::findKernel(const std::string& name) const {
            auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                       [&name](const spv_map::kernel &iter) {
                                           return iter.name == name;
                                       });

            return (kernel == kernels.end() ? nullptr : &(*kernel));
        }

    } // namespace details

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags) {
        const vk::Filter filter = ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ?
                                   vk::Filter::eLinear :
                                   vk::Filter::eNearest);
        const vk::Bool32 unnormalizedCoordinates = ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_TRUE : VK_FALSE);
        const auto addressMode = find_address_mode(opencl_flags);
        if (unnormalizedCoordinates && (addressMode != vk::SamplerAddressMode::eClampToEdge && addressMode != vk::SamplerAddressMode::eClampToBorder)) {
            throw std::runtime_error("This OpenCL sampler cannot be represented in Vulkan");
        }

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

    device_t::device_t(vk::PhysicalDevice                  physicalDevice,
                       vk::Device                          device,
                       vk::PhysicalDeviceMemoryProperties  memoryProperties,
                       vk::DescriptorPool                  descriptorPool,
                       vk::CommandPool                     commandPool,
                       vk::Queue                           computeQueue)
            : mPhysicalDevice(physicalDevice),
              mDevice(device),
              mMemoryProperties(memoryProperties),
              mDescriptorPool(descriptorPool),
              mCommandPool(commandPool),
              mComputeQueue(computeQueue),
              mSamplerCache(new sampler_cache_t)
    {
    }

    kernel_module::kernel_module(const std::string& moduleName) :
            mDevice(),
            mName(moduleName),
            mShaderModule(),
            mSpvMap()
    {
        const std::string mapFilename = mName + ".spvmap";
        mSpvMap = create_spv_map(mapFilename.c_str());
    }

    kernel_module::~kernel_module() {
    }

    void kernel_module::load(device_t device) {
        if (getShaderModule()) {
            throw std::runtime_error("kernel_module already loaded");
        }
        mDevice = device;

        const std::string spvFilename = mName + ".spv";
        mShaderModule = create_shader(mDevice.mDevice, spvFilename.c_str());

        //
        // Create literal sampler descriptor set
        //

        std::vector<vk::DescriptorImageInfo> literalSamplerInfo;
        for (auto s : mSpvMap.samplers) {
            vk::DescriptorImageInfo samplerInfo;
            samplerInfo.setSampler(getCachedSampler(mDevice, s));

            literalSamplerInfo.push_back(samplerInfo);
        }

        if (!literalSamplerInfo.empty()) {
            assert(0 == mSpvMap.samplers_desc_set);
            mLiteralSamplerDescriptorLayout = create_literalsampler_descriptor_layout(mDevice.mDevice, mSpvMap);

            mLiteralSamplerDescriptor = allocate_descriptor_set(mDevice.mDevice,
                                                                mDevice.mDescriptorPool,
                                                                *mLiteralSamplerDescriptorLayout);

            vk::WriteDescriptorSet literalSamplerSet;
            literalSamplerSet.setDstSet(*mLiteralSamplerDescriptor)
                    .setDstBinding(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eSampler);

            std::vector<vk::WriteDescriptorSet> literalSamplerDescriptorWrites;

            for (auto& lsd : literalSamplerInfo) {
                literalSamplerSet.setPImageInfo(&lsd);
                literalSamplerDescriptorWrites.push_back(literalSamplerSet);
                ++literalSamplerSet.dstBinding;
            }

            mDevice.mDevice.updateDescriptorSets(literalSamplerDescriptorWrites, nullptr);
        }
        else {
            assert(-1 == mSpvMap.samplers_desc_set);
        }
    }

    std::vector<std::string> kernel_module::getEntryPoints() const {
        std::vector<std::string> result;

        std::transform(mSpvMap.kernels.begin(), mSpvMap.kernels.end(),
                       std::back_inserter(result),
                       [](const details::spv_map::kernel& k) { return k.name; });

        return result;
    }

    kernel_layout_t kernel_module::createKernelLayout(const std::string& entryPoint) const {
        if (!isLoaded()) {
            throw std::runtime_error("cannot create layout for unloaded module");
        }

        kernel_layout_t result;

        const auto kernel_arg_map = mSpvMap.findKernel(entryPoint);
        if (!kernel_arg_map) {
            throw std::runtime_error("cannot create kernel layout for unknown entry point");
        }

        if (-1 != kernel_arg_map->descriptor_set) {
            result.mArgumentDescriptorLayout = create_arg_descriptor_layout(mDevice.mDevice, mSpvMap, entryPoint);

            result.mArgumentsDescriptor = allocate_descriptor_set(mDevice.mDevice,
                                                                  mDevice.mDescriptorPool,
                                                                  *result.mArgumentDescriptorLayout);
        }

        result.mLiteralSamplerDescriptor = *mLiteralSamplerDescriptor;

        std::vector<vk::DescriptorSetLayout> layouts;
        if (mLiteralSamplerDescriptorLayout) layouts.push_back(*mLiteralSamplerDescriptorLayout);
        if (result.mArgumentDescriptorLayout) layouts.push_back(*result.mArgumentDescriptorLayout);
        result.mPipelineLayout = create_pipeline_layout(mDevice.mDevice, layouts);

        return result;
    }

    kernel kernel_module::createKernel(const std::string&   entryPoint,
                                       const vk::Extent3D&  workgroup_sizes)
    {
        return kernel(mDevice,
                      createKernelLayout(entryPoint),
                      *mShaderModule,
                      entryPoint,
                      workgroup_sizes);
    }

    kernel::kernel()
    {
    }

    kernel::kernel(device_t             device,
                   kernel_layout_t      layout,
                   vk::ShaderModule     shaderModule,
                   std::string          entryPoint,
                   const vk::Extent3D&  workgroup_sizes) :
            mDevice(device),
            mShaderModule(shaderModule),
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mLayout(std::move(layout)),
            mPipeline()
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
        swap(mPipeline, other.mPipeline);
    }

    kernel_invocation kernel::createInvocation()
    {
        return kernel_invocation(*this,
                                 mDevice,
                                 *mLayout.mArgumentsDescriptor);
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

        mPipeline = mDevice.mDevice.createComputePipelineUnique(vk::PipelineCache(), createInfo);
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
            : mKernel(nullptr)
    {
        // this space intentionally left blank
    }

    kernel_invocation::kernel_invocation(kernel&            kernel,
                                         device_t           device,
                                         vk::DescriptorSet  argumentDescSet)
            : kernel_invocation()
    {
        mKernel = &kernel;
        mDevice = device;
        mArgumentDescriptorSet = argumentDescSet;

        mCommand = vulkan_utils::allocate_command_buffer(mDevice.mDevice, mDevice.mCommandPool);

        vk::QueryPoolCreateInfo poolCreateInfo;
        poolCreateInfo.setQueryType(vk::QueryType::eTimestamp)
                .setQueryCount(kQueryIndex_Count);

        mQueryPool = mDevice.mDevice.createQueryPoolUnique(poolCreateInfo);
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
        swap(mCommand, other.mCommand);
        swap(mQueryPool, other.mQueryPool);

        swap(mArgumentDescriptorSet, other.mArgumentDescriptorSet);

        swap(mSpecConstantArguments, other.mSpecConstantArguments);
        swap(mImageMemoryBarriers, other.mImageMemoryBarriers);

        swap(mImageArgumentInfo, other.mImageArgumentInfo);
        swap(mBufferArgumentInfo, other.mBufferArgumentInfo);
        swap(mArgumentDescriptorWrites, other.mArgumentDescriptorWrites);
    }

    void kernel_invocation::addStorageBufferArgument(vulkan_utils::storage_buffer& buffer) {
        // TODO: fix storage_buffer so it doesn't need to advertise buf as a public member
        // TODO: add memory barrier support to storage_buffer
        vk::DescriptorBufferInfo bufferInfo;
        bufferInfo.setRange(VK_WHOLE_SIZE)
                .setBuffer(*buffer.buf);
        mBufferArgumentInfo.push_back(bufferInfo);

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(mArgumentDescriptorWrites.size())
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer) {
        // TODO: fix uniform_buffer so it doesn't need to advertise buf as a public member
        // TODO: add memory barrier support to uniform_buffer
        vk::DescriptorBufferInfo bufferInfo;
        bufferInfo.setRange(VK_WHOLE_SIZE)
                .setBuffer(*buffer.buf);
        mBufferArgumentInfo.push_back(bufferInfo);

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(mArgumentDescriptorWrites.size())
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
                .setDstBinding(mArgumentDescriptorWrites.size())
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampler);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addReadOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eShaderReadOnlyOptimal));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(mArgumentDescriptorWrites.size())
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampledImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addWriteOnlyImageArgument(vulkan_utils::image& image) {
        mImageMemoryBarriers.push_back(image.prepare(vk::ImageLayout::eGeneral));
        mImageArgumentInfo.push_back(image.use());

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(mArgumentDescriptorSet)
                .setDstBinding(mArgumentDescriptorWrites.size())
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageImage);
        mArgumentDescriptorWrites.push_back(argSet);
    }

    void kernel_invocation::addLocalArraySizeArgument(unsigned int numElements) {
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
        mDevice.mDevice.updateDescriptorSets(writeSets, nullptr);
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
                                  { { vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead } },    // memory barriers
                                  nullptr,    // buffer memory barriers
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

        mDevice.mComputeQueue.submit(submitInfo, nullptr);

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
        mDevice.mComputeQueue.waitIdle();
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t timestamps[kQueryIndex_Count];
        mDevice.mDevice.getQueryPoolResults(*mQueryPool,
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

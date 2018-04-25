//
// Created by Eric Berdahl on 10/22/17.
//

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include "clspv_utils.hpp"
#include "util.hpp"

namespace clspv_utils {

    namespace {
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

        vk::UniqueCommandBuffer allocate_command_buffer(vk::Device device, vk::CommandPool cmd_pool) {
            vk::CommandBufferAllocateInfo allocInfo;
            allocInfo.setCommandPool(cmd_pool)
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(1);

            auto buffers = device.allocateCommandBuffersUnique(allocInfo);
            assert(buffers.size() == 1);
            return std::move(buffers[0]);
        }

        std::vector<vk::UniqueDescriptorSet> allocate_descriptor_sets(
                vk::Device                                      device,
                vk::DescriptorPool                              pool,
                vk::ArrayProxy<vk::UniqueDescriptorSetLayout>   layouts)
        {
            std::vector<vk::DescriptorSetLayout> rawLayouts = vulkan_utils::extractUniques(layouts);

            vk::DescriptorSetAllocateInfo createInfo;
            createInfo.setDescriptorPool(pool)
                    .setDescriptorSetCount(rawLayouts.size())
                    .setPSetLayouts(rawLayouts.size() ? rawLayouts.data() : nullptr);

            return device.allocateDescriptorSetsUnique(createInfo);
        }

        vk::UniqueDescriptorSetLayout create_descriptor_set_layout(
                vk::Device                          device,
                vk::ArrayProxy<vk::DescriptorType>  descriptorTypes)
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

        std::vector<vk::UniqueDescriptorSetLayout> create_descriptor_layouts(vk::Device device,
                                                                const details::spv_map& spvMap,
                                                                const std::string&      entryPoint) {
            const auto kernel = spvMap.findKernel(entryPoint);
            if (!kernel) {
                throw std::runtime_error("entryPoint not found; cannot create descriptor layout");
            }

            std::vector<vk::UniqueDescriptorSetLayout> result;
            std::vector<vk::DescriptorType> descriptorTypes;

            if (!spvMap.samplers.empty()) {
                assert(0 == spvMap.samplers_desc_set);

                descriptorTypes.clear();
                descriptorTypes.resize(spvMap.samplers.size(), vk::DescriptorType::eSampler);
                result.push_back(create_descriptor_set_layout(device, descriptorTypes));
            }

            assert(kernel->descriptor_set == (spvMap.samplers.empty() ? 0 : 1));

            descriptorTypes.clear();

            // If the caller has asked only for a pipeline layout for a single entry point,
            // create empty descriptor layouts for all argument descriptors other than the
            // one used by the requested entry point.
            for (auto &ka : kernel->args) {
                // ignore any argument not in offset 0
                if (0 != ka.offset) continue;

                descriptorTypes.push_back(find_descriptor_type(ka.kind));
            }

            result.push_back(create_descriptor_set_layout(device, descriptorTypes));

            return result;
        }

        vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                      device,
                                                        vk::ArrayProxy<vk::UniqueDescriptorSetLayout>   descriptors) {
            std::vector<vk::DescriptorSetLayout> rawLayouts = vulkan_utils::extractUniques(descriptors);

            vk::PipelineLayoutCreateInfo createInfo;
            createInfo.setSetLayoutCount(descriptors.size())
                    .setPSetLayouts(rawLayouts.size() ? rawLayouts.data() : nullptr);

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
    } // anonymous namespace

    namespace details {

        spv_map spv_map::parse(std::istream &in) {
            spv_map result;

            while (!in.eof()) {
                // read one line
                std::string line;
                std::getline(in, line);

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

        void pipeline::reset() {
            mPipeline.reset();

            mArgumentsDescriptor = nullptr;
            mLiteralSamplerDescriptor = nullptr;
            mDescriptors.clear();

            mPipelineLayout.reset();
            mDescriptorLayouts.clear();
        }

    } // namespace details

    execution_time_t::execution_time_t() :
            cpu_duration(0),
            timestamps()
    {
    }

    kernel_module::kernel_module(vk::Device         device,
                                 vk::DescriptorPool pool,
                                 const std::string& moduleName) :
            mName(moduleName),
            mDevice(device),
            mDescriptorPool(pool),
            mShaderModule(),
            mSpvMap() {
        const std::string spvFilename = moduleName + ".spv";
        mShaderModule = create_shader(mDevice, spvFilename.c_str());

        const std::string mapFilename = moduleName + ".spvmap";
        mSpvMap = create_spv_map(mapFilename.c_str());
    }

    kernel_module::~kernel_module() {
    }

    std::vector<std::string> kernel_module::getEntryPoints() const {
        std::vector<std::string> result;

        std::transform(mSpvMap.kernels.begin(), mSpvMap.kernels.end(),
                       std::back_inserter(result),
                       [](const details::spv_map::kernel& k) { return k.name; });

        return result;
    }

    details::pipeline kernel_module::createPipeline(const std::string&          entryPoint,
                                                    const WorkgroupDimensions&  workGroupSizes_a,
                                                    vk::ArrayProxy<int32_t>     otherSpecConstants) const {
        details::pipeline result;

        result.mDescriptorLayouts = create_descriptor_layouts(mDevice, mSpvMap, entryPoint);

        result.mPipelineLayout = create_pipeline_layout(mDevice, result.mDescriptorLayouts);
        result.mDescriptors = allocate_descriptor_sets(mDevice, mDescriptorPool,
                                                       result.mDescriptorLayouts);


        if (-1 != mSpvMap.samplers_desc_set) {
            result.mLiteralSamplerDescriptor = *result.mDescriptors[mSpvMap.samplers_desc_set];
        }

        const auto kernel_arg_map = mSpvMap.findKernel(entryPoint);
        if (kernel_arg_map && -1 != kernel_arg_map->descriptor_set) {
            result.mArgumentsDescriptor = *result.mDescriptors[kernel_arg_map->descriptor_set];
        }

        std::vector<int32_t> specConstants = {
                workGroupSizes_a.x,
                workGroupSizes_a.y,
                1
        };
        std::copy(otherSpecConstants.begin(), otherSpecConstants.end(), std::back_inserter(specConstants));

        std::vector<vk::SpecializationMapEntry> specializationEntries;
        uint32_t index = 0;
        std::generate_n(std::back_inserter(specializationEntries),
                        specConstants.size(),
                        [&index] () {
                            const uint32_t current = index++;
                            return vk::SpecializationMapEntry(current, current * sizeof(int32_t), sizeof(int32_t));
                        });
        vk::SpecializationInfo specializationInfo;
        specializationInfo.setMapEntryCount(specConstants.size())
                .setPMapEntries(specializationEntries.data())
                .setDataSize(specConstants.size() * sizeof(int32_t))
                .setPData(specConstants.data());

        vk::ComputePipelineCreateInfo createInfo;
        createInfo.setLayout(*result.mPipelineLayout)
                .stage.setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(*mShaderModule)
                .setPName(entryPoint.c_str())
                .setPSpecializationInfo(&specializationInfo);

        result.mPipeline = mDevice.createComputePipelineUnique(vk::PipelineCache(), createInfo);

        return result;
    }

    kernel::kernel(const kernel_module&         module,
                   std::string                  entryPoint,
                   const WorkgroupDimensions&   workgroup_sizes) :
            mModule(&module),
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mPipeline() {
        updatePipeline(nullptr);
    }

    kernel::~kernel() {
    }

    void kernel::updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants) const {
        mPipeline = mModule->createPipeline(mEntryPoint, mWorkgroupSizes, otherSpecConstants);
    }

    void kernel::bindCommand(vk::CommandBuffer command) const {
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *mPipeline.mPipeline);

        auto regular = vulkan_utils::extractUniques(mPipeline.mDescriptors);

        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                *mPipeline.mPipelineLayout,
                                0,
                                regular,
                                nullptr);
    }

    kernel_invocation::kernel_invocation(vk::Device                                 device,
                                         vk::CommandPool                            cmdPool,
                                         const vk::PhysicalDeviceMemoryProperties&  memoryProperties) :
            mDevice(device),
            mCommand(),
            mMemoryProperties(memoryProperties),
            mLiteralSamplers(),
            mDescriptorArguments(),
            mSpecConstantArguments()
    {
        mCommand = allocate_command_buffer(mDevice, cmdPool);

        vk::QueryPoolCreateInfo poolCreateInfo;
        poolCreateInfo.setQueryType(vk::QueryType::eTimestamp)
                .setQueryCount(kQueryIndex_Count);

        mQueryPool = device.createQueryPoolUnique(poolCreateInfo);
    }

    kernel_invocation::~kernel_invocation() {
    }

    void kernel_invocation::addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers) {
        mLiteralSamplers.insert(mLiteralSamplers.end(), samplers.begin(), samplers.end());
    }

    void kernel_invocation::addStorageBufferArgument(vk::Buffer buf) {
        arg item;

        item.type = vk::DescriptorType::eStorageBuffer;
        item.buffer = buf;

        mDescriptorArguments.push_back(item);
    }

    void kernel_invocation::addUniformBufferArgument(vk::Buffer buf) {
        arg item;

        item.type = vk::DescriptorType::eUniformBuffer;
        item.buffer = buf;

        mDescriptorArguments.push_back(item);
    }

    void kernel_invocation::addSamplerArgument(vk::Sampler samp) {
        arg item;

        item.type = vk::DescriptorType::eSampler;
        item.sampler = samp;

        mDescriptorArguments.push_back(item);
    }

    void kernel_invocation::addReadOnlyImageArgument(vk::ImageView im) {
        arg item;

        item.type = vk::DescriptorType::eSampledImage;
        item.image = im;

        mDescriptorArguments.push_back(item);
    }

    void kernel_invocation::addWriteOnlyImageArgument(vk::ImageView im) {
        arg item;

        item.type = vk::DescriptorType::eStorageImage;
        item.image = im;

        mDescriptorArguments.push_back(item);
    }

    void kernel_invocation::addLocalArraySizeArgument(unsigned int numElements) {
        mSpecConstantArguments.push_back(numElements);
    }

    void kernel_invocation::addPodArgument(const void* pod, std::size_t sizeofPod) {
        vulkan_utils::uniform_buffer scalar_args(mDevice, mMemoryProperties, sizeofPod);

        vulkan_utils::copyToDeviceMemory(scalar_args.mem, pod, sizeofPod);
        addUniformBufferArgument(*scalar_args.buf);

        mPodBuffers.push_back(std::move(scalar_args));
    }

    void kernel_invocation::updateDescriptorSets(vk::DescriptorSet literalSamplerDescSet,
                                                 vk::DescriptorSet argumentDescSet) {
        std::vector<vk::DescriptorImageInfo>    imageList;
        std::vector<vk::DescriptorBufferInfo>   bufferList;

        //
        // Collect information about the literal samplers
        //
        for (auto s : mLiteralSamplers) {
            vk::DescriptorImageInfo samplerInfo;
            samplerInfo.setSampler(s);

            imageList.push_back(samplerInfo);
        }

        //
        // Collect information about the arguments
        //
        for (auto& a : mDescriptorArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage: {
                    vk::DescriptorImageInfo imageInfo;
                    imageInfo.setImageView(a.image)
                            .setImageLayout(vk::ImageLayout::eGeneral);

                    imageList.push_back(imageInfo);
                    break;
                }

                case vk::DescriptorType::eStorageBuffer:
                case vk::DescriptorType::eUniformBuffer: {
                    vk::DescriptorBufferInfo bufferInfo;
                    bufferInfo.setRange(VK_WHOLE_SIZE)
                            .setBuffer(a.buffer);

                    bufferList.push_back(bufferInfo);
                    break;
                }

                case vk::DescriptorType::eSampler: {
                    vk::DescriptorImageInfo samplerInfo;
                    samplerInfo.setSampler(a.sampler);

                    imageList.push_back(samplerInfo);
                    break;
                }

                default:
                    assert(0 && "unkown argument type");
            }
        }

        //
        // Set up to create the descriptor set write structures
        // We will iterate the param lists in the same order,
        // picking up image and buffer infos in order.
        //

        std::vector<vk::WriteDescriptorSet> writeSets;
        auto nextImage = imageList.begin();
        auto nextBuffer = bufferList.begin();

        //
        // Update the literal samplers' descriptor set
        //

        vk::WriteDescriptorSet literalSamplerSet;
        literalSamplerSet.setDstSet(literalSamplerDescSet)
                .setDstBinding(0)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eSampler);

        assert(mLiteralSamplers.empty() || literalSamplerDescSet);

        for (auto s : mLiteralSamplers) {
            literalSamplerSet.setPImageInfo(&(*nextImage));
            ++nextImage;

            writeSets.push_back(literalSamplerSet);

            ++literalSamplerSet.dstBinding;
        }

        //
        // Update the kernel's argument descriptor set
        //

        vk::WriteDescriptorSet argSet;
        argSet.setDstSet(argumentDescSet)
                .setDstBinding(0)
                .setDescriptorCount(1);

        assert(mDescriptorArguments.empty() || argumentDescSet);

        for (auto& a : mDescriptorArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage:
                    argSet.setDescriptorType(a.type)
                            .setPImageInfo(&(*nextImage));
                    ++nextImage;
                    break;

                case vk::DescriptorType::eUniformBuffer:
                case vk::DescriptorType::eStorageBuffer:
                    argSet.setDescriptorType(a.type)
                            .setPBufferInfo(&(*nextBuffer));
                    ++nextBuffer;
                    break;

                case vk::DescriptorType::eSampler:
                    argSet.setDescriptorType(vk::DescriptorType::eSampler)
                            .setPImageInfo(&(*nextImage));
                    ++nextImage;
                    break;

                default:
                    assert(0 && "unkown argument type");
            }

            writeSets.push_back(argSet);

            ++argSet.dstBinding;
        }

        //
        // Do the actual descriptor set updates
        //
        mDevice.updateDescriptorSets(writeSets, nullptr);
    }

    void kernel_invocation::fillCommandBuffer(const kernel&                 inKernel,
                                              const WorkgroupDimensions&    num_workgroups) {
        mCommand->begin(vk::CommandBufferBeginInfo());

        inKernel.bindCommand(*mCommand);

        mCommand->resetQueryPool(*mQueryPool, kQueryIndex_FirstIndex, kQueryIndex_Count);

        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_StartOfExecution);
        mCommand->pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  vk::DependencyFlags(),
                                  { { vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead } },    // memory barriers
                                  nullptr,    // buffer memory barriers
                                  nullptr);    // image memory barriers
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostHostBarrier);
        mCommand->dispatch(num_workgroups.x, num_workgroups.y, 1);
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostExecution);
        mCommand->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eHost,
                                  vk::DependencyFlags(),
                                  { { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead} },    // memory barriers
                                  nullptr,    // buffer memory barriers
                                  nullptr);    // image memory barriers
        mCommand->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *mQueryPool, kQueryIndex_PostGPUBarrier);

        mCommand->end();
    }

    void kernel_invocation::submitCommand(vk::Queue queue) {
        vk::CommandBuffer rawCommand = *mCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        queue.submit(submitInfo, nullptr);

    }

    execution_time_t kernel_invocation::run(vk::Queue                   queue,
                                            const kernel&               kern,
                                            const WorkgroupDimensions&  num_workgroups) {
        // HACK re-create the pipeline if the invocation includes spec constant arguments.
        // TODO factor the pipeline recreation better, possibly along with an overhaul of kernel
        // management
        if (!mSpecConstantArguments.empty()) {
            kern.updatePipeline(mSpecConstantArguments);
        }

        updateDescriptorSets(kern.getLiteralSamplerDescSet(), kern.getArgumentDescSet());
        fillCommandBuffer(kern, num_workgroups);

        auto start = std::chrono::high_resolution_clock::now();
        submitCommand(queue);
        queue.waitIdle();
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t timestamps[kQueryIndex_Count];
        mDevice.getQueryPoolResults(*mQueryPool,
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

//
// Created by Eric Berdahl on 10/22/17.
//

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include "clspv_utils.hpp"
#include "util.hpp"

namespace clspv_utils {

    namespace {

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
                vk::Device                                          device,
                vk::DescriptorPool                                  pool,
                const std::vector<vk::UniqueDescriptorSetLayout>&   layouts) {
            std::vector<vk::DescriptorSetLayout> rawLayouts = vulkan_utils::extractUniques(layouts);

            vk::DescriptorSetAllocateInfo createInfo;
            createInfo.setDescriptorPool(pool)
                    .setDescriptorSetCount(rawLayouts.size())
                    .setPSetLayouts(rawLayouts.size() ? rawLayouts.data() : nullptr);

            return device.allocateDescriptorSetsUnique(createInfo);
        }

        vk::UniqueDescriptorSetLayout create_descriptor_set_layout(
                vk::Device                              device,
                const std::vector<vk::DescriptorType>&  descriptorTypes) {
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
            assert(!entryPoint.empty());

            std::vector<vk::UniqueDescriptorSetLayout> result;
            std::vector<vk::DescriptorType> descriptorTypes;

            if (!spvMap.samplers.empty()) {
                assert(0 == spvMap.samplers_desc_set);

                descriptorTypes.clear();
                descriptorTypes.resize(spvMap.samplers.size(), vk::DescriptorType::eSampler);
                result.push_back(create_descriptor_set_layout(device, descriptorTypes));
            }

            const auto kernel = spvMap.findKernel(entryPoint);
            if (kernel) {
                assert(kernel->descriptor_set == (spvMap.samplers.empty() ? 0 : 1));

                descriptorTypes.clear();

                // If the caller has asked only for a pipeline layout for a single entry point,
                // create empty descriptor layouts for all argument descriptors other than the
                // one used by the requested entry point.
                for (auto &ka : kernel->args) {
                    // ignore any argument not in offset 0
                    if (0 != ka.offset) continue;

                    vk::DescriptorType argType;

                    switch (ka.kind) {
                        case details::spv_map::arg::kind_pod:
                        case details::spv_map::arg::kind_buffer:
                            argType = vk::DescriptorType::eStorageBuffer;
                            break;

                        case details::spv_map::arg::kind_ro_image:
                            argType = vk::DescriptorType::eSampledImage;
                            break;

                        case details::spv_map::arg::kind_wo_image:
                            argType = vk::DescriptorType::eStorageImage;
                            break;

                        case details::spv_map::arg::kind_sampler:
                            argType = vk::DescriptorType::eSampler;
                            break;

                        default:
                            assert(0 && "unkown argument type");
                    }

                    descriptorTypes.push_back(argType);
                }

                result.push_back(create_descriptor_set_layout(device, descriptorTypes));
            };

            return result;
        }

        vk::UniquePipelineLayout create_pipeline_layout(vk::Device                                          device,
                                                        const std::vector<vk::UniqueDescriptorSetLayout>&   descriptors) {
            std::vector<vk::DescriptorSetLayout> rawLayouts = vulkan_utils::extractUniques(descriptors);

            vk::PipelineLayoutCreateInfo createInfo;
            createInfo.setSetLayoutCount(descriptors.size())
                    .setPSetLayouts(rawLayouts.size() ? rawLayouts.data() : nullptr);

            return device.createPipelineLayoutUnique(createInfo);
        }

    } // anonymous namespace

    namespace details {

        spv_map::arg::kind_t spv_map::parse_argType(const std::string &argType) {
            arg::kind_t result = arg::kind_unknown;

            if (argType == "pod") {
                result = arg::kind_pod;
            } else if (argType == "buffer") {
                result = arg::kind_buffer;
            } else if (argType == "ro_image") {
                result = arg::kind_ro_image;
            } else if (argType == "wo_image") {
                result = arg::kind_wo_image;
            } else if (argType == "sampler") {
                result = arg::kind_sampler;
            } else {
                assert(0 && "unknown spvmap arg type");
            }

            return result;
        }

        spv_map spv_map::parse(std::istream &in) {
            spv_map result;

            while (!in.eof()) {
                // read one line
                std::string line;
                std::getline(in, line);

                std::istringstream in_line(line);
                std::string key = read_csv_field(in_line);
                std::string value = read_csv_field(in_line);
                if ("sampler" == key) {
                    auto s = result.samplers.insert(result.samplers.end(), spv_map::sampler());
                    assert(s != result.samplers.end());

                    s->opencl_flags = std::atoi(value.c_str());

                    while (!in_line.eof()) {
                        key = read_csv_field(in_line);
                        value = read_csv_field(in_line);

                        if ("descriptorSet" == key) {
                            // all samplers, if any, are documented to share descriptor set 0
                            const int ds = std::atoi(value.c_str());
                            assert(ds == 0);

                            if (-1 == result.samplers_desc_set) {
                                result.samplers_desc_set = ds;
                            }
                        } else if ("binding" == key) {
                            s->binding = std::atoi(value.c_str());
                        }
                    }
                } else if ("kernel" == key) {
                    auto kernel = result.findKernel(value);
                    if (!kernel) {
                        result.kernels.push_back(spv_map::kernel());
                        kernel = &result.kernels.back();
                        kernel->name = value;
                    }
                    assert(kernel);

                    auto ka = kernel->args.end();

                    while (!in_line.eof()) {
                        key = read_csv_field(in_line);
                        value = read_csv_field(in_line);

                        if ("argOrdinal" == key) {
                            assert(ka == kernel->args.end());

                            const int arg_index = std::atoi(value.c_str());

                            if (kernel->args.size() <= arg_index) {
                                kernel->args.resize(arg_index + 1, spv_map::arg());
                            }
                            ka = std::next(kernel->args.begin(), arg_index);

                            assert(ka != kernel->args.end());
                        } else if ("descriptorSet" == key) {
                            const int ds = std::atoi(value.c_str());
                            if (-1 == kernel->descriptor_set) {
                                kernel->descriptor_set = ds;
                            }

                            // all args for a kernel are documented to share the same descriptor set
                            assert(ds == kernel->descriptor_set);
                        } else if ("binding" == key) {
                            ka->binding = std::atoi(value.c_str());
                        } else if ("offset" == key) {
                            ka->offset = std::atoi(value.c_str());
                        } else if ("argKind" == key) {
                            ka->kind = parse_argType(value);
                        }
                    }
                }
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

    kernel_module::kernel_module(VkDevice           device,
                                 VkDescriptorPool   pool,
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

    details::pipeline kernel_module::createPipeline(const std::string&         entryPoint,
                                                    const WorkgroupDimensions& work_group_sizes) const {
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

        const unsigned int num_workgroup_sizes = 3;
        const int32_t workGroupSizes[num_workgroup_sizes] = {
                work_group_sizes.x,
                work_group_sizes.y,
                1
        };
        const vk::SpecializationMapEntry specializationEntries[num_workgroup_sizes] = {
                {
                        0,                          // specialization constant 0 - workgroup size X
                        0 * sizeof(int32_t),          // offset - start of workGroupSizes array
                        sizeof(workGroupSizes[0])   // sizeof the first element
                },
                {
                        1,                          // specialization constant 1 - workgroup size Y
                        1 * sizeof(int32_t),            // offset - one element into the array
                        sizeof(workGroupSizes[1])   // sizeof the second element
                },
                {
                        2,                          // specialization constant 2 - workgroup size Z
                        2 * sizeof(int32_t),          // offset - two elements into the array
                        sizeof(workGroupSizes[2])   // sizeof the second element
                }
        };
        vk::SpecializationInfo specializationInfo;
        specializationInfo.setMapEntryCount(num_workgroup_sizes)
                .setPMapEntries(specializationEntries)
                .setDataSize(sizeof(workGroupSizes))
                .setPData(workGroupSizes);

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
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mPipeline() {
        mPipeline = module.createPipeline(entryPoint, workgroup_sizes);
    }

    kernel::~kernel() {
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
            mArguments() {
        mCommand = allocate_command_buffer(mDevice, cmdPool);
    }

    kernel_invocation::~kernel_invocation() {
        std::for_each(mPodBuffers.begin(), mPodBuffers.end(), std::mem_fun_ref(&vulkan_utils::buffer::reset));
    }

    void kernel_invocation::addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers) {
        mLiteralSamplers.insert(mLiteralSamplers.end(), samplers.begin(), samplers.end());
    }

    void kernel_invocation::addBufferArgument(VkBuffer buf) {
        arg item;

        item.type = vk::DescriptorType::eStorageBuffer;
        item.buffer = vk::Buffer(buf);

        mArguments.push_back(item);
    }

    void kernel_invocation::addSamplerArgument(VkSampler samp) {
        arg item;

        item.type = vk::DescriptorType::eSampler;
        item.sampler = vk::Sampler(samp);

        mArguments.push_back(item);
    }

    void kernel_invocation::addReadOnlyImageArgument(VkImageView im) {
        arg item;

        item.type = vk::DescriptorType::eSampledImage;
        item.image = vk::ImageView(im);

        mArguments.push_back(item);
    }

    void kernel_invocation::addWriteOnlyImageArgument(VkImageView im) {
        arg item;

        item.type = vk::DescriptorType::eStorageImage;
        item.image = vk::ImageView(im);

        mArguments.push_back(item);
    }

    void kernel_invocation::addPodArgument(const void* pod, std::size_t sizeofPod) {
        vulkan_utils::buffer scalar_args((VkDevice) mDevice, mMemoryProperties, sizeofPod);
        mPodBuffers.push_back(scalar_args);

        {
            vulkan_utils::memory_map scalar_map(scalar_args);
            memcpy(scalar_map.data, pod, sizeofPod);
        }

        addBufferArgument(scalar_args.buf);
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
        for (auto& a : mArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage: {
                    vk::DescriptorImageInfo imageInfo;
                    imageInfo.setImageView(a.image)
                            .setImageLayout(vk::ImageLayout::eGeneral);

                    imageList.push_back(imageInfo);
                    break;
                }

                case vk::DescriptorType::eStorageBuffer: {
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

        assert(mArguments.empty() || argumentDescSet);

        for (auto& a : mArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage:
                    argSet.setDescriptorType(a.type)
                            .setPImageInfo(&(*nextImage));
                    ++nextImage;
                    break;

                case vk::DescriptorType::eStorageBuffer:
                    argSet.setDescriptorType(vk::DescriptorType::eStorageBuffer)
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

        mCommand->dispatch(num_workgroups.x, num_workgroups.y, 1);
        mCommand->end();
    }

    void kernel_invocation::submitCommand(vk::Queue queue) {
        vk::CommandBuffer rawCommand = *mCommand;
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBufferCount(1)
                .setPCommandBuffers(&rawCommand);

        queue.submit(submitInfo, nullptr);

    }

    void kernel_invocation::run(vk::Queue                   queue,
                                const kernel&               kern,
                                const WorkgroupDimensions&  num_workgroups) {
        updateDescriptorSets(kern.getLiteralSamplerDescSet(), kern.getArgumentDescSet());
        fillCommandBuffer(kern, num_workgroups);
        submitCommand(queue);

        queue.waitIdle();
    }

} // namespace clspv_utils

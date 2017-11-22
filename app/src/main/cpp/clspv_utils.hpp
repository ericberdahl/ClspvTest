//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPV_UTILS_HPP
#define CLSPV_UTILS_HPP

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulkan_utils.hpp"

namespace clspv_utils {

    namespace details {
        struct spv_map {
            struct sampler {
                sampler() : opencl_flags(0), binding(-1) {};

                int opencl_flags;
                int binding;
            };

            struct arg {
                enum kind_t {
                    kind_unknown, kind_pod, kind_buffer, kind_ro_image, kind_wo_image, kind_sampler
                };

                arg() : kind(kind_unknown), binding(-1), offset(0) {};

                kind_t kind;
                int binding;
                int offset;
            };

            struct kernel {
                kernel() : name(), descriptor_set(-1), args() {};

                std::string name;
                int descriptor_set;
                std::vector<arg> args;
            };

            static arg::kind_t parse_argType(const std::string &argType);

            static spv_map parse(std::istream &in);

            spv_map() : samplers(), samplers_desc_set(-1), kernels() {};

            kernel* findKernel(const std::string& name);
            const kernel* findKernel(const std::string& name) const;

            std::vector<sampler> samplers;
            int samplers_desc_set;
            std::vector<kernel> kernels;
        };

        struct pipeline {
            void    reset();

            std::vector<vk::UniqueDescriptorSetLayout>  mDescriptorLayouts;
            vk::UniquePipelineLayout                    mPipelineLayout;
            std::vector<vk::UniqueDescriptorSet>        mDescriptors;
            vk::DescriptorSet                           mLiteralSamplerDescriptor;
            vk::DescriptorSet                           mArgumentsDescriptor;
            vk::UniquePipeline                          mPipeline;
        };
    } // namespace details

    struct WorkgroupDimensions {
        WorkgroupDimensions(int xDim = 1, int yDim = 1) : x(xDim), y(yDim) {}

        int x;
        int y;
    };

    class kernel_module {
    public:
        kernel_module(VkDevice              device,
                      VkDescriptorPool      pool,
                      const std::string&    moduleName);

        ~kernel_module();

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;

        details::pipeline           createPipeline(const std::string&           entryPoint,
                                                   const WorkgroupDimensions&   work_group_sizes) const;

    private:
        std::string                         mName;
        vk::Device                          mDevice;
        vk::DescriptorPool                  mDescriptorPool;
        vk::UniqueShaderModule              mShaderModule;
        details::spv_map                    mSpvMap;
    };

    class kernel {
    public:
        kernel(VkDevice                     device,
               const kernel_module&         module,
               std::string                  entryPoint,
               const WorkgroupDimensions&   workgroup_sizes);

        ~kernel();

        void bindCommand(VkCommandBuffer command) const;

        std::string getEntryPoint() const { return mEntryPoint; }
        WorkgroupDimensions getWorkgroupSize() const { return mWorkgroupSizes; }

        VkDescriptorSet getLiteralSamplerDescSet() const { return (VkDescriptorSet) mPipeline.mLiteralSamplerDescriptor; }
        VkDescriptorSet getArgumentDescSet() const { return (VkDescriptorSet) mPipeline.mArgumentsDescriptor; }

    private:
        std::string         mEntryPoint;
        WorkgroupDimensions mWorkgroupSizes;
        details::pipeline   mPipeline;
    };

    class kernel_invocation {
    public:
        kernel_invocation(VkDevice              device,
                          VkCommandPool         cmdPool,
                          const VkPhysicalDeviceMemoryProperties&   memoryProperties);

        ~kernel_invocation();

        template <typename Iterator>
        void    addLiteralSamplers(Iterator first, Iterator last);

        void    addBufferArgument(VkBuffer buf);
        void    addReadOnlyImageArgument(VkImageView image);
        void    addWriteOnlyImageArgument(VkImageView image);
        void    addSamplerArgument(VkSampler samp);

        template <typename T>
        void    addPodArgument(const T& pod);

        void    run(VkQueue                     queue,
                    const kernel&               kern,
                    const WorkgroupDimensions&  num_workgroups);

    private:
        void        fillCommandBuffer(const kernel&                 kern,
                                      const WorkgroupDimensions&    num_workgroups);
        void        updateDescriptorSets(VkDescriptorSet literalSamplerSet,
                                         VkDescriptorSet argumentSet);
        void        submitCommand(VkQueue queue);

    private:
        struct arg {
            vk::DescriptorType  type;
            vk::Buffer          buffer;
            vk::Sampler         sampler;
            vk::ImageView       image;
        };

    private:
        VkDevice                            mDevice;
        VkCommandPool                       mCmdPool;
        VkCommandBuffer                     mCommand;
        VkPhysicalDeviceMemoryProperties    mMemoryProperties;

        std::vector<VkSampler>              mLiteralSamplers;
        std::vector<arg>                    mArguments;
        std::vector<vulkan_utils::buffer>   mPodBuffers;
    };

    template <typename Iterator>
    void kernel_invocation::addLiteralSamplers(Iterator first, Iterator last) {
        mLiteralSamplers.insert(mLiteralSamplers.end(), first, last);
    }

    template <typename T>
    void kernel_invocation::addPodArgument(const T& pod) {
        vulkan_utils::buffer scalar_args(mDevice, mMemoryProperties, sizeof(T));
        mPodBuffers.push_back(scalar_args);

        {
            vulkan_utils::memory_map scalar_map(scalar_args);
            memcpy(scalar_map.data, &pod, sizeof(T));
        }

        addBufferArgument(scalar_args.buf);
    }
}

#endif //CLSPV_UTILS_HPP

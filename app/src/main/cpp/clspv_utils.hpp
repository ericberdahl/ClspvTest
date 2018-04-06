//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPV_UTILS_HPP
#define CLSPV_UTILS_HPP

#include <chrono>
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
                    kind_unknown, kind_pod, kind_pod_ubo, kind_buffer, kind_ro_image, kind_wo_image, kind_sampler
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
        kernel_module(vk::Device            device,
                      vk::DescriptorPool    pool,
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
        kernel(const kernel_module&         module,
               std::string                  entryPoint,
               const WorkgroupDimensions&   workgroup_sizes);

        ~kernel();

        void bindCommand(vk::CommandBuffer command) const;

        std::string getEntryPoint() const { return mEntryPoint; }
        WorkgroupDimensions getWorkgroupSize() const { return mWorkgroupSizes; }

        vk::DescriptorSet getLiteralSamplerDescSet() const { return mPipeline.mLiteralSamplerDescriptor; }
        vk::DescriptorSet getArgumentDescSet() const { return mPipeline.mArgumentsDescriptor; }

    private:
        std::string         mEntryPoint;
        WorkgroupDimensions mWorkgroupSizes;
        details::pipeline   mPipeline;
    };

    class kernel_invocation {
    public:
        kernel_invocation(vk::Device                                device,
                          vk::CommandPool                           cmdPool,
                          const vk::PhysicalDeviceMemoryProperties& memoryProperties);

        ~kernel_invocation();

        void    addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers);

        void    addStorageBufferArgument(vk::Buffer buf);
        void    addUniformBufferArgument(vk::Buffer buf);
        void    addReadOnlyImageArgument(vk::ImageView image);
        void    addWriteOnlyImageArgument(vk::ImageView image);
        void    addSamplerArgument(vk::Sampler samp);

        void    addPodArgument(const void* pod, std::size_t sizeofPod);

        template <typename T>
        void    addPodArgument(const T& pod);

        typedef std::chrono::duration<double> execution_time_t;

        execution_time_t    run(vk::Queue                   queue,
                                const kernel&               kern,
                                const WorkgroupDimensions&  num_workgroups);

    private:
        void        fillCommandBuffer(const kernel&                 kern,
                                      const WorkgroupDimensions&    num_workgroups);
        void        updateDescriptorSets(vk::DescriptorSet literalSamplerSet,
                                         vk::DescriptorSet argumentSet);
        void        submitCommand(vk::Queue queue);

    private:
        struct arg {
            vk::DescriptorType  type;
            vk::Buffer          buffer;
            vk::Sampler         sampler;
            vk::ImageView       image;
        };

    private:
        vk::Device                                  mDevice;
        vk::UniqueCommandBuffer                     mCommand;
        vk::PhysicalDeviceMemoryProperties          mMemoryProperties;

        std::vector<vk::Sampler>                    mLiteralSamplers;
        std::vector<arg>                            mArguments;
        std::vector<vulkan_utils::uniform_buffer>   mPodBuffers;
    };

    template <typename T>
    inline void kernel_invocation::addPodArgument(const T& pod) {
        addPodArgument(&pod, sizeof(pod));
    }
}

#endif //CLSPV_UTILS_HPP

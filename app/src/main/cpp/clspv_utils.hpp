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
                int opencl_flags    = 0;
                int descriptor_set  = -1;
                int binding         = -1;
            };

            struct arg {
                enum kind_t {
                    kind_unknown,
                    kind_pod,
                    kind_pod_ubo,
                    kind_buffer,
                    kind_ro_image,
                    kind_wo_image,
                    kind_sampler,
                    kind_local
                };

                kind_t  kind            = kind_unknown;
                int     ordinal         = -1;
                int     descriptor_set  = -1;
                int     binding         = -1;
                int     offset          = -1;
                int     spec_constant   = -1;
            };

            struct kernel {
                std::string         name;
                int                 descriptor_set  = -1;
                std::vector<arg>    args;
            };

            static spv_map parse(std::istream &in);

            kernel* findKernel(const std::string& name);
            const kernel* findKernel(const std::string& name) const;

            std::vector<sampler>    samplers;
            int                     samplers_desc_set   = -1;
            std::vector<kernel>     kernels;
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

    struct execution_time_t {
        struct vulkan_timestamps_t {
            uint64_t start          = 0;
            uint64_t host_barrier   = 0;
            uint64_t execution      = 0;
            uint64_t gpu_barrier    = 0;
        };

        execution_time_t();

        std::chrono::duration<double>   cpu_duration;
        vulkan_timestamps_t             timestamps;
    };

    class kernel_module {
    public:
        kernel_module(vk::Device            device,
                      vk::DescriptorPool    pool,
                      const std::string&    moduleName);

        ~kernel_module();

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;
        vk::Device                  getDevice() const { return mDevice; }

        details::pipeline           createPipeline(const std::string&           entryPoint,
                                                   const WorkgroupDimensions&   workGroupSizes,
                                                   vk::ArrayProxy<int32_t>      otherSpecConstants) const;

    private:
        std::string                         mName;
        vk::Device                          mDevice;
        vk::DescriptorPool                  mDescriptorPool;
        vk::UniqueShaderModule              mShaderModule;
        details::spv_map                    mSpvMap;
    };

    class kernel {
    public:
        kernel(kernel_module&               module,
               std::string                  entryPoint,
               const WorkgroupDimensions&   workgroup_sizes);

        ~kernel();

        void                bindCommand(vk::CommandBuffer command) const;

        std::string         getEntryPoint() const { return mEntryPoint; }
        WorkgroupDimensions getWorkgroupSize() const { return mWorkgroupSizes; }

        kernel_module&          getModule() { return mModule; }
        const kernel_module&    getModule() const { return mModule; }

        vk::DescriptorSet   getLiteralSamplerDescSet() const { return mPipeline.mLiteralSamplerDescriptor; }
        vk::DescriptorSet   getArgumentDescSet() const { return mPipeline.mArgumentsDescriptor; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

    private:
        std::reference_wrapper<kernel_module>   mModule;
        std::string                             mEntryPoint;
        WorkgroupDimensions                     mWorkgroupSizes;
        details::pipeline                       mPipeline;
    };

    class kernel_invocation {
    public:
        kernel_invocation(kernel&                                   kernel,
                          vk::CommandPool                           cmdPool,
                          const vk::PhysicalDeviceMemoryProperties& memoryProperties);

        ~kernel_invocation();

        void    addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers);

        void    addStorageBufferArgument(vk::Buffer buf);
        void    addUniformBufferArgument(vk::Buffer buf);
        void    addReadOnlyImageArgument(vk::ImageView image);
        void    addWriteOnlyImageArgument(vk::ImageView image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        void    addPodArgument(const void* pod, std::size_t sizeofPod);

        template <typename T>
        void    addPodArgument(const T& pod);

        execution_time_t    run(vk::Queue                   queue,
                                const WorkgroupDimensions&  num_workgroups);

    private:
        void        fillCommandBuffer(const WorkgroupDimensions&    num_workgroups);
        void        updateDescriptorSets();
        void        submitCommand(vk::Queue queue);

        vk::Device  getDevice() const { return mKernel.get().getModule().getDevice(); }

    private:
        enum QueryIndex {
            kQueryIndex_FirstIndex = 0,
            kQueryIndex_StartOfExecution = 0,
            kQueryIndex_PostHostBarrier = 1,
            kQueryIndex_PostExecution = 2,
            kQueryIndex_PostGPUBarrier= 3,
            kQueryIndex_Count = 4
        };

    private:
        struct arg {
            vk::DescriptorType  type;
            vk::Buffer          buffer;
            vk::Sampler         sampler;
            vk::ImageView       image;
        };

    private:
        std::reference_wrapper<kernel>              mKernel;
        vk::UniqueCommandBuffer                     mCommand;
        vk::PhysicalDeviceMemoryProperties          mMemoryProperties;
        vk::UniqueQueryPool                         mQueryPool;

        std::vector<vk::Sampler>                    mLiteralSamplers;
        std::vector<arg>                            mDescriptorArguments;
        std::vector<int32_t>                        mSpecConstantArguments;
        std::vector<vulkan_utils::uniform_buffer>   mPodBuffers;
    };

    template <typename T>
    inline void kernel_invocation::addPodArgument(const T& pod) {
        addPodArgument(&pod, sizeof(pod));
    }
}

#endif //CLSPV_UTILS_HPP

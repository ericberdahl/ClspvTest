//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPV_UTILS_HPP
#define CLSPV_UTILS_HPP

#include <chrono>
#include <map>
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

    struct layout_t {
        std::vector<vk::UniqueDescriptorSetLayout>  mDescriptorLayouts;
        vk::UniquePipelineLayout                    mPipelineLayout;
        std::vector<vk::UniqueDescriptorSet>        mDescriptors;
        vk::DescriptorSet                           mLiteralSamplerDescriptor;
        vk::DescriptorSet                           mArgumentsDescriptor;
    };

    struct device_t {
        device_t(const device_t&) = delete;
        device_t& operator=(const device_t&) = delete;

        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::DescriptorPool                  mDescriptorPool;
        vk::CommandPool                     mCommandPool;
        vk::Queue                           mComputeQueue;

        std::map<int,vk::UniqueSampler>     mSamplerCache;
    };

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags);

    class kernel_module {
    public:
        kernel_module(device_t&             device,
                      const std::string&    moduleName);

        ~kernel_module();

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;
        vk::ShaderModule            getShaderModule() const { return *mShaderModule; }

        device_t&                   getDevice() { return mDevice.get(); }
        const device_t&             getDevice() const { return mDevice.get(); }

        layout_t                    createLayout(const std::string& entryPoint) const;

        vk::ArrayProxy<const vk::Sampler>   getLiteralSamplersHack() const { return mSamplers; }

    private:
        std::reference_wrapper<device_t>    mDevice;
        std::string                         mName;
        vk::UniqueShaderModule              mShaderModule;
        details::spv_map                    mSpvMap;
        std::vector<vk::Sampler>            mSamplers;
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

        device_t&           getDevice() { return getModule().getDevice(); }
        const device_t&     getDevice() const { return getModule().getDevice(); }

        vk::DescriptorSet   getLiteralSamplerDescSet() const { return mLayout.mLiteralSamplerDescriptor; }
        vk::DescriptorSet   getArgumentDescSet() const { return mLayout.mArgumentsDescriptor; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

    private:
        std::reference_wrapper<kernel_module>   mModule;
        std::string                             mEntryPoint;
        WorkgroupDimensions                     mWorkgroupSizes;
        layout_t                                mLayout;
        vk::UniquePipeline                      mPipeline;
    };

    class kernel_invocation {
    public:
        explicit    kernel_invocation(kernel& kernel);

                    ~kernel_invocation();

        void    addStorageBufferArgument(vulkan_utils::storage_buffer& buffer);
        void    addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer);
        void    addReadOnlyImageArgument(vulkan_utils::image& image);
        void    addWriteOnlyImageArgument(vulkan_utils::image& image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        void    addPodArgument(const void* pod, std::size_t sizeofPod);

        template <typename T>
        void    addPodArgument(const T& pod);

        execution_time_t    run(const WorkgroupDimensions& num_workgroups);

    private:
        void    addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers);

        void    fillCommandBuffer(const WorkgroupDimensions&    num_workgroups);
        void    updateDescriptorSets();
        void    submitCommand();

        const device_t& getDevice() const { return mKernel.get().getModule().getDevice(); }

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
        std::reference_wrapper<kernel>              mKernel;
        vk::UniqueCommandBuffer                     mCommand;
        vk::UniqueQueryPool                         mQueryPool;

        std::vector<int32_t>                        mSpecConstantArguments;
        std::vector<vulkan_utils::uniform_buffer>   mPodBuffers;
        std::vector<vk::ImageMemoryBarrier>         mImageMemoryBarriers;

        std::vector<vk::DescriptorImageInfo>        mLiteralSamplerDescriptors;
        std::vector<vk::DescriptorImageInfo>        mImageDescriptors;
        std::vector<vk::DescriptorBufferInfo>       mBufferDescriptors;
        std::vector<vk::WriteDescriptorSet>         mArgumentDescriptorWrites;
    };

    template <typename T>
    inline void kernel_invocation::addPodArgument(const T& pod) {
        addPodArgument(&pod, sizeof(pod));
    }
}

#endif //CLSPV_UTILS_HPP

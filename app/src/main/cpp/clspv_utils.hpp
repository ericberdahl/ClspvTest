//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPV_UTILS_HPP
#define CLSPV_UTILS_HPP

#include <chrono>
#include <map>
#include <memory>
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
        typedef std::map<int,vk::UniqueSampler> sampler_cache_t;

        device_t() {}

        device_t(vk::PhysicalDevice                  physicalDevice,
                 vk::Device                          device,
                 vk::PhysicalDeviceMemoryProperties  memoryProperties,
                 vk::DescriptorPool                  descriptorPool,
                 vk::CommandPool                     commandPool,
                 vk::Queue                           computeQueue);

        vk::PhysicalDevice                  mPhysicalDevice;
        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::DescriptorPool                  mDescriptorPool;
        vk::CommandPool                     mCommandPool;
        vk::Queue                           mComputeQueue;

        std::shared_ptr<sampler_cache_t>    mSamplerCache;
    };

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags);

    class kernel;

    // TODO: Refactor kernel_module into two classes, module_interface and kernel_module. module_interface is more about which entries exist and kernel_module is more about a loaded module.
    class kernel_module {
    public:
        explicit                    kernel_module(const std::string& moduleName);

                                    ~kernel_module();

        kernel                      createKernel(const std::string&     entryPoint,
                                                 const vk::Extent3D&    workgroup_sizes);

        void                        load(device_t device);
        bool                        isLoaded() const { return (bool)getShaderModule(); }

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;
        vk::ShaderModule            getShaderModule() const { return *mShaderModule; }

        layout_t                    createLayout(const std::string& entryPoint) const;

        vk::ArrayProxy<const vk::Sampler>   getLiteralSamplersHack() const { return mSamplers; }

    private:
        std::string                 mName;
        details::spv_map            mSpvMap;
        std::vector<vk::Sampler>    mSamplers;
        device_t                    mDevice;
        vk::UniqueShaderModule      mShaderModule;
    };

    class kernel_invocation;

    class kernel {
    public:
        kernel();

        kernel(kernel_module&       module,
               device_t             device,
               std::string          entryPoint,
               const vk::Extent3D&  workgroup_sizes);

        kernel(kernel&& other);

        ~kernel();

        kernel& operator=(kernel&& other);

        kernel_invocation   createInvocation();
        void                bindCommand(vk::CommandBuffer command) const;

        std::string         getEntryPoint() const { return mEntryPoint; }
        vk::Extent3D        getWorkgroupSize() const { return mWorkgroupSizes; }

        kernel_module&      getModule() const { return *mModule; }

        const device_t&     getDevice() { return mDevice; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

        void                swap(kernel& other);

    private:
        kernel_module*      mModule = nullptr;
        device_t            mDevice;
        std::string         mEntryPoint;
        vk::Extent3D        mWorkgroupSizes;
        layout_t            mLayout;
        vk::UniquePipeline  mPipeline;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }

    class kernel_invocation {
    public:
                    kernel_invocation();

        explicit    kernel_invocation(kernel&           kernel,
                                      device_t          device,
                                      vk::DescriptorSet literalSamplerDescSet,
                                      vk::DescriptorSet argumentDescSet);

                    kernel_invocation(kernel_invocation&& other);

                    ~kernel_invocation();

        void    addStorageBufferArgument(vulkan_utils::storage_buffer& buffer);
        void    addUniformBufferArgument(vulkan_utils::uniform_buffer& buffer);
        void    addReadOnlyImageArgument(vulkan_utils::image& image);
        void    addWriteOnlyImageArgument(vulkan_utils::image& image);
        void    addSamplerArgument(vk::Sampler samp);
        void    addLocalArraySizeArgument(unsigned int numElements);

        execution_time_t    run(const vk::Extent3D& num_workgroups);

        void    swap(kernel_invocation& other);

    private:
        void    bindCommand();
        void    updatePipeline();
        void    fillCommandBuffer(const vk::Extent3D&    num_workgroups);
        void    updateDescriptorSets();
        void    submitCommand();

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
        kernel*                                 mKernel;
        device_t                                mDevice;
        vk::UniqueCommandBuffer                 mCommand;
        vk::UniqueQueryPool                     mQueryPool;

        vk::DescriptorSet                       mArgumentDescriptorSet;

        std::vector<int32_t>                    mSpecConstantArguments;
        std::vector<vk::ImageMemoryBarrier>     mImageMemoryBarriers;

        std::vector<vk::DescriptorImageInfo>    mImageArgumentInfo;
        std::vector<vk::DescriptorBufferInfo>   mBufferArgumentInfo;
        std::vector<vk::WriteDescriptorSet>     mArgumentDescriptorWrites;

        void    addLiteralSamplers(vk::ArrayProxy<const vk::Sampler> samplers);
        std::vector<vk::DescriptorImageInfo>    mLiteralSamplerInfo;
        vk::DescriptorSet                       mLiteralSamplerDescriptorSet;
        std::vector<vk::WriteDescriptorSet>     mLiteralSamplerDescriptorWrites;
    };

    inline void swap(kernel_invocation & lhs, kernel_invocation & rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPV_UTILS_HPP

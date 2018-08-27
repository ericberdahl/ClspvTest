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

    struct sampler_spec_t {
        int opencl_flags    = 0;
        int descriptor_set  = -1;
        int binding         = -1;
    };

    struct arg_spec_t {
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

    class kernel_interface {
    public:
        typedef std::vector<arg_spec_t>                 arg_list_t;
        typedef vk::ArrayProxy<const sampler_spec_t>    sampler_list_proxy_t;

        kernel_interface();

        kernel_interface(std::string           entryPoint,
                         sampler_list_proxy_t  samplers,
                         arg_list_t            arguments);

        int                         getArgDescriptorSet() const;
        const std::string&          getEntryPoint() const { return mName; }
        vk::UniqueDescriptorSetLayout createArgDescriptorLayout(const device_t& device) const;

        const sampler_list_proxy_t& getLiteralSamplers() const { return mLiteralSamplers; }
        int                         getLiteralSamplersDescriptorSet() const;

    private:
        void        validate() const;

    private:
        std::string             mName;
        sampler_list_proxy_t    mLiteralSamplers;
    public:
        arg_list_t              mArgSpecs;  // TODO: make mArgSpecs private
    };

    struct spv_map {
    public:
        typedef std::vector<sampler_spec_t>     sampler_list_t;
        typedef std::vector<kernel_interface>   kernel_list_t;

        spv_map();

        static spv_map parse(std::istream &in);

        const kernel_interface* findKernel(const std::string& entryPoint) const;

        std::vector<std::string>    getEntryPoints() const;

        int                         getLiteralSamplersDescriptorSet() const;

        vk::UniqueDescriptorSetLayout createLiteralSamplerDescriptorLayout(const device_t& device) const;

    private:
        void    addLiteralSampler(sampler_spec_t sampler);

    public:
        sampler_list_t  samplers;
        kernel_list_t   kernels;
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

    struct kernel_layout_t {
        vk::DescriptorSet               mLiteralSamplerDescriptor;

        vk::UniqueDescriptorSetLayout   mArgumentDescriptorLayout;
        vk::UniqueDescriptorSet         mArgumentsDescriptor;

        vk::UniquePipelineLayout        mPipelineLayout;
    };

    bool isSamplerSupported(int opencl_flags);

    vk::UniqueSampler createCompatibleSampler(vk::Device device, int opencl_flags);

    class kernel_invocation;

    class kernel {
    public:
        typedef vk::ArrayProxy<const arg_spec_t> arg_list_proxy_t;

                            kernel();

                            kernel(device_t             device,
                                   kernel_layout_t      layout,
                                   vk::ShaderModule     shaderModule,
                                   std::string          entryPoint,
                                   const vk::Extent3D&  workgroup_sizes,
                                   arg_list_proxy_t     args);

                            kernel(kernel&& other);

                            ~kernel();

        kernel&             operator=(kernel&& other);

        kernel_invocation   createInvocation();
        void                bindCommand(vk::CommandBuffer command) const;

        std::string         getEntryPoint() const { return mEntryPoint; }
        vk::Extent3D        getWorkgroupSize() const { return mWorkgroupSizes; }

        const device_t&     getDevice() { return mDevice; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

        void                swap(kernel& other);

    private:
        device_t            mDevice;
        vk::ShaderModule    mShaderModule;
        std::string         mEntryPoint;
        vk::Extent3D        mWorkgroupSizes;
        kernel_layout_t     mLayout;
        vk::UniquePipeline  mPipeline;
        arg_list_proxy_t    mArgList;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }

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
        std::vector<std::string>    getEntryPoints() const { return mSpvMap.getEntryPoints(); }
        vk::ShaderModule            getShaderModule() const { return *mShaderModule; }

    private:
        kernel_layout_t             createKernelLayout(const std::string& entryPoint) const;

    private:
        std::string                     mName;
        spv_map                         mSpvMap;
        device_t                        mDevice;
        vk::UniqueShaderModule          mShaderModule;

        vk::UniqueDescriptorSetLayout   mLiteralSamplerDescriptorLayout;
        vk::UniqueDescriptorSet         mLiteralSamplerDescriptor;
    };

    class kernel_invocation {
    public:
        typedef kernel::arg_list_proxy_t arg_list_proxy_t;

                    kernel_invocation();

        explicit    kernel_invocation(kernel&           kernel,
                                      device_t          device,
                                      vk::DescriptorSet argumentDescSet,
                                      arg_list_proxy_t  argList);

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

        // Sanity check that the nth argument (specified by ordinal) has the indicated
        // spvmap type. Throw an exception if false. Return the binding number if true.
        std::uint32_t   validateArgType(std::size_t ordinal, vk::DescriptorType kind) const;
        std::uint32_t   validateArgType(std::size_t ordinal, arg_spec_t::kind_t kind) const;

        std::size_t countArguments() const;

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
        kernel*                                 mKernel = nullptr;
        device_t                                mDevice;
        arg_list_proxy_t                        mArgList;
        vk::UniqueCommandBuffer                 mCommand;
        vk::UniqueQueryPool                     mQueryPool;

        vk::DescriptorSet                       mArgumentDescriptorSet;

        std::vector<vk::BufferMemoryBarrier>    mBufferMemoryBarriers;
        std::vector<vk::ImageMemoryBarrier>     mImageMemoryBarriers;

        std::vector<vk::DescriptorImageInfo>    mImageArgumentInfo;
        std::vector<vk::DescriptorBufferInfo>   mBufferArgumentInfo;

        std::vector<vk::WriteDescriptorSet>     mArgumentDescriptorWrites;
        std::vector<int32_t>                    mSpecConstantArguments;
    };

    inline void swap(kernel_invocation & lhs, kernel_invocation & rhs)
    {
        lhs.swap(rhs);
    }
}

#endif //CLSPV_UTILS_HPP

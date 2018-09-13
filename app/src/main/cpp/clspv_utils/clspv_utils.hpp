//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_CLSPV_UTILS_HPP
#define CLSPVUTILS_CLSPV_UTILS_HPP

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulkan_utils/vulkan_utils.hpp"

namespace clspv_utils {

    class device;
    class kernel;
    class kernel_interface;
    class kernel_invocation;
    class kernel_module;
    class module_interface;

    struct arg_spec_t;
    struct execution_time_t;
    struct kernel_layout_t;
    struct sampler_spec_t;


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

    class device {
    public:
        struct descriptor_group_t
        {
            vk::DescriptorSet       descriptor;
            vk::DescriptorSetLayout layout;
        };

        typedef vk::ArrayProxy<const sampler_spec_t> sampler_list_proxy_t;

        device() {}

        device(vk::PhysicalDevice   physicalDevice,
               vk::Device           device,
               vk::DescriptorPool   descriptorPool,
               vk::CommandPool      commandPool,
               vk::Queue            computeQueue);

        vk::PhysicalDevice  getPhysicalDevice() const { return mPhysicalDevice; }
        vk::Device          getDevice() const { return mDevice; }
        vk::DescriptorPool  getDescriptorPool() const { return mDescriptorPool; }
        vk::CommandPool     getCommandPool() const { return mCommandPool; }
        vk::Queue           getComputeQueue() const { return mComputeQueue; }

        const vk::PhysicalDeviceMemoryProperties&   getMemoryProperties() const { return mMemoryProperties; }

        vk::Sampler                     getCachedSampler(int opencl_flags);

        vk::UniqueDescriptorSetLayout   createSamplerDescriptorLayout(const sampler_list_proxy_t& samplers) const;
        vk::UniqueDescriptorSet         createSamplerDescriptor(const sampler_list_proxy_t& samplers,
                                                                vk::DescriptorSetLayout layout);

        descriptor_group_t              getCachedSamplerDescriptorGroup(const sampler_list_proxy_t& samplers);

    private:
        struct unique_descriptor_group_t
        {
            vk::UniqueDescriptorSet       descriptor;
            vk::UniqueDescriptorSetLayout layout;
        };

        typedef std::map<std::size_t,unique_descriptor_group_t> descriptor_cache_t;
        typedef std::map<int,vk::UniqueSampler> sampler_cache_t;

    private:
        vk::PhysicalDevice                  mPhysicalDevice;
        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::DescriptorPool                  mDescriptorPool;
        vk::CommandPool                     mCommandPool;
        vk::Queue                           mComputeQueue;

        std::shared_ptr<descriptor_cache_t> mSamplerDescriptorCache;
        std::shared_ptr<sampler_cache_t>    mSamplerCache;
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
        vk::UniqueDescriptorSetLayout createArgDescriptorLayout(const device& dev) const;

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

    class module_interface {
    public:
        typedef std::vector<sampler_spec_t>     sampler_list_t;
        typedef std::vector<kernel_interface>   kernel_list_t;

                                        module_interface();

        explicit                        module_interface(const std::string& moduleName);

        const kernel_interface*         findKernelInterface(const std::string& entryPoint) const;

        std::vector<std::string>        getEntryPoints() const;

        int                             getLiteralSamplersDescriptorSet() const;

        kernel_module                   load(device dev) const;

    private:
        void    addLiteralSampler(sampler_spec_t sampler);

    private:
        std::string     mName;
        sampler_list_t  mSamplers;
        kernel_list_t   mKernels;
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

    class kernel_module {
    public:
        typedef vk::ArrayProxy<const kernel_interface>  kernel_list_proxy_t;

                                    kernel_module();

                                    kernel_module(kernel_module&& other);

                                    kernel_module(const std::string&        moduleName,
                                                  device                    dev,
                                                  vk::DescriptorSet         literalSamplerDescriptor,
                                                  vk::DescriptorSetLayout   literalSamplerDescriptorLayout,
                                                  kernel_list_proxy_t       kernelInterfaces);

                                    ~kernel_module();

        kernel_module&              operator=(kernel_module&& other);

        void                        swap(kernel_module& other);

        kernel                      createKernel(const std::string&     entryPoint,
                                                 const vk::Extent3D&    workgroup_sizes);

        bool                        isLoaded() const { return (bool)getShaderModule(); }

        std::string                 getName() const { return mName; }
        std::vector<std::string>    getEntryPoints() const;
        vk::ShaderModule            getShaderModule() const { return *mShaderModule; }

    private:
        kernel_layout_t             createKernelLayout(const std::string& entryPoint) const;

    private:
        std::string                     mName;
        device                          mDevice;
        kernel_list_proxy_t             mKernelInterfaces;

        vk::DescriptorSetLayout         mLiteralSamplerDescriptorLayout;
        vk::DescriptorSet               mLiteralSamplerDescriptor;
        vk::UniqueShaderModule          mShaderModule;
        vk::UniquePipelineCache         mPipelineCache;
    };

    inline void swap(kernel_module& lhs, kernel_module& rhs)
    {
        lhs.swap(rhs);
    }

    class kernel {
    public:
        typedef vk::ArrayProxy<const arg_spec_t> arg_list_proxy_t;

        kernel();

        kernel(device               dev,
               kernel_layout_t      layout,
               vk::ShaderModule     shaderModule,
               vk::PipelineCache    pipelineCache,
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

        const device&       getDevice() { return mDevice; }

        void                updatePipeline(vk::ArrayProxy<int32_t> otherSpecConstants);

        void                swap(kernel& other);

    private:
        device              mDevice;
        vk::ShaderModule    mShaderModule;
        std::string         mEntryPoint;
        vk::Extent3D        mWorkgroupSizes;
        kernel_layout_t     mLayout;
        vk::PipelineCache   mPipelineCache;
        vk::UniquePipeline  mPipeline;
        arg_list_proxy_t    mArgList;
    };

    inline void swap(kernel& lhs, kernel& rhs)
    {
        lhs.swap(rhs);
    }

    class kernel_invocation {
    public:
        typedef kernel::arg_list_proxy_t arg_list_proxy_t;

                    kernel_invocation();

        explicit    kernel_invocation(kernel&           kernel,
                                      device            device,
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
        device                                  mDevice;
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

#endif //CLSPVUTILS_CLSPV_UTILS_HPP

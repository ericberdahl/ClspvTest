//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef VULKAN_UTILS_HPP
#define VULKAN_UTILS_HPP

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

namespace vulkan_utils {
    class buffer;
    class image;

    template <typename Type, typename Deleter>
    std::vector<Type> extractUniques(vk::ArrayProxy<vk::UniqueHandle<Type,Deleter> > uniques) {
        std::vector<Type> result;
        result.reserve(uniques.size());
        for (auto & u : uniques) {
            result.push_back(*u);
        }
        return result;
    };

    template <typename Type, typename Deleter>
    std::vector<Type> extractUniques(const std::vector<vk::UniqueHandle<Type,Deleter> >& uniques) {
        std::vector<Type> result;
        result.reserve(uniques.size());
        for (auto & u : uniques) {
            result.push_back(*u);
        }
        return result;
    };

    vk::UniqueDeviceMemory allocate_device_memory(vk::Device device,
                                                  const vk::MemoryRequirements&             mem_reqs,
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props,
                                                  vk::MemoryPropertyFlags                   property_flags = vk::MemoryPropertyFlags());

    vk::UniqueCommandBuffer allocate_command_buffer(vk::Device device, vk::CommandPool cmd_pool);

    vk::UniquePipeline create_compute_pipeline(vk::Device                       device,
                                               vk::ShaderModule                 shaderModule,
                                               const char*                      entryPoint,
                                               vk::PipelineLayout               pipelineLayout,
                                               vk::PipelineCache                pipelineCache,
                                               vk::ArrayProxy<std::uint32_t>    specConstants);

    buffer createUniformBuffer(vk::Device device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               vk::DeviceSize                           num_bytes);

    buffer createStorageBuffer(vk::Device device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               vk::DeviceSize                           num_bytes);

    buffer createStagingBuffer(vk::Device                               device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               const image&                             image,
                               bool                                     isForInitialzation,
                               bool                                     isForReadback);

    double timestamp_delta_ns(std::uint64_t                         startTimestamp,
                              std::uint64_t                         endTimestamp,
                              const vk::PhysicalDeviceProperties&   deviceProperties,
                              const vk::QueueFamilyProperties&      queueFamilyProperties);

    vk::Extent3D computeNumberWorkgroups(const vk::Extent3D& workgroupSize, const vk::Extent3D& dataSize);

    void copyBufferToImage(vk::CommandBuffer    commandBuffer,
                           buffer&              buffer,
                           image&               image);

    void copyImageToBuffer(vk::CommandBuffer    commandBuffer,
                           image&               image,
                           buffer&              buffer);

    template <typename T>
    using mapped_ptr = std::unique_ptr<T, std::function<void (void*)> >;

    class buffer {
    public:
        buffer () {}

        buffer (vk::Device device,
                const vk::PhysicalDeviceMemoryProperties memoryProperties,
                vk::DeviceSize                           num_bytes,
                vk::BufferUsageFlags                     usage);

        buffer (const buffer & other) = delete;

        buffer (buffer && other);

        ~buffer ();

        buffer & operator=(const buffer & other) = delete;

        buffer & operator=(buffer && other);

        void    swap(buffer & other);

        vk::BufferMemoryBarrier  prepareForShaderRead();
        vk::BufferMemoryBarrier  prepareForShaderWrite();

        vk::BufferMemoryBarrier  prepareForTransferSrc();
        vk::BufferMemoryBarrier  prepareForTransferDst();

        vk::DescriptorBufferInfo use();

        vk::BufferUsageFlags     getUsage() const { return mUsage; }

    public:
        template <typename T>
        inline mapped_ptr<T> map()
        {
            auto basicMap = map();
            return mapped_ptr<T>(static_cast<T*>(basicMap.release()), basicMap.get_deleter());
        }

        mapped_ptr<void> map();

    private:
        void    unmap();

    private:
        vk::BufferUsageFlags    mUsage;
        bool                    mIsMapped = false;

        vk::Device              mDevice;
        vk::UniqueDeviceMemory  mDeviceMemory;
        vk::UniqueBuffer        mBuffer;
    };

    inline void swap(buffer & lhs, buffer & rhs)
    {
        lhs.swap(rhs);
    }

    class image {
    public:
        enum Usage {
            kUsage_ReadOnly,
            kUsage_ReadWrite
        };

    public:
        static bool supportsFormatUse(vk::PhysicalDevice device, vk::Format format, Usage usage);

        image();

        image(vk::Device                                dev,
              const vk::PhysicalDeviceMemoryProperties  memoryProperties,
              vk::Extent3D                              extent,
              vk::Format                                format,
              Usage                                     usage);

        image(const image& other) = delete;

        image(image&& other);

        ~image();

        image&  operator=(const image& other) = delete;

        image&  operator=(image&& other);

        void    swap(image& other);

        vk::DescriptorImageInfo use();
        vk::ImageMemoryBarrier  prepare(vk::ImageLayout newLayout);

        vk::Extent3D getExtent() const { return mExtent; }
        vk::Format getFormat() const { return mFormat; }

    private:
        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::ImageLayout                     mImageLayout;
        vk::UniqueDeviceMemory              mDeviceMemory;
        vk::Extent3D                        mExtent;
        vk::UniqueImage                     mImage;
        vk::UniqueImageView                 mImageView;
        vk::Format                          mFormat;
    };

    inline void swap(image& lhs, image& rhs)
    {
        lhs.swap(rhs);
    }
}

std::ostream& operator<<(std::ostream& os, vk::MemoryPropertyFlags vkFlags);
std::ostream& operator<<(std::ostream& os, vk::MemoryHeapFlags vkFlags);

std::ostream& operator<<(std::ostream& os, const vk::MemoryType& memoryType);
std::ostream& operator<<(std::ostream& os, const vk::MemoryHeap& memoryHeap);

#endif //VULKAN_UTILS_HPP

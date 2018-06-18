//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef VULKAN_UTILS_HPP
#define VULKAN_UTILS_HPP

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.hpp>

namespace vulkan_utils {

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

    class device_memory {
    public:
        struct unmapper_t {
            unmapper_t(device_memory* s) : self(s) {}

            void    operator()(const void* ptr) { self->unmap(); }

            device_memory*  self;
        };

        template <typename T>
        using mapped_ptr = std::unique_ptr<T, unmapper_t>;

    public:
        device_memory() {}

        device_memory(vk::Device                                dev,
                      const vk::MemoryRequirements&             mem_reqs,
                      const vk::PhysicalDeviceMemoryProperties  mem_props);

        device_memory(const device_memory& other) = delete;

        device_memory(device_memory&& other);

        ~device_memory();

        device_memory&  operator=(const device_memory& other) = delete;

        device_memory&  operator=(device_memory&& other);

        void    swap(device_memory& other);

        vk::Device          getDevice() const { return mDevice; }

        void    bind(vk::Buffer buffer, vk::DeviceSize memoryOffset);

        void    bind(vk::Image image, vk::DeviceSize memoryOffset);

        template <typename T>
        inline mapped_ptr<T> map()
        {
            auto basicMap = map();
            return std::unique_ptr<T, unmapper_t>(static_cast<T*>(basicMap.release()), basicMap.get_deleter());
        }

        mapped_ptr<void> map();

    private:
        void    unmap();

    private:
        vk::Device              mDevice;
        vk::UniqueDeviceMemory  mMemory;
        bool                    mMapped;
    };

    inline void swap(device_memory& lhs, device_memory& rhs)
    {
        lhs.swap(rhs);
    }

    class uniform_buffer {
    public:
        template <typename T>
        using mapped_ptr = device_memory::mapped_ptr<T>;

    public:
        uniform_buffer () {}

        uniform_buffer (vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes);

        uniform_buffer (const uniform_buffer& other) = delete;

        uniform_buffer (uniform_buffer&& other);

        ~uniform_buffer();

        uniform_buffer& operator=(const uniform_buffer& other) = delete;

        uniform_buffer& operator=(uniform_buffer&& other);

        void    swap(uniform_buffer& other);

    public:
        template <typename T = void>
        inline mapped_ptr<T> map()
        {
            return mem.map<T>();
        }

    private:
        device_memory       mem;
    public:
        vk::UniqueBuffer    buf;
    };

    inline void swap(uniform_buffer& lhs, uniform_buffer& rhs)
    {
        lhs.swap(rhs);
    }

    class storage_buffer {
    public:
        template <typename T>
        using mapped_ptr = device_memory::mapped_ptr<T>;

    public:
        storage_buffer () {}

        storage_buffer (vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes);

        storage_buffer (const storage_buffer & other) = delete;

        storage_buffer (storage_buffer && other);

        ~storage_buffer ();

        storage_buffer & operator=(const storage_buffer & other) = delete;

        storage_buffer & operator=(storage_buffer && other);

        void    swap(storage_buffer & other);

    public:
        template <typename T = void>
        inline mapped_ptr<T> map()
        {
            return mem.map<T>();
        }

    private:
        device_memory       mem;
    public:
        vk::UniqueBuffer    buf;
    };

    inline void swap(storage_buffer & lhs, storage_buffer & rhs)
    {
        lhs.swap(rhs);
    }

    class staging_buffer;

    class image {
    public:
        enum Usage {
            kUsage_ReadOnly,
            kUsage_ReadWrite
        };

    public:
        image();

        image(vk::Device                                dev,
              const vk::PhysicalDeviceMemoryProperties  memoryProperties,
              uint32_t                                  width,
              uint32_t                                  height,
              vk::Format                                format,
              Usage                                     usage);

        image(const image& other) = delete;

        image(image&& other);

        ~image();

        image&  operator=(const image& other) = delete;

        image&  operator=(image&& other);

        void    swap(image& other);

        staging_buffer  createStagingBuffer();

        vk::DescriptorImageInfo use();
        vk::ImageMemoryBarrier  prepare(vk::ImageLayout newLayout);

    private:
        vk::Device                          mDevice;
        vk::PhysicalDeviceMemoryProperties  mMemoryProperties;
        vk::ImageLayout                     mImageLayout;
        vk::UniqueDeviceMemory              mDeviceMemory;
        uint32_t                            mWidth;
        uint32_t                            mHeight;
        vk::UniqueImage                     mImage;
        vk::UniqueImageView                 mImageView;
    };

    inline void swap(image& lhs, image& rhs)
    {
        lhs.swap(rhs);
    }

    class staging_buffer {
    public:
        template <typename T>
        using mapped_ptr = device_memory::mapped_ptr <T>;

    public:
        staging_buffer ();

        staging_buffer (vk::Device                           device,
                        vk::PhysicalDeviceMemoryProperties   memoryProperties,
                        image*                               image,
                        uint32_t                             width,
                        uint32_t                             height,
                        std::size_t                          pixelSize);

        staging_buffer (const staging_buffer & other) = delete;

        staging_buffer (staging_buffer && other);

        ~staging_buffer ();

        staging_buffer & operator=(const staging_buffer & other) = delete;

        staging_buffer & operator=(staging_buffer && other);

        void    swap(staging_buffer & other);

        void    copyToImage(vk::CommandBuffer commandBuffer);
        void    copyFromImage(vk::CommandBuffer commandBuffer);

        template <typename T>
        inline mapped_ptr<T> map()
        {
            return mDeviceMemory.map<T>();
        }

    private:
        vk::Device              mDevice;
        image*                  mImage;
        uint32_t                mWidth;
        uint32_t                mHeight;
        device_memory           mDeviceMemory;
        vk::UniqueBuffer        mBuffer;
    };

    inline void swap(staging_buffer & lhs, staging_buffer & rhs)
    {
        lhs.swap(rhs);
    }

    double timestamp_delta_ns(uint64_t                              startTimestamp,
                              uint64_t                              endTimestamp,
                              const vk::PhysicalDeviceProperties&   deviceProperties,
                              const vk::QueueFamilyProperties&      queueFamilyProperties);
}

std::ostream& operator<<(std::ostream& os, vk::MemoryPropertyFlags vkFlags);
std::ostream& operator<<(std::ostream& os, const vk::MemoryType& memoryType);

#endif //VULKAN_UTILS_HPP

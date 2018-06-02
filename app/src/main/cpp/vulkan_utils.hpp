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
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props);

    class device_memory {
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

        void    bind(vk::Image im, vk::DeviceSize memoryOffset) const;
        void    bind(vk::Buffer buf, vk::DeviceSize memoryOffset) const;

        template <typename Fn>
        void mappedOp(Fn f) {
            device_memory* const self = this;
            auto unmapper = [self](void* ptr) {
                self->unmap();
            };

            void* memMap = map();
            std::unique_ptr<void, decltype(unmapper)> unmapper_ptr(memMap, unmapper);

            f(memMap);
        }

        template <typename Fn>
        void mappedOp(device_memory& extraMap, Fn f) {
            device_memory* const self = this;
            auto myUnmapper = [self](void* ptr) {
                self->unmap();
            };

            device_memory* const extra = &extraMap;
            auto extraUnmapper = [extra](void* ptr) {
                extra->unmap();
            };

            void* myMemMap = map();
            std::unique_ptr<void, decltype(myUnmapper)> myUnmapperPtr(myMemMap, myUnmapper);

            void* extraMemMap = extraMap.map();
            std::unique_ptr<void, decltype(extraUnmapper)> extraUnmapperPtr(extraMemMap, extraUnmapper);

            f(myMemMap, extraMemMap);
        }

    private:
        void*   map();
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

    void copyToDeviceMemory(device_memory& dst, const void* src, std::size_t numBytes);

    void copyFromDeviceMemory(void* dst, device_memory& src, std::size_t numBytes);

    template <typename T>
    void fillDeviceMemory(device_memory& dst, std::size_t numElements, const T& element) {
        dst.mappedOp([numElements, &element](void* memMap) {
            T* dest_ptr = static_cast<T*>(memMap);
            std::fill(dest_ptr, dest_ptr + numElements, element);
        });
    }

    struct uniform_buffer {
        uniform_buffer () {}

        uniform_buffer (vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes);

        uniform_buffer (const uniform_buffer& other) = delete;

        uniform_buffer (uniform_buffer&& other);

        ~uniform_buffer();

        uniform_buffer& operator=(const uniform_buffer& other) = delete;

        uniform_buffer& operator=(uniform_buffer&& other);

        void    swap(uniform_buffer& other);

        device_memory       mem;
        vk::UniqueBuffer    buf;
    };

    inline void swap(uniform_buffer& lhs, uniform_buffer& rhs)
    {
        lhs.swap(rhs);
    }

    struct storage_buffer {
        storage_buffer () {}

        storage_buffer (vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes);

        storage_buffer (const storage_buffer & other) = delete;

        storage_buffer (storage_buffer && other);

        ~storage_buffer ();

        storage_buffer & operator=(const storage_buffer & other) = delete;

        storage_buffer & operator=(storage_buffer && other);

        void    swap(storage_buffer & other);

        device_memory       mem;
        vk::UniqueBuffer    buf;
    };

    inline void swap(storage_buffer & lhs, storage_buffer & rhs)
    {
        lhs.swap(rhs);
    }

    struct image {
        image() {}

        image(vk::Device                                dev,
              const vk::PhysicalDeviceMemoryProperties  memoryProperties,
              uint32_t                                  width,
              uint32_t                                  height,
              vk::Format                                format) : image() {
            allocate(dev, memoryProperties, width, height, format);
        };

        image(const image& other) = delete;

        image(image&& other);

        ~image();

        image&  operator=(const image& other) = delete;

        image&  operator=(image&& other);

        void    swap(image& other);

        void    allocate(vk::Device                                 dev,
                         const vk::PhysicalDeviceMemoryProperties&  memory_properties,
                         uint32_t                                   width,
                         uint32_t                                   height,
                         vk::Format                                 format);
        void    reset();

        device_memory       mem;
        vk::UniqueImage     im;
        vk::UniqueImageView view;
    };

    inline void swap(image& lhs, image& rhs)
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

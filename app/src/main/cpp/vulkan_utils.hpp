//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef VULKAN_UTILS_HPP
#define VULKAN_UTILS_HPP

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.hpp>

struct sample_info;

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

    struct device_memory {
        device_memory() {}

        device_memory(vk::Device                                dev,
                      const vk::MemoryRequirements&             mem_reqs,
                      const vk::PhysicalDeviceMemoryProperties  mem_props)
                : device_memory() {
            allocate(dev, mem_reqs, mem_props);
        };

        device_memory(const device_memory& other) = delete;

        device_memory(device_memory&& other);

        ~device_memory();

        device_memory&  operator=(const device_memory& other) = delete;

        device_memory&  operator=(device_memory&& other);

        void    swap(device_memory& other);

        void    allocate(vk::Device                                 dev,
                         const vk::MemoryRequirements&              mem_reqs,
                         const vk::PhysicalDeviceMemoryProperties&  mem_props);
        void    reset();

        vk::Device              device;
        vk::UniqueDeviceMemory  mem;
    };

    inline void swap(device_memory& lhs, device_memory& rhs)
    {
        lhs.swap(rhs);
    }

    struct uniform_buffer {
        uniform_buffer () {}

        uniform_buffer (const sample_info &info, vk::DeviceSize num_bytes);

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

        storage_buffer (const sample_info &info, vk::DeviceSize num_bytes);

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

        image(const sample_info&  info,
              uint32_t      width,
              uint32_t      height,
              vk::Format    format);

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

    class memory_map {
    public:
        memory_map() : data(nullptr) {}

        memory_map(vk::Device device, vk::DeviceMemory memory);

        memory_map(const device_memory& mem) : memory_map(mem.device, *mem.mem) {}

        memory_map(const storage_buffer & buf) : memory_map(buf.mem) {}

        memory_map(const image& im) : memory_map(im.mem) {}

        memory_map(const memory_map& other) = delete;

        memory_map(memory_map&& other);

        ~memory_map();

        memory_map& operator=(const memory_map& other) = delete;

        memory_map& operator=(memory_map&& other);

        void    swap(memory_map& other);

        void*   map();
        void    unmap();

    private:
        vk::Device        dev;
        vk::DeviceMemory  mem;
        void*             data;
    };

    inline void swap(memory_map& lhs, memory_map& rhs)
    {
        lhs.swap(rhs);
    }
}

std::ostream& operator<<(std::ostream& os, vk::MemoryPropertyFlags vkFlags);
std::ostream& operator<<(std::ostream& os, const vk::MemoryType& memoryType);

#endif //VULKAN_UTILS_HPP

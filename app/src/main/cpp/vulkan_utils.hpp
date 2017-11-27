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
    std::vector<Type> extractUniques(const std::vector<vk::UniqueHandle<Type,Deleter> >& uniques) {
        std::vector<Type> result;
        result.reserve(uniques.size());
        for (auto & u : uniques) {
            result.push_back(*u);
        }
        return result;
    };

    void throwIfNotSuccess(VkResult result, const std::string& label);

    vk::UniqueDeviceMemory allocate_device_memory(vk::Device device,
                                                  const vk::MemoryRequirements&             mem_reqs,
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props);

    struct device_memory {
        device_memory() : device(), mem() {}
        device_memory(vk::Device                                dev,
                      const vk::MemoryRequirements&             mem_reqs,
                      const vk::PhysicalDeviceMemoryProperties  mem_props)
                : device_memory() {
            allocate(dev, mem_reqs, mem_props);
        };

        ~device_memory();

        void    allocate(vk::Device                                 dev,
                         const vk::MemoryRequirements&              mem_reqs,
                         const vk::PhysicalDeviceMemoryProperties&  mem_props);
        void    reset();

        vk::Device        device;
        vk::DeviceMemory  mem;
    };

    struct buffer {
        buffer() : mem(), buf() {}
        buffer(const sample_info &info, vk::DeviceSize num_bytes);

        buffer(vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes) : buffer() {
            allocate(dev, memoryProperties, num_bytes);
        };

        ~buffer();

        void    allocate(vk::Device dev, const vk::PhysicalDeviceMemoryProperties& memory_properties, vk::DeviceSize num_bytes);
        void    reset();

        device_memory   mem;
        vk::Buffer      buf;
    };

    struct image {
        image() : mem(), im(), view() {}
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

        ~image();

        void    allocate(vk::Device                                 dev,
                         const vk::PhysicalDeviceMemoryProperties&  memory_properties,
                         uint32_t                                   width,
                         uint32_t                                   height,
                         vk::Format                                 format);
        void    reset();

        device_memory   mem;
        vk::Image       im;
        vk::ImageView   view;
    };

    class memory_map {
    public:
        memory_map(vk::Device dev, vk::DeviceMemory mem);
        memory_map(const device_memory& mem) : memory_map(mem.device, mem.mem) {}
        memory_map(const buffer& buf) : memory_map(buf.mem) {}
        memory_map(const image& im) : memory_map(im.mem) {}

        memory_map(const memory_map& other) = delete;
        memory_map& operator=(const memory_map& other) = delete;

        ~memory_map();

        void*   map();
        void    unmap();

    private:
        vk::Device        dev;
        vk::DeviceMemory  mem;
        void*             data;
    };
}

#endif //VULKAN_UTILS_HPP

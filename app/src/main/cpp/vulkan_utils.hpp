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

    void throwIfNotSuccess(VkResult result, const std::string& label);

    struct device_memory {
        device_memory() : device(VK_NULL_HANDLE), mem(VK_NULL_HANDLE) {}
        device_memory(VkDevice                                  dev,
                      const VkMemoryRequirements&               mem_reqs,
                      const VkPhysicalDeviceMemoryProperties    memoryProperties)
                : device_memory() {
            allocate(dev, mem_reqs, memoryProperties);
        };

        void    allocate(VkDevice                                   dev,
                         const VkMemoryRequirements&                mem_reqs,
                         const VkPhysicalDeviceMemoryProperties&    memory_properties);
        void    reset();

        VkDevice        device;
        VkDeviceMemory  mem;
    };

    struct buffer {
        buffer() : mem(), buf(VK_NULL_HANDLE) {}
        buffer(const sample_info &info, VkDeviceSize num_bytes);

        buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties memoryProperties, VkDeviceSize num_bytes) : buffer() {
            allocate(dev, memoryProperties, num_bytes);
        };

        void    allocate(VkDevice dev, const VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceSize num_bytes);
        void    reset();

        device_memory   mem;
        VkBuffer        buf;
    };

    struct image {
        image() : mem(), im(VK_NULL_HANDLE), view(VK_NULL_HANDLE) {}
        image(const sample_info&  info,
              uint32_t      width,
              uint32_t      height,
              VkFormat      format);

        image(VkDevice dev,
              const VkPhysicalDeviceMemoryProperties memoryProperties,
              uint32_t                                   width,
              uint32_t                                   height,
              VkFormat                                   format) : image() {
            allocate(dev, memoryProperties, width, height, format);
        };

        void    allocate(VkDevice                                   dev,
                         const VkPhysicalDeviceMemoryProperties&    memory_properties,
                         uint32_t                                   width,
                         uint32_t                                   height,
                         VkFormat                                   format);
        void    reset();

        device_memory   mem;
        VkImage         im;
        VkImageView     view;
    };

    struct memory_map {
        memory_map(VkDevice dev, VkDeviceMemory mem);
        memory_map(const device_memory& mem) : memory_map(mem.device, mem.mem) {}
        memory_map(const buffer& buf) : memory_map(buf.mem) {}
        memory_map(const image& im) : memory_map(im.mem) {}
        ~memory_map();

        VkDevice        dev;
        VkDeviceMemory  mem;
        void*           data;
    };
}

#endif //VULKAN_UTILS_HPP

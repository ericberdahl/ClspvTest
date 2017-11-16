//
// Created by Eric Berdahl on 10/22/17.
//

#include <cassert>
#include <limits>

#include "util.hpp"
#include "vulkan_utils.hpp"

VkResult vkCreateDebugReportCallbackEXT(
        VkInstance                                  instance,
        const VkDebugReportCallbackCreateInfoEXT*   pCreateInfo,
        const VkAllocationCallbacks*                pAllocator,
        VkDebugReportCallbackEXT*                   pCallback) {
    PFN_vkCreateDebugReportCallbackEXT fn = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
    if (!fn) return VK_ERROR_INITIALIZATION_FAILED;

    return fn(instance, pCreateInfo, pAllocator, pCallback);
}

void vkDestroyDebugReportCallbackEXT(
        VkInstance                                  instance,
        VkDebugReportCallbackEXT                    callback,
        const VkAllocationCallbacks*                pAllocator) {
    PFN_vkDestroyDebugReportCallbackEXT fn = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");

    if (fn) fn(instance, callback, pAllocator);
}

namespace vulkan_utils {

    namespace {
        uint32_t find_compatible_memory_index(const VkPhysicalDeviceMemoryProperties& memory_properties,
                                              uint32_t   typeBits,
                                              VkFlags    requirements_mask) {
            uint32_t result = std::numeric_limits<uint32_t>::max();
            assert(memory_properties.memoryTypeCount < std::numeric_limits<uint32_t>::max());

            // Search memtypes to find first index with those properties
            for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
                if ((typeBits & 1) == 1) {
                    // Type is available, does it match user properties?
                    if ((memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                        result = i;
                        break;
                    }
                }
                typeBits >>= 1;
            }

            return result;
        }
    }

    void throwIfNotSuccess(VkResult result, const std::string& label) {
        if (VK_SUCCESS != result) {
            throw vk::SystemError( vk::make_error_code( vk::Result(result) ), label );
        }
    }

    buffer::buffer(const sample_info &info, VkDeviceSize num_bytes) :
            buffer(info.device, info.memory_properties, num_bytes)
    {

    }

    image::image(const sample_info& info,
                 uint32_t           width,
                 uint32_t           height,
                 VkFormat           format) :
            image(info.device, info.memory_properties, width, height, format)
    {

    }

    memory_map::memory_map(VkDevice device, VkDeviceMemory memory) :
            dev(device), mem(memory), data(nullptr) {
        throwIfNotSuccess(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data),
                          "vkMapMemory");
    }

    memory_map::~memory_map() {
        if (dev && mem) {
            vkUnmapMemory(dev, mem);
        }
    }

    void device_memory::allocate(VkDevice dev,
                                 const VkMemoryRequirements &mem_reqs,
                                 const VkPhysicalDeviceMemoryProperties &memory_properties) {
        reset();

        device = dev;

        // Allocate memory for the buffer
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = find_compatible_memory_index(memory_properties,
                                                                  mem_reqs.memoryTypeBits,
                                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() &&
               "No mappable, coherent memory");
        throwIfNotSuccess(vkAllocateMemory(device, &alloc_info, NULL, &mem),
                          "vkAllocateMemory");
    }

    void device_memory::reset() {
        if (mem != VK_NULL_HANDLE) {
            vkFreeMemory(device, mem, NULL);
            mem = VK_NULL_HANDLE;
        }

        device = VK_NULL_HANDLE;
    }

    void buffer::allocate(VkDevice dev,
                          const VkPhysicalDeviceMemoryProperties &memory_properties,
                          VkDeviceSize inNumBytes) {
        reset();

        // Allocate the buffer
        VkBufferCreateInfo buf_info = {};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_info.size = inNumBytes;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        throwIfNotSuccess(vkCreateBuffer(dev, &buf_info, NULL, &buf),
                          "vkCreateBuffer");

        // Find out what we need in order to allocate memory for the buffer
        VkMemoryRequirements mem_reqs = {};
        vkGetBufferMemoryRequirements(dev, buf, &mem_reqs);

        mem.allocate(dev, mem_reqs, memory_properties);

        // Bind the memory to the buffer object
        throwIfNotSuccess(vkBindBufferMemory(dev, buf, mem.mem, 0),
                          "vkBindBufferMemory");
    }

    void buffer::reset() {
        if (buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(mem.device, buf, NULL);
            buf = VK_NULL_HANDLE;
        }

        mem.reset();
    }

    void image::allocate(VkDevice dev,
                         const VkPhysicalDeviceMemoryProperties &memory_properties,
                         uint32_t width,
                         uint32_t height,
                         VkFormat format) {
        reset();

        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_GENERAL;

        throwIfNotSuccess(vkCreateImage(dev, &imageInfo, nullptr, &im),
                          "vkCreateImage");

        // Find out what we need in order to allocate memory for the image
        VkMemoryRequirements mem_reqs = {};
        vkGetImageMemoryRequirements(dev, im, &mem_reqs);

        mem.allocate(dev, mem_reqs, memory_properties);

        // Bind the memory to the image object
        throwIfNotSuccess(vkBindImageMemory(dev, im, mem.mem, 0),
                          "vkBindImageMemory");

        // Allocate the image view
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = im;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;

        throwIfNotSuccess(vkCreateImageView(dev, &viewInfo, nullptr, &view),
                          "vkCreateImageView");
    }

    void image::reset() {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(mem.device, view, NULL);
            view = VK_NULL_HANDLE;
        }
        if (im != VK_NULL_HANDLE) {
            vkDestroyImage(mem.device, im, NULL);
            im = VK_NULL_HANDLE;
        }

        mem.reset();
    }

} // namespace vulkan_utils

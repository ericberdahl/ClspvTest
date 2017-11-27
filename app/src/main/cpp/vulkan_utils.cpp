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
        uint32_t find_compatible_memory_index(const vk::PhysicalDeviceMemoryProperties& mem_props,
                                              uint32_t   typeBits,
                                              vk::MemoryPropertyFlags   requirements_mask) {
            uint32_t result = std::numeric_limits<uint32_t>::max();
            assert(mem_props.memoryTypeCount < std::numeric_limits<uint32_t>::max());

            // Search memtypes to find first index with those properties
            for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
                if ((typeBits & 1) == 1) {
                    // Type is available, does it match user properties?
                    if ((mem_props.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
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

    vk::UniqueDeviceMemory allocate_device_memory(vk::Device device,
                                                  const vk::MemoryRequirements&             mem_reqs,
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props)
    {
        // Allocate memory for the buffer
        vk::MemoryAllocateInfo alloc_info;
        alloc_info.setAllocationSize(mem_reqs.size)
                .setMemoryTypeIndex(find_compatible_memory_index(mem_props,
                                                                 mem_reqs.memoryTypeBits,
                                                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() &&
               "No mappable, coherent memory");
        return device.allocateMemoryUnique(alloc_info);
    }

    buffer::buffer(const sample_info &info, vk::DeviceSize num_bytes) :
            buffer(*info.device, info.memory_properties, num_bytes)
    {

    }

    image::image(const sample_info& info,
                 uint32_t           width,
                 uint32_t           height,
                 vk::Format         format) :
            image(*info.device, info.memory_properties, width, height, format)
    {

    }

    memory_map::memory_map(vk::Device device, vk::DeviceMemory memory) :
            dev(device),
            mem(memory),
            data(nullptr)
    {
        map();
    }

    memory_map::~memory_map()
    {
        unmap();
    }

    void* memory_map::map()
    {
        if (!data) {
            data = dev.mapMemory(mem, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
        }
        return data;
    }

    void memory_map::unmap()
    {
        if (data) {
            dev.unmapMemory(mem);
            data = nullptr;
        }
    }

    device_memory::~device_memory() {
        if (device || mem) {
            LOGI("device_memory was not reset");
        }
    }

    void device_memory::allocate(vk::Device                                 dev,
                                 const vk::MemoryRequirements&              mem_reqs,
                                 const vk::PhysicalDeviceMemoryProperties&  memory_properties) {
        reset();

        auto memory = allocate_device_memory(dev, mem_reqs, memory_properties);

        device = dev;
        mem = memory.release();
    }

    void device_memory::reset() {
        if (mem) {
            device.freeMemory(mem);
            mem = nullptr;
        }

        device = nullptr;
    }

    buffer::~buffer() {
        if (buf) {
            LOGI("buffer was not reset");
        }
    }

    void buffer::allocate(vk::Device                                dev,
                          const vk::PhysicalDeviceMemoryProperties& memory_properties,
                          vk::DeviceSize                            inNumBytes) {
        reset();

        // Allocate the buffer
        vk::BufferCreateInfo buf_info;
        buf_info.setUsage(vk::BufferUsageFlagBits::eStorageBuffer)
                .setSize(inNumBytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        buf = dev.createBuffer(buf_info);

        mem.allocate(dev, dev.getBufferMemoryRequirements(buf), memory_properties);

        // Bind the memory to the buffer object
        dev.bindBufferMemory(buf, mem.mem, 0);
    }

    void buffer::reset() {
        if (buf) {
            mem.device.destroyBuffer(buf);
            buf = nullptr;
        }

        mem.reset();
    }

    image::~image() {
        if (im) {
            LOGI("image was not reset");
        }
    }

    void image::allocate(vk::Device                                 dev,
                         const vk::PhysicalDeviceMemoryProperties&  memory_properties,
                         uint32_t                                   width,
                         uint32_t                                   height,
                         vk::Format                                 format)
    {
        reset();

        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(vk::ImageType::e2D)
                .setFormat(format)
                .setExtent(vk::Extent3D(width, height, 1))
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setTiling(vk::ImageTiling::eLinear)
                .setUsage(vk::ImageUsageFlagBits::eStorage |
                                  vk::ImageUsageFlagBits::eSampled |
                                  vk::ImageUsageFlagBits::eTransferDst |
                                  vk::ImageUsageFlagBits::eTransferSrc)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setInitialLayout(vk::ImageLayout::eGeneral);

        im = dev.createImage(imageInfo);

        // Find out what we need in order to allocate memory for the image
        mem.allocate(dev, dev.getImageMemoryRequirements(im), memory_properties);

        // Bind the memory to the image object
        dev.bindImageMemory(im, mem.mem, 0);

        // Allocate the image view
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.setImage(im)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format)
                .subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
                                 .setLevelCount(1)
                                 .setLayerCount(1);

        view = dev.createImageView(viewInfo);
    }

    void image::reset() {
        if (view) {
            mem.device.destroyImageView(view);
            view = nullptr;
        }
        if (im) {
            mem.device.destroyImage(im);
            im = nullptr;
        }

        mem.reset();
    }

} // namespace vulkan_utils

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

    memory_map::memory_map(memory_map&& other) :
            memory_map()
    {
        swap(other);
    }

    memory_map::~memory_map()
    {
        unmap();
    }

    memory_map& memory_map::operator=(memory_map&& other)
    {
        swap(other);
        return *this;
    }

    void memory_map::swap(memory_map& other)
    {
        using std::swap;

        swap(dev, other.dev);
        swap(mem, other.mem);
        swap(data, other.data);
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

    device_memory::device_memory(device_memory&& other) :
            device_memory()
    {
        swap(other);
    }

    device_memory::~device_memory() {
        if (!device || !mem) {
            // LOGI("device_memory was reset");
        }
    }

    device_memory& device_memory::operator=(device_memory&& other)
    {
        swap(other);
        return *this;
    }

    void device_memory::swap(device_memory& other)
    {
        using std::swap;

        swap(device, other.device);
        swap(mem, other.mem);
    }

    void device_memory::allocate(vk::Device                                 dev,
                                 const vk::MemoryRequirements&              mem_reqs,
                                 const vk::PhysicalDeviceMemoryProperties&  memory_properties) {
        reset();

        mem = allocate_device_memory(dev, mem_reqs, memory_properties);
        device = dev;
    }

    void device_memory::reset() {
        mem.reset();
        device = nullptr;
    }

    buffer::buffer(buffer&& other) :
            buffer()
    {
        swap(other);
    }

    buffer::~buffer() {
        if (!buf) {
            // LOGI("buffer was reset");
        }
    }

    buffer& buffer::operator=(buffer&& other)
    {
        swap(other);
        return *this;
    }

    void buffer::swap(buffer& other)
    {
        using std::swap;

        swap(mem, other.mem);
        swap(buf, other.buf);
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

        buf = dev.createBufferUnique(buf_info);

        mem.allocate(dev, dev.getBufferMemoryRequirements(*buf), memory_properties);

        // Bind the memory to the buffer object
        dev.bindBufferMemory(*buf, *mem.mem, 0);
    }

    void buffer::reset() {
        buf.reset();
        mem.reset();
    }

    image::image(image&& other) :
            image()
    {
        swap(other);
    }

    image::~image() {
        if (!im) {
            // LOGI("image was reset");
        }
    }

    image& image::operator=(image&& other)
    {
        swap(other);
        return *this;
    }

    void image::swap(image& other)
    {
        using std::swap;

        swap(mem, other.mem);
        swap(im, other.im);
        swap(view, other.view);
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

        im = dev.createImageUnique(imageInfo);

        // Find out what we need in order to allocate memory for the image
        mem.allocate(dev, dev.getImageMemoryRequirements(*im), memory_properties);

        // Bind the memory to the image object
        dev.bindImageMemory(*im, *mem.mem, 0);

        // Allocate the image view
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.setImage(*im)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format)
                .subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
                                 .setLevelCount(1)
                                 .setLayerCount(1);

        view = dev.createImageViewUnique(viewInfo);
    }

    void image::reset() {
        view.reset();
        im.reset();
        mem.reset();
    }

} // namespace vulkan_utils

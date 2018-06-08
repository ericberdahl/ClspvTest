//
// Created by Eric Berdahl on 10/22/17.
//

#include <cassert>
#include <iterator>
#include <limits>

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

            if (result == std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("no compatible memory for allocation");
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
                                                                 vk::MemoryPropertyFlagBits::eHostVisible));
        assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() &&
               "No mappable memory");
        return device.allocateMemoryUnique(alloc_info);
    }

    void copyFromDeviceMemory(void* dst, device_memory& src, std::size_t numBytes)
    {
        src.mappedOp([dst, numBytes](void* src_ptr) {
            std::memcpy(dst, src_ptr, numBytes);
        });
    }

    void copyToDeviceMemory(device_memory& dst, const void* src, std::size_t numBytes)
    {
        dst.mappedOp([src, numBytes](void* dest_ptr) {
            std::memcpy(dest_ptr, src, numBytes);
        });
    }

    device_memory::device_memory(vk::Device                                dev,
                                 const vk::MemoryRequirements&             mem_reqs,
                                 const vk::PhysicalDeviceMemoryProperties  mem_props)
            : mDevice(dev),
              mMemory(allocate_device_memory(dev, mem_reqs, mem_props)),
              mMapped(false)
    {
    }

    device_memory::device_memory(device_memory&& other) :
            device_memory()
    {
        swap(other);
    }

    device_memory::~device_memory()
    {
    }

    device_memory& device_memory::operator=(device_memory&& other)
    {
        swap(other);
        return *this;
    }

    void device_memory::swap(device_memory& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mMemory, other.mMemory);
        swap(mMapped, other.mMapped);
    }

    void device_memory::bind(vk::Buffer buf, vk::DeviceSize memoryOffset) const
    {
        mDevice.bindBufferMemory(buf, *mMemory, memoryOffset);
    }

    void device_memory::bind(vk::Image im, vk::DeviceSize memoryOffset) const
    {
        mDevice.bindImageMemory(im, *mMemory, memoryOffset);
    }

    device_memory::mapped_ptr_t device_memory::map()
    {
        if (mMapped) {
            throw std::runtime_error("device_memory is already mapped");
        }

        void* memMap = mDevice.mapMemory(*mMemory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
        mMapped = true;

        const vk::MappedMemoryRange mappedRange(*mMemory, 0, VK_WHOLE_SIZE);
        mDevice.invalidateMappedMemoryRanges(mappedRange);

        return mapped_ptr_t(memMap, unmapper_t(this));
    }

    void device_memory::unmap()
    {
        if (!mMapped) {
            throw std::runtime_error("device_memory is not mapped");
        }

        const vk::MappedMemoryRange mappedRange(*mMemory, 0, VK_WHOLE_SIZE);
        mDevice.flushMappedMemoryRanges(mappedRange);

        mDevice.unmapMemory(*mMemory);
        mMapped = false;
    }

    uniform_buffer::uniform_buffer(vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes) :
            uniform_buffer()
    {
        // Allocate the buffer
        vk::BufferCreateInfo buf_info;
        buf_info.setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
                .setSize(num_bytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        buf = dev.createBufferUnique(buf_info);

        mem = device_memory(dev, dev.getBufferMemoryRequirements(*buf), memoryProperties);

        // Bind the memory to the buffer object
        mem.bind(*buf, 0);
    }

    uniform_buffer::uniform_buffer(uniform_buffer&& other) :
            uniform_buffer()
    {
        swap(other);
    }

    uniform_buffer::~uniform_buffer() {
    }

    uniform_buffer& uniform_buffer::operator=(uniform_buffer&& other)
    {
        swap(other);
        return *this;
    }

    void uniform_buffer::swap(uniform_buffer& other)
    {
        using std::swap;

        swap(mem, other.mem);
        swap(buf, other.buf);
    }

    storage_buffer::storage_buffer(vk::Device dev, const vk::PhysicalDeviceMemoryProperties memoryProperties, vk::DeviceSize num_bytes) :
            storage_buffer()
    {
        // Allocate the buffer
        vk::BufferCreateInfo buf_info;
        buf_info.setUsage(vk::BufferUsageFlagBits::eStorageBuffer)
                .setSize(num_bytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        buf = dev.createBufferUnique(buf_info);

        mem = device_memory(dev, dev.getBufferMemoryRequirements(*buf), memoryProperties);

        // Bind the memory to the buffer object
        mem.bind(*buf, 0);
    }

    storage_buffer::storage_buffer(storage_buffer&& other) :
            storage_buffer()
    {
        swap(other);
    }

    storage_buffer::~storage_buffer() {
    }

    storage_buffer& storage_buffer::operator=(storage_buffer&& other)
    {
        swap(other);
        return *this;
    }

    void storage_buffer::swap(storage_buffer& other)
    {
        using std::swap;

        swap(mem, other.mem);
        swap(buf, other.buf);
    }

    image::image()
            : mem(),
              im(),
              view()
    {
        // this space intentionally left blank
    }

    image::image(vk::Device                                dev,
                 const vk::PhysicalDeviceMemoryProperties  memoryProperties,
                 uint32_t                                  width,
                 uint32_t                                  height,
                 vk::Format                                format)
            : image()
    {
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
                .setInitialLayout(vk::ImageLayout::ePreinitialized);

        im = dev.createImageUnique(imageInfo);

        // Find out what we need in order to allocate memory for the image
        mem = device_memory(dev, dev.getImageMemoryRequirements(*im), memoryProperties);

        // Bind the memory to the image object
        mem.bind(*im, 0);

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

    image::image(image&& other)
            : image()
    {
        swap(other);
    }

    image::~image()
    {
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

    double timestamp_delta_ns(uint64_t                              startTimestamp,
                              uint64_t                              endTimestamp,
                              const vk::PhysicalDeviceProperties&   deviceProperties,
                              const vk::QueueFamilyProperties&      queueFamilyProperties) {
        uint64_t timestampDelta;
        if (endTimestamp >= startTimestamp) {
            timestampDelta = endTimestamp - startTimestamp;
        }
        else {
            const uint64_t maxTimestamp = std::numeric_limits<uint64_t>::max() >> (64 - queueFamilyProperties.timestampValidBits);
            timestampDelta = maxTimestamp - startTimestamp + endTimestamp + 1;
        }

        return timestampDelta * deviceProperties.limits.timestampPeriod;
    }

} // namespace vulkan_utils

std::ostream& operator<<(std::ostream& os, vk::MemoryPropertyFlags vkFlags) {
    const VkMemoryPropertyFlags flags = (VkMemoryPropertyFlags) vkFlags;
    if (0 == flags) {
        os << "0";
    }
    else {
        std::vector<const char*> bits;
        bits.reserve(5);
        if (0 != (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            bits.push_back("eDeviceLocal");
        }
        if (0 != (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            bits.push_back("eHostVisible");
        }
        if (0 != (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            bits.push_back("eHostCoherent");
        }
        if (0 != (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
            bits.push_back("eHostCached");
        }
        if (0 != (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) {
            bits.push_back("eLazilyAllocated");
        }
        os << std::hex << std::showbase << (int)flags << '(';
        std::copy(bits.begin(), bits.end(), std::ostream_iterator<const char*>(os, " | "));
        os << ')';
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const vk::MemoryType& memoryType) {
    os << "heapIndex:" << memoryType.heapIndex << " flags:" << memoryType.propertyFlags;
    return os;
}
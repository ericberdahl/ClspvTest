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

    vk::UniqueDeviceMemory allocate_device_memory(vk::Device                                device,
                                                  const vk::MemoryRequirements&             mem_reqs,
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props,
                                                  vk::MemoryPropertyFlags                   property_flags)
    {
        // Allocate memory for the buffer
        vk::MemoryAllocateInfo alloc_info;
        alloc_info.setAllocationSize(mem_reqs.size)
                .setMemoryTypeIndex(find_compatible_memory_index(mem_props,
                                                                 mem_reqs.memoryTypeBits,
                                                                 property_flags));
        assert(alloc_info.memoryTypeIndex < std::numeric_limits<uint32_t>::max() &&
               "No mappable memory");
        return device.allocateMemoryUnique(alloc_info);
    }

    vk::UniqueCommandBuffer allocate_command_buffer(vk::Device device, vk::CommandPool cmd_pool) {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(cmd_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);

        auto buffers = device.allocateCommandBuffersUnique(allocInfo);
        assert(buffers.size() == 1);
        return std::move(buffers[0]);
    }
    
    device_memory::device_memory(vk::Device                                dev,
                                 const vk::MemoryRequirements&             mem_reqs,
                                 const vk::PhysicalDeviceMemoryProperties  mem_props)
            : mDevice(dev),
              mMemory(allocate_device_memory(dev, mem_reqs, mem_props, vk::MemoryPropertyFlagBits::eHostVisible)),
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

    std::unique_ptr<void, device_memory::unmapper_t> device_memory::map()
    {
        if (mMapped) {
            throw std::runtime_error("device_memory is already mapped");
        }

        void* memMap = mDevice.mapMemory(*mMemory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
        mMapped = true;

        const vk::MappedMemoryRange mappedRange(*mMemory, 0, VK_WHOLE_SIZE);
        mDevice.invalidateMappedMemoryRanges(mappedRange);

        return std::unique_ptr<void, device_memory::unmapper_t>(memMap, unmapper_t(this));
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
        dev.bindBufferMemory(*buf, mem.getDeviceMemory(), 0);
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
        buf_info.setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc)
                .setSize(num_bytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        buf = dev.createBufferUnique(buf_info);

        mem = device_memory(dev, dev.getBufferMemoryRequirements(*buf), memoryProperties);

        // Bind the memory to the buffer object
        dev.bindBufferMemory(*buf, mem.getDeviceMemory(), 0);
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
            : mDevice(),
              mMemoryProperties(),
              mImageLayout(vk::ImageLayout::eUndefined),
              mDeviceMemory(),
              mWidth(0),
              mHeight(0),
              mImage(),
              mImageView()
    {
        // this space intentionally left blank
    }

    void image::swap(image& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mMemoryProperties, other.mMemoryProperties);
        swap(mImageLayout, other.mImageLayout);
        swap(mDeviceMemory, other.mDeviceMemory);
        swap(mWidth, other.mWidth);
        swap(mHeight, other.mHeight);
        swap(mImage, other.mImage);
        swap(mImageView, other.mImageView);
    }

    image::image(vk::Device                                dev,
                 const vk::PhysicalDeviceMemoryProperties  memoryProperties,
                 uint32_t                                  width,
                 uint32_t                                  height,
                 vk::Format                                format)
            : image()
    {
        mDevice = dev;
        mMemoryProperties = memoryProperties;
        mWidth = width;
        mHeight = height;

        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(vk::ImageType::e2D)
                .setFormat(format)
                .setExtent(vk::Extent3D(width, height, 1))
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setTiling(vk::ImageTiling::eOptimal)
                .setUsage(vk::ImageUsageFlagBits::eStorage |
                          vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eTransferDst |
                          vk::ImageUsageFlagBits::eTransferSrc)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setInitialLayout(mImageLayout);

        mImage = mDevice.createImageUnique(imageInfo);

        // allocate device memory for the image
        mDeviceMemory = allocate_device_memory(mDevice,
                                               mDevice.getImageMemoryRequirements(*mImage),
                                               mMemoryProperties);

        // Bind the memory to the image object
        mDevice.bindImageMemory(*mImage, *mDeviceMemory, 0);

        // Allocate the image view
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.setImage(*mImage)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format)
                .subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLevelCount(1)
                .setLayerCount(1);

        mImageView = mDevice.createImageViewUnique(viewInfo);
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

    staging_buffer image::createStagingBuffer()
    {
        return staging_buffer(mDevice,
                              mMemoryProperties,
                              this,
                              mWidth,
                              mHeight,
                              16);   // TODO: Get correct pixel size
    }

    vk::DescriptorImageInfo image::use()
    {
        vk::DescriptorImageInfo result;
        result.setImageView(*mImageView)
                .setImageLayout(mImageLayout);
        return result;
    }

    vk::ImageMemoryBarrier image::prepare(vk::ImageLayout newLayout)
    {
        if (newLayout == vk::ImageLayout::eUndefined)
        {
            throw std::runtime_error("images cannot be transitioned to undefined layout");
        }

        const auto accessMap = {
                std::make_pair(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead),
                std::make_pair(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits::eTransferWrite),
                std::make_pair(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eTransferRead),
                std::make_pair(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite),
                std::make_pair(vk::ImageLayout::eUndefined, vk::AccessFlagBits(0))
        };

        auto layoutFinder = [](decltype(accessMap)::const_reference item, vk::ImageLayout layout) {
            return item.first == layout;
        };

        auto oldAccess = std::find_if(accessMap.begin(), accessMap.end(), std::bind(layoutFinder, std::placeholders::_1, mImageLayout));
        assert(oldAccess != accessMap.end());

        auto newAccess = std::find_if(accessMap.begin(), accessMap.end(), std::bind(layoutFinder, std::placeholders::_1, newLayout));
        if (newAccess == accessMap.end())
        {
            throw std::runtime_error("new image layout is unsupported");
        }

        vk::ImageMemoryBarrier result;
        result.setSrcAccessMask(oldAccess->second)
                .setDstAccessMask(newAccess->second)
                .setOldLayout(mImageLayout)
                .setNewLayout(newLayout)
                .setImage(*mImage);
        result.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLevelCount(1)
                .setLayerCount(1);

        mImageLayout = newLayout;

        return result;
    }

    staging_buffer::staging_buffer()
            : mDevice(),
              mImage(),
              mWidth(0),
              mHeight(0),
              mDeviceMemory(),
              mBuffer(),
              mMapped(false)
    {
        // this space intentionally left blank
    }

    void staging_buffer::swap(staging_buffer& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mImage, other.mImage);
        swap(mWidth, other.mWidth);
        swap(mHeight, other.mHeight);
        swap(mDeviceMemory, other.mDeviceMemory);
        swap(mBuffer, other.mBuffer);
        swap(mMapped, other.mMapped);
    }


    staging_buffer::staging_buffer(vk::Device                           device,
                                   vk::PhysicalDeviceMemoryProperties   memoryProperties,
                                   image*                               image,
                                   uint32_t                             width,
                                   uint32_t                             height,
                                   std::size_t                          pixelSize)
            : staging_buffer()
    {
        mDevice = device;
        mImage = image;
        mWidth = width;
        mHeight = height;

        const std::size_t num_bytes = pixelSize * width * height;

        // Allocate the buffer
        vk::BufferCreateInfo buf_info;
        buf_info.setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc)
                .setSize(num_bytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        mBuffer = mDevice.createBufferUnique(buf_info);

        mDeviceMemory = allocate_device_memory(mDevice,
                                               mDevice.getBufferMemoryRequirements(*mBuffer),
                                               memoryProperties,
                                               vk::MemoryPropertyFlagBits::eHostVisible);

        // Bind the memory to the buffer object
        mDevice.bindBufferMemory(*mBuffer, *mDeviceMemory, 0);
    }

    staging_buffer::staging_buffer(staging_buffer&& other)
            : staging_buffer()
    {
        swap(other);
    }

    staging_buffer::~staging_buffer() {
    }

    staging_buffer& staging_buffer::operator=(staging_buffer&& other)
    {
        swap(other);
        return *this;
    }

    std::unique_ptr<void, staging_buffer::unmapper_t> staging_buffer::map()
    {
        if (mMapped) {
            throw std::runtime_error("staging_buffer is already mapped");
        }

        void* memMap = mDevice.mapMemory(*mDeviceMemory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
        mMapped = true;

        const vk::MappedMemoryRange mappedRange(*mDeviceMemory, 0, VK_WHOLE_SIZE);
        mDevice.invalidateMappedMemoryRanges(mappedRange);

        return std::unique_ptr<void, staging_buffer::unmapper_t>(memMap, unmapper_t(this));
    }

    void staging_buffer::unmap()
    {
        if (!mMapped) {
            throw std::runtime_error("staging_buffer is not mapped");
        }

        const vk::MappedMemoryRange mappedRange(*mDeviceMemory, 0, VK_WHOLE_SIZE);
        mDevice.flushMappedMemoryRanges(mappedRange);

        mDevice.unmapMemory(*mDeviceMemory);
        mMapped = false;
    }

    void staging_buffer::copyToImage(vk::CommandBuffer commandBuffer)
    {
        vk::BufferMemoryBarrier bufferBarrier;
        bufferBarrier.setSrcAccessMask(vk::AccessFlagBits::eHostWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
                .setBuffer(*mBuffer)
                .setSize(VK_WHOLE_SIZE);

        vk::ImageMemoryBarrier imageBarrier = mImage->prepare(vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(mWidth)
                .setBufferImageHeight(mHeight)
                .setImageExtent(vk::Extent3D(mWidth, mHeight, 1));
        copyRegion.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlags(),
                                      nullptr,         // memory barriers
                                      bufferBarrier,   // buffer memory barriers
                                      imageBarrier);   // image memory barriers

        commandBuffer.copyBufferToImage(*mBuffer, imageBarrier.image, imageBarrier.newLayout, copyRegion);
    }

    void staging_buffer::copyFromImage(vk::CommandBuffer commandBuffer)
    {
        vk::BufferMemoryBarrier bufferBarrier;
        bufferBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eShaderRead)
                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setBuffer(*mBuffer)
                .setSize(VK_WHOLE_SIZE);

        vk::ImageMemoryBarrier imageBarrier = mImage->prepare(vk::ImageLayout::eTransferSrcOptimal);

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(mWidth)
                .setBufferImageHeight(mHeight)
                .setImageExtent(vk::Extent3D(mWidth, mHeight, 1));
        copyRegion.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlags(),
                                      nullptr,         // memory barriers
                                      bufferBarrier,   // buffer memory barriers
                                      imageBarrier);   // image memory barriers

        commandBuffer.copyImageToBuffer(imageBarrier.image, imageBarrier.newLayout, *mBuffer, copyRegion);
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
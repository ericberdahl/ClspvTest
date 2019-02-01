//
// Created by Eric Berdahl on 10/22/17.
//

#include "vulkan_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

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

namespace {
    const std::map<VkFormat, std::size_t> kFormatSizeTable = {
            {VK_FORMAT_UNDEFINED,                   3},
            {VK_FORMAT_R4G4_UNORM_PACK8,            1},
            {VK_FORMAT_R4G4B4A4_UNORM_PACK16,       2},
            {VK_FORMAT_B4G4R4A4_UNORM_PACK16,       2},
            {VK_FORMAT_R5G6B5_UNORM_PACK16,         2},
            {VK_FORMAT_B5G6R5_UNORM_PACK16,         2},
            {VK_FORMAT_R5G5B5A1_UNORM_PACK16,       2},
            {VK_FORMAT_B5G5R5A1_UNORM_PACK16,       2},
            {VK_FORMAT_A1R5G5B5_UNORM_PACK16,       2},
            {VK_FORMAT_R8_UNORM,                    1},
            {VK_FORMAT_R8_SNORM,                    1},
            {VK_FORMAT_R8_USCALED,                  1},
            {VK_FORMAT_R8_SSCALED,                  1},
            {VK_FORMAT_R8_UINT,                     1},
            {VK_FORMAT_R8_SINT,                     1},
            {VK_FORMAT_R8_SRGB,                     1},
            {VK_FORMAT_R8G8_UNORM,                  2},
            {VK_FORMAT_R8G8_SNORM,                  2},
            {VK_FORMAT_R8G8_USCALED,                2},
            {VK_FORMAT_R8G8_SSCALED,                2},
            {VK_FORMAT_R8G8_UINT,                   2},
            {VK_FORMAT_R8G8_SINT,                   2},
            {VK_FORMAT_R8G8_SRGB,                   2},
            {VK_FORMAT_R8G8B8_UNORM,                3},
            {VK_FORMAT_R8G8B8_SNORM,                3},
            {VK_FORMAT_R8G8B8_USCALED,              3},
            {VK_FORMAT_R8G8B8_SSCALED,              3},
            {VK_FORMAT_R8G8B8_UINT,                 3},
            {VK_FORMAT_R8G8B8_SINT,                 3},
            {VK_FORMAT_R8G8B8_SRGB,                 3},
            {VK_FORMAT_B8G8R8_UNORM,                3},
            {VK_FORMAT_B8G8R8_SNORM,                3},
            {VK_FORMAT_B8G8R8_USCALED,              3},
            {VK_FORMAT_B8G8R8_SSCALED,              3},
            {VK_FORMAT_B8G8R8_UINT,                 3},
            {VK_FORMAT_B8G8R8_SINT,                 3},
            {VK_FORMAT_B8G8R8_SRGB,                 3},
            {VK_FORMAT_R8G8B8A8_UNORM,              4},
            {VK_FORMAT_R8G8B8A8_SNORM,              4},
            {VK_FORMAT_R8G8B8A8_USCALED,            4},
            {VK_FORMAT_R8G8B8A8_SSCALED,            4},
            {VK_FORMAT_R8G8B8A8_UINT,               4},
            {VK_FORMAT_R8G8B8A8_SINT,               4},
            {VK_FORMAT_R8G8B8A8_SRGB,               4},
            {VK_FORMAT_B8G8R8A8_UNORM,              4},
            {VK_FORMAT_B8G8R8A8_SNORM,              4},
            {VK_FORMAT_B8G8R8A8_USCALED,            4},
            {VK_FORMAT_B8G8R8A8_SSCALED,            4},
            {VK_FORMAT_B8G8R8A8_UINT,               4},
            {VK_FORMAT_B8G8R8A8_SINT,               4},
            {VK_FORMAT_B8G8R8A8_SRGB,               4},
            {VK_FORMAT_A8B8G8R8_UNORM_PACK32,       4},
            {VK_FORMAT_A8B8G8R8_SNORM_PACK32,       4},
            {VK_FORMAT_A8B8G8R8_USCALED_PACK32,     4},
            {VK_FORMAT_A8B8G8R8_SSCALED_PACK32,     4},
            {VK_FORMAT_A8B8G8R8_UINT_PACK32,        4},
            {VK_FORMAT_A8B8G8R8_SINT_PACK32,        4},
            {VK_FORMAT_A8B8G8R8_SRGB_PACK32,        4},
            {VK_FORMAT_A2R10G10B10_UNORM_PACK32,    4},
            {VK_FORMAT_A2R10G10B10_SNORM_PACK32,    4},
            {VK_FORMAT_A2R10G10B10_USCALED_PACK32,  4},
            {VK_FORMAT_A2R10G10B10_SSCALED_PACK32,  4},
            {VK_FORMAT_A2R10G10B10_UINT_PACK32,     4},
            {VK_FORMAT_A2R10G10B10_SINT_PACK32,     4},
            {VK_FORMAT_A2B10G10R10_UNORM_PACK32,    4},
            {VK_FORMAT_A2B10G10R10_SNORM_PACK32,    4},
            {VK_FORMAT_A2B10G10R10_USCALED_PACK32,  4},
            {VK_FORMAT_A2B10G10R10_SSCALED_PACK32,  4},
            {VK_FORMAT_A2B10G10R10_UINT_PACK32,     4},
            {VK_FORMAT_A2B10G10R10_SINT_PACK32,     4},
            {VK_FORMAT_R16_UNORM,                   2},
            {VK_FORMAT_R16_SNORM,                   2},
            {VK_FORMAT_R16_USCALED,                 2},
            {VK_FORMAT_R16_SSCALED,                 2},
            {VK_FORMAT_R16_UINT,                    2},
            {VK_FORMAT_R16_SINT,                    2},
            {VK_FORMAT_R16_SFLOAT,                  2},
            {VK_FORMAT_R16G16_UNORM,                4},
            {VK_FORMAT_R16G16_SNORM,                4},
            {VK_FORMAT_R16G16_USCALED,              4},
            {VK_FORMAT_R16G16_SSCALED,              4},
            {VK_FORMAT_R16G16_UINT,                 4},
            {VK_FORMAT_R16G16_SINT,                 4},
            {VK_FORMAT_R16G16_SFLOAT,               4},
            {VK_FORMAT_R16G16B16_UNORM,             6},
            {VK_FORMAT_R16G16B16_SNORM,             6},
            {VK_FORMAT_R16G16B16_USCALED,           6},
            {VK_FORMAT_R16G16B16_SSCALED,           6},
            {VK_FORMAT_R16G16B16_UINT,              6},
            {VK_FORMAT_R16G16B16_SINT,              6},
            {VK_FORMAT_R16G16B16_SFLOAT,            6},
            {VK_FORMAT_R16G16B16A16_UNORM,          8},
            {VK_FORMAT_R16G16B16A16_SNORM,          8},
            {VK_FORMAT_R16G16B16A16_USCALED,        8},
            {VK_FORMAT_R16G16B16A16_SSCALED,        8},
            {VK_FORMAT_R16G16B16A16_UINT,           8},
            {VK_FORMAT_R16G16B16A16_SINT,           8},
            {VK_FORMAT_R16G16B16A16_SFLOAT,         8},
            {VK_FORMAT_R32_UINT,                    4},
            {VK_FORMAT_R32_SINT,                    4},
            {VK_FORMAT_R32_SFLOAT,                  4},
            {VK_FORMAT_R32G32_UINT,                 8},
            {VK_FORMAT_R32G32_SINT,                 8},
            {VK_FORMAT_R32G32_SFLOAT,               8},
            {VK_FORMAT_R32G32B32_UINT,              12},
            {VK_FORMAT_R32G32B32_SINT,              12},
            {VK_FORMAT_R32G32B32_SFLOAT,            12},
            {VK_FORMAT_R32G32B32A32_UINT,           16},
            {VK_FORMAT_R32G32B32A32_SINT,           16},
            {VK_FORMAT_R32G32B32A32_SFLOAT,         16},
            {VK_FORMAT_R64_UINT,                    8},
            {VK_FORMAT_R64_SINT,                    8},
            {VK_FORMAT_R64_SFLOAT,                  8},
            {VK_FORMAT_R64G64_UINT,                 16},
            {VK_FORMAT_R64G64_SINT,                 16},
            {VK_FORMAT_R64G64_SFLOAT,               16},
            {VK_FORMAT_R64G64B64_UINT,              24},
            {VK_FORMAT_R64G64B64_SINT,              24},
            {VK_FORMAT_R64G64B64_SFLOAT,            24},
            {VK_FORMAT_R64G64B64A64_UINT,           32},
            {VK_FORMAT_R64G64B64A64_SINT,           32},
            {VK_FORMAT_R64G64B64A64_SFLOAT,         32},
            {VK_FORMAT_B10G11R11_UFLOAT_PACK32,     4},
            {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,      4},
            {VK_FORMAT_D16_UNORM,                   2},
            {VK_FORMAT_X8_D24_UNORM_PACK32,         4},
            {VK_FORMAT_D32_SFLOAT,                  4},
            {VK_FORMAT_S8_UINT,                     1},
            {VK_FORMAT_D16_UNORM_S8_UINT,           3},
            {VK_FORMAT_D24_UNORM_S8_UINT,           4},
            {VK_FORMAT_D32_SFLOAT_S8_UINT,          8},
            {VK_FORMAT_BC1_RGB_UNORM_BLOCK,         8},
            {VK_FORMAT_BC1_RGB_SRGB_BLOCK,          8},
            {VK_FORMAT_BC1_RGBA_UNORM_BLOCK,        8},
            {VK_FORMAT_BC1_RGBA_SRGB_BLOCK,         8},
            {VK_FORMAT_BC2_UNORM_BLOCK,             16},
            {VK_FORMAT_BC2_SRGB_BLOCK,              16},
            {VK_FORMAT_BC3_UNORM_BLOCK,             16},
            {VK_FORMAT_BC3_SRGB_BLOCK,              16},
            {VK_FORMAT_BC4_UNORM_BLOCK,             8},
            {VK_FORMAT_BC4_SNORM_BLOCK,             8},
            {VK_FORMAT_BC5_UNORM_BLOCK,             16},
            {VK_FORMAT_BC5_SNORM_BLOCK,             16},
            {VK_FORMAT_BC6H_UFLOAT_BLOCK,           16},
            {VK_FORMAT_BC6H_SFLOAT_BLOCK,           16},
            {VK_FORMAT_BC7_UNORM_BLOCK,             16},
            {VK_FORMAT_BC7_SRGB_BLOCK,              16},
            {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,     8},
            {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,      8},
            {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,   8},
            {VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,    8},
            {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,   16},
            {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,    8},
            {VK_FORMAT_EAC_R11_UNORM_BLOCK,         8},
            {VK_FORMAT_EAC_R11_SNORM_BLOCK,         8},
            {VK_FORMAT_EAC_R11G11_UNORM_BLOCK,      16},
            {VK_FORMAT_EAC_R11G11_SNORM_BLOCK,      16},
            {VK_FORMAT_ASTC_4x4_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_4x4_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_5x4_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_5x4_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_5x5_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_5x5_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_6x5_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_6x5_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_6x6_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_6x6_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_8x5_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_8x5_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_8x6_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_8x6_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_8x8_UNORM_BLOCK,        16},
            {VK_FORMAT_ASTC_8x8_SRGB_BLOCK,         16},
            {VK_FORMAT_ASTC_10x5_UNORM_BLOCK,       16},
            {VK_FORMAT_ASTC_10x5_SRGB_BLOCK,        16},
            {VK_FORMAT_ASTC_10x6_UNORM_BLOCK,       16},
            {VK_FORMAT_ASTC_10x6_SRGB_BLOCK,        16},
            {VK_FORMAT_ASTC_10x8_UNORM_BLOCK,       16},
            {VK_FORMAT_ASTC_10x8_SRGB_BLOCK,        16},
            {VK_FORMAT_ASTC_10x10_UNORM_BLOCK,      16},
            {VK_FORMAT_ASTC_10x10_SRGB_BLOCK,       16},
            {VK_FORMAT_ASTC_12x10_UNORM_BLOCK,      16},
            {VK_FORMAT_ASTC_12x10_SRGB_BLOCK,       16},
            {VK_FORMAT_ASTC_12x12_UNORM_BLOCK,      16},
            {VK_FORMAT_ASTC_12x12_SRGB_BLOCK,       16},
            {VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG, 8},
            {VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG, 8},
            {VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG, 8},
            {VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG, 8},
            {VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,  8},
            {VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,  8},
            {VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,  8},
            {VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,  8},
    };

    const vk::MemoryType* find_compatible_memory(const vk::MemoryType*    first,
                                                 const vk::MemoryType*    last,
                                                 std::uint32_t            typeBits,
                                                 vk::MemoryPropertyFlags  requirements_mask)
    {
        // Search the sequence to find the first MemoryType with the indicated properties
        for (; first != last; ++first)
        {
            if ((typeBits & 1) == 1)
            {
                // Type is available, does it match user properties?
                if ((first->propertyFlags & requirements_mask) == requirements_mask)
                {
                    return first;
                }
            }
            typeBits >>= 1;
        }

        return last;
    }

    vk::BufferMemoryBarrier prepare_buffer_for_read(vk::Buffer buf)
    {
        vk::BufferMemoryBarrier result;

        result.setSrcAccessMask(vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead)
                .setSize(VK_WHOLE_SIZE)
                .setBuffer(buf);

        return result;
    }

    vk::BufferMemoryBarrier prepare_buffer_for_write(vk::Buffer buf)
    {
        vk::BufferMemoryBarrier result;

        result.setSrcAccessMask(vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eShaderRead)
                .setDstAccessMask(vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite)
                .setSize(VK_WHOLE_SIZE)
                .setBuffer(buf);

        return result;
    }

    void fail_runtime_error(const char* what)
    {
        throw std::runtime_error(what);
    }
}

namespace vulkan_utils {

    vk::UniqueDeviceMemory allocate_device_memory(vk::Device                                device,
                                                  const vk::MemoryRequirements&             mem_reqs,
                                                  const vk::PhysicalDeviceMemoryProperties& mem_props,
                                                  vk::MemoryPropertyFlags                   property_flags)
    {
        auto last = mem_props.memoryTypes + mem_props.memoryTypeCount;
        auto found = find_compatible_memory(mem_props.memoryTypes, last, mem_reqs.memoryTypeBits, property_flags);
        if (found == last)
        {
            fail_runtime_error("No mappable device memory");
        }

        // Allocate memory for the buffer
        vk::MemoryAllocateInfo alloc_info;
        alloc_info.setAllocationSize(mem_reqs.size)
                .setMemoryTypeIndex(std::distance(mem_props.memoryTypes, found));
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

    void device_memory::bind(vk::Buffer buffer, vk::DeviceSize memoryOffset)
    {
        mDevice.bindBufferMemory(buffer, *mMemory, memoryOffset);
    }

    void device_memory::bind(vk::Image image, vk::DeviceSize memoryOffset)
    {
        mDevice.bindImageMemory(image, *mMemory, memoryOffset);
    }

    std::unique_ptr<void, device_memory::unmapper_t> device_memory::map()
    {
        if (mMapped) {
            fail_runtime_error("device_memory is already mapped");
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
            fail_runtime_error("device_memory is not mapped");
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

    vk::BufferMemoryBarrier uniform_buffer::prepareForRead()
    {
        return prepare_buffer_for_read(*buf);
    }

    vk::DescriptorBufferInfo uniform_buffer::use()
    {
        vk::DescriptorBufferInfo result;
        result.setRange(VK_WHOLE_SIZE)
                .setBuffer(*buf);
        return result;
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

    vk::BufferMemoryBarrier storage_buffer::prepareForRead()
    {
        return prepare_buffer_for_read(*buf);
    }

    vk::BufferMemoryBarrier storage_buffer::prepareForWrite()
    {
        return prepare_buffer_for_write(*buf);
    }

    vk::DescriptorBufferInfo storage_buffer::use()
    {
        vk::DescriptorBufferInfo result;
        result.setRange(VK_WHOLE_SIZE)
                .setBuffer(*buf);
        return result;
    }

    image::image()
            : mDevice(),
              mMemoryProperties(),
              mImageLayout(vk::ImageLayout::eUndefined),
              mDeviceMemory(),
              mExtent(),
              mImage(),
              mImageView(),
              mFormat(vk::Format::eUndefined)
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
        swap(mExtent, other.mExtent);
        swap(mImage, other.mImage);
        swap(mImageView, other.mImageView);
        swap(mFormat, other.mFormat);
    }

    bool image::supportsFormatUse(vk::PhysicalDevice device, vk::Format format, Usage usage)
    {
        vk::FormatProperties properties = device.getFormatProperties(format);

        vk::FormatFeatureFlags requiredFeatures = vk::FormatFeatureFlagBits::eSampledImage;
        if (kUsage_ReadWrite == usage)
        {
            requiredFeatures |= vk::FormatFeatureFlagBits::eStorageImage;
        }

        return (requiredFeatures == (properties.optimalTilingFeatures & requiredFeatures));
    }

    image::image(vk::Device                                 dev,
                 const vk::PhysicalDeviceMemoryProperties   memoryProperties,
                 vk::Extent3D                               extent,
                 vk::Format                                 format,
                 Usage                                      usage)
            : image()
    {
        if (extent.width < 1 || extent.height < 1 || extent.depth < 1)
        {
            fail_runtime_error("Invalid extent for image -- at least one dimension is 0");
        }

        const bool is3D = (extent.depth > 1);

        mDevice = dev;
        mMemoryProperties = memoryProperties;
        mExtent = extent;
        mFormat = format;

        vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eSampled |
                                         vk::ImageUsageFlagBits::eTransferDst |
                                         vk::ImageUsageFlagBits::eTransferSrc;
        if (kUsage_ReadWrite == usage)
        {
            imageUsage |= vk::ImageUsageFlagBits::eStorage;
        }

        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(is3D ? vk::ImageType::e3D : vk::ImageType::e2D)
                .setFormat(mFormat)
                .setExtent(mExtent)
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setTiling(vk::ImageTiling::eOptimal)
                .setUsage(imageUsage)
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
                .setViewType(is3D ? vk::ImageViewType::e3D : vk::ImageViewType::e2D)
                .setFormat(mFormat)
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
        auto found = kFormatSizeTable.find((VkFormat)mFormat);
        if (found == kFormatSizeTable.end()) {
            fail_runtime_error("cannot map image format to pixel size");
        }
        if (0 == found->second) {
            fail_runtime_error("image format pixels are not a knowable size");
        }

        return staging_buffer(mDevice,
                              mMemoryProperties,
                              this,
                              mExtent,
                              found->second);
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
            fail_runtime_error("images cannot be transitioned to undefined layout");
        }

        // TODO: if the layout isn't changing, no barrier is needed

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
            fail_runtime_error("new image layout is unsupported");
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
              mStorageBuffer(),
              mExtent(0)
    {
        // this space intentionally left blank
    }

    void staging_buffer::swap(staging_buffer& other)
    {
        using std::swap;

        swap(mDevice, other.mDevice);
        swap(mImage, other.mImage);
        swap(mStorageBuffer, other.mStorageBuffer);
        swap(mExtent, other.mExtent);
    }


    staging_buffer::staging_buffer(vk::Device                           device,
                                   vk::PhysicalDeviceMemoryProperties   memoryProperties,
                                   image*                               image,
                                   vk::Extent3D                         extent,
                                   std::size_t                          pixelSize)
            : staging_buffer()
    {
        mDevice = device;
        mImage = image;
        mExtent = extent;


        const std::size_t num_bytes = pixelSize * mExtent.width * mExtent.height * mExtent.depth;

        mStorageBuffer = storage_buffer(device, memoryProperties, num_bytes);
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

    void staging_buffer::copyToImage(vk::CommandBuffer commandBuffer)
    {
        vk::BufferMemoryBarrier bufferBarrier = mStorageBuffer.prepareForRead();
        vk::ImageMemoryBarrier imageBarrier = mImage->prepare(vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(mExtent.width)
                .setBufferImageHeight(mExtent.height)
                .setImageExtent(mExtent);
        copyRegion.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlags(),
                                      nullptr,         // memory barriers
                                      bufferBarrier,   // buffer memory barriers
                                      imageBarrier);   // image memory barriers

        commandBuffer.copyBufferToImage(bufferBarrier.buffer, imageBarrier.image, imageBarrier.newLayout, copyRegion);
    }

    void staging_buffer::copyFromImage(vk::CommandBuffer commandBuffer)
    {
        vk::BufferMemoryBarrier bufferBarrier = mStorageBuffer.prepareForWrite();
        vk::ImageMemoryBarrier imageBarrier = mImage->prepare(vk::ImageLayout::eTransferSrcOptimal);

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(mExtent.width)
                .setBufferImageHeight(mExtent.height)
                .setImageExtent(mExtent);
        copyRegion.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlags(),
                                      nullptr,         // memory barriers
                                      bufferBarrier,   // buffer memory barriers
                                      imageBarrier);   // image memory barriers

        commandBuffer.copyImageToBuffer(imageBarrier.image, imageBarrier.newLayout, bufferBarrier.buffer, copyRegion);
    }

    double timestamp_delta_ns(std::uint64_t                         startTimestamp,
                              std::uint64_t                         endTimestamp,
                              const vk::PhysicalDeviceProperties&   deviceProperties,
                              const vk::QueueFamilyProperties&      queueFamilyProperties) {
        std::uint64_t timestampDelta;
        if (endTimestamp >= startTimestamp) {
            timestampDelta = endTimestamp - startTimestamp;
        }
        else {
            const std::uint64_t maxTimestamp = std::numeric_limits<std::uint64_t>::max() >> (64 - queueFamilyProperties.timestampValidBits);
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
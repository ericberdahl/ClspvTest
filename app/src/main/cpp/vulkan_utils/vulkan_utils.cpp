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
        vk::UniqueDeviceMemory result;

        const auto last = mem_props.memoryTypes + mem_props.memoryTypeCount;
        const auto found = find_compatible_memory(mem_props.memoryTypes, last, mem_reqs.memoryTypeBits, property_flags);
        if (found != last)
        {
            // Allocate memory for the buffer
            vk::MemoryAllocateInfo alloc_info;
            alloc_info.setAllocationSize(mem_reqs.size)
                    .setMemoryTypeIndex(std::distance(mem_props.memoryTypes, found));
            result = device.allocateMemoryUnique(alloc_info);
        }

        return result;
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

    buffer createUniformBuffer(vk::Device device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               vk::DeviceSize                           num_bytes)
    {
        return buffer(device,
                      memoryProperties,
                      num_bytes,
                      vk::BufferUsageFlagBits::eUniformBuffer);
    }

    buffer createStorageBuffer(vk::Device device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               vk::DeviceSize                           num_bytes)
    {
        return buffer(device,
                      memoryProperties,
                      num_bytes,
                      vk::BufferUsageFlagBits::eStorageBuffer);
    }

    buffer createStagingBuffer(vk::Device                               device,
                               const vk::PhysicalDeviceMemoryProperties memoryProperties,
                               const image&                             image,
                               bool                                     isForInitialzation,
                               bool                                     isForReadback)
    {
        const auto found = kFormatSizeTable.find((VkFormat)image.getFormat());
        if (found == kFormatSizeTable.end()) {
            fail_runtime_error("cannot map image format to pixel size");
        }
        if (0 == found->second) {
            fail_runtime_error("image format pixels are not a knowable size");
        }

        const auto extent = image.getExtent();
        const std::size_t num_bytes = found->second * extent.width * extent.height * extent.depth;

        const vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eStorageBuffer
                                              | (isForInitialzation ? vk::BufferUsageFlagBits::eTransferSrc : vk::BufferUsageFlagBits())
                                              | (isForReadback ? vk::BufferUsageFlagBits::eTransferDst : vk::BufferUsageFlagBits());

        return buffer(device,
                      memoryProperties,
                      num_bytes,
                      usageFlags);
    }

    buffer::buffer(vk::Device                               device,
                   const vk::PhysicalDeviceMemoryProperties memoryProperties,
                   vk::DeviceSize                           num_bytes,
                   vk::BufferUsageFlags                     usage) :
            buffer()
    {
        mUsage = usage;
        mDevice = device;

        // Allocate the buffer
        vk::BufferCreateInfo buf_info;
        buf_info.setUsage(mUsage)
                .setSize(num_bytes)
                .setSharingMode(vk::SharingMode::eExclusive);

        mBuffer = mDevice.createBufferUnique(buf_info);

        const auto memReqs = mDevice.getBufferMemoryRequirements(*mBuffer);
        mDeviceMemory = allocate_device_memory(mDevice,
                                               memReqs,
                                               memoryProperties,
                                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached);

        if (!mDeviceMemory)
        {
            mDeviceMemory = allocate_device_memory(mDevice,
                                                   memReqs,
                                                   memoryProperties,
                                                   vk::MemoryPropertyFlagBits::eHostVisible);
        }

        if (!mDeviceMemory)
        {
            fail_runtime_error("Cannot allocate device memory");
        }

        // Bind the memory to the buffer object
        mDevice.bindBufferMemory(*mBuffer, *mDeviceMemory, 0);
    }

    buffer::buffer(buffer&& other) :
            buffer()
    {
        swap(other);
    }

    buffer::~buffer() {
    }

    buffer& buffer::operator=(buffer&& other)
    {
        swap(other);
        return *this;
    }

    void buffer::swap(buffer& other)
    {
        using std::swap;

        swap(mUsage, other.mUsage);
        swap(mIsMapped, other.mIsMapped);

        swap(mDevice, other.mDevice);
        swap(mDeviceMemory, other.mDeviceMemory);
        swap(mBuffer, other.mBuffer);
    }

    vk::BufferMemoryBarrier buffer::prepareForShaderRead()
    {
        vk::BufferMemoryBarrier result;

        if (!(mUsage & (vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer)))
        {
            fail_runtime_error("buffer was not constructed as either a storage or uniform buffer");
        }

        result.setSrcAccessMask(vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eShaderWrite)
              .setSize(VK_WHOLE_SIZE)
              .setBuffer(*mBuffer);

        if (mUsage & vk::BufferUsageFlagBits::eTransferDst)
            result.srcAccessMask |= vk::AccessFlagBits::eTransferWrite;

        if (mUsage & vk::BufferUsageFlagBits::eUniformBuffer)
            result.dstAccessMask |= vk::AccessFlagBits::eUniformRead;
        if (mUsage & vk::BufferUsageFlagBits::eStorageBuffer)
            result.dstAccessMask |= vk::AccessFlagBits::eShaderRead;

        return result;
    }

    vk::BufferMemoryBarrier buffer::prepareForShaderWrite()
    {
        vk::BufferMemoryBarrier result;

        if (!(mUsage & vk::BufferUsageFlagBits::eStorageBuffer))
        {
            fail_runtime_error("buffer was not constructed as a storage buffer");
        }

        result.setSrcAccessMask(vk::AccessFlagBits::eShaderRead)
                .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setSize(VK_WHOLE_SIZE)
                .setBuffer(*mBuffer);

        if (mUsage & vk::BufferUsageFlagBits::eTransferSrc)
            result.srcAccessMask |= (vk::AccessFlagBits::eTransferRead);

        return result;
    }

    vk::BufferMemoryBarrier buffer::prepareForTransferSrc()
    {
        if (!(mUsage & vk::BufferUsageFlagBits::eTransferSrc))
        {
            fail_runtime_error("buffer was not constructed as a potential transfer source");
        }

        vk::BufferMemoryBarrier result;
        result.setSrcAccessMask(vk::AccessFlagBits::eHostWrite)
              .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
              .setSize(VK_WHOLE_SIZE)
              .setBuffer(*mBuffer);

        if (mUsage & (vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer))
            result.srcAccessMask |= vk::AccessFlagBits::eShaderWrite;
        if (mUsage & vk::BufferUsageFlagBits::eTransferDst)
            result.srcAccessMask |= vk::AccessFlagBits::eTransferWrite;

        return result;
    }

    vk::BufferMemoryBarrier buffer::prepareForTransferDst()
    {
        if (!(mUsage & vk::BufferUsageFlagBits::eTransferDst))
        {
            fail_runtime_error("buffer was not constructed as a potential transfer destination");
        }

        vk::BufferMemoryBarrier result;
        result.setSrcAccessMask(vk::AccessFlagBits::eHostRead)
              .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
              .setSize(VK_WHOLE_SIZE)
              .setBuffer(*mBuffer);

        if (mUsage & (vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer))
            result.srcAccessMask |= vk::AccessFlagBits::eShaderRead;
        if (mUsage & vk::BufferUsageFlagBits::eTransferSrc)
            result.srcAccessMask |= vk::AccessFlagBits::eTransferRead;

        return result;
    }

    vk::DescriptorBufferInfo buffer::use()
    {
        vk::DescriptorBufferInfo result;
        result.setRange(VK_WHOLE_SIZE)
                .setBuffer(*mBuffer);
        return result;
    }

    mapped_ptr<void> buffer::map()
    {
        // TODO check that the memory is host visible

        if (mIsMapped) {
            fail_runtime_error("buffer is already mapped");
        }

        void* memMap = mDevice.mapMemory(*mDeviceMemory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
        mapped_ptr<void> result(memMap, std::bind(&buffer::unmap, this));
        mIsMapped = true;

        // TODO only do cache management on incoherent memory

        const vk::MappedMemoryRange mappedRange(*mDeviceMemory, 0, VK_WHOLE_SIZE);
        mDevice.invalidateMappedMemoryRanges(mappedRange);

        return result;
    }

    void buffer::unmap()
    {
        if (!mIsMapped) {
            fail_runtime_error("buffer is not mapped");
        }

        // TODO only do cache management on incoherent memory

        const vk::MappedMemoryRange mappedRange(*mDeviceMemory, 0, VK_WHOLE_SIZE);
        mDevice.flushMappedMemoryRanges(mappedRange);

        mDevice.unmapMemory(*mDeviceMemory);
        mIsMapped = false;
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
        if (!mDeviceMemory)
        {
            fail_runtime_error("Cannot allocate device memory for image");
        }

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

    vk::Extent3D computeNumberWorkgroups(const vk::Extent3D& workgroupSize, const vk::Extent3D& dataSize)
    {
        const vk::Extent3D result(
                (dataSize.width + workgroupSize.width - 1) / workgroupSize.width,
                (dataSize.height + workgroupSize.height - 1) / workgroupSize.height,
                (dataSize.depth + workgroupSize.depth - 1) / workgroupSize.depth);
        return result;
    }

    void copyBufferToImage(vk::CommandBuffer    commandBuffer,
                           buffer&              buffer,
                           image&               image)
    {
        vk::BufferMemoryBarrier bufferBarrier = buffer.prepareForTransferSrc();
        vk::ImageMemoryBarrier imageBarrier = image.prepare(vk::ImageLayout::eTransferDstOptimal);

        const auto imageExtent = image.getExtent();

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(imageExtent.width)
                  .setBufferImageHeight(imageExtent.height)
                  .setImageExtent(imageExtent);
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

    void copyImageToBuffer(vk::CommandBuffer    commandBuffer,
                           image&               image,
                           buffer&              buffer)
    {
        vk::BufferMemoryBarrier bufferBarrier = buffer.prepareForTransferDst();
        vk::ImageMemoryBarrier imageBarrier = image.prepare(vk::ImageLayout::eTransferSrcOptimal);

        const auto imageExtent = image.getExtent();

        vk::BufferImageCopy copyRegion;
        copyRegion.setBufferRowLength(imageExtent.width)
                  .setBufferImageHeight(imageExtent.height)
                  .setImageExtent(imageExtent);
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

} // namespace vulkan_utils

std::ostream& operator<<(std::ostream& os, vk::MemoryPropertyFlags flags) {
    os << std::hex << std::showbase << (VkMemoryPropertyFlags)flags << vk::to_string(flags);

    return os;
}

std::ostream& operator<<(std::ostream& os, vk::MemoryHeapFlags flags) {
    os << std::hex << std::showbase << (VkMemoryHeapFlags)flags << vk::to_string(flags);

    return os;
}

std::ostream& operator<<(std::ostream& os, const vk::MemoryType& memoryType) {
    os << "heapIndex:" << memoryType.heapIndex << " flags:" << memoryType.propertyFlags;
    return os;
}

std::ostream& operator<<(std::ostream& os, const vk::MemoryHeap& memoryHeap) {
    os << "size:" << memoryHeap.size << " flags:" << memoryHeap.flags;
    return os;
}

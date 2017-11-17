/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "clspv_utils.hpp"
#include "fill_kernel.hpp"
#include "fp_utils.hpp"
#include "gpu_types.hpp"
#include "half.hpp"
#include "opencl_types.hpp"
#include "pixels.hpp"
#include "test_utils.hpp"
#include "util_init.hpp"
#include "vulkan_utils.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <iterator>
#include <string>

#include <vulkan/vulkan.hpp>

/* ============================================================================================== */

VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT        msgFlags,
                                       VkDebugReportObjectTypeEXT   objType,
                                       uint64_t                     srcObject,
                                       size_t                       location,
                                       int32_t                      msgCode,
                                       const char*                  pLayerPrefix,
                                       const char*                  pMsg,
                                       void*                        pUserData) {
    std::ostringstream message;

    if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        message << "PERFORMANCE: ";
    }

    message << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        LOGE("%s", message.str().c_str());
    } else if ((msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) || (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) {
        LOGW("%s", message.str().c_str());
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        LOGI("%s", message.str().c_str());
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        LOGD("%s", message.str().c_str());
    }

    /*
     * false indicates that layer should not bail-out of an
     * API call that had validation failures. This may mean that the
     * app dies inside the driver due to invalid parameter(s).
     * That's what would happen without validation layers, so we'll
     * keep that behavior here.
     */
    return VK_FALSE;
}

/* ============================================================================================== */

void init_compute_queue_family_index(struct sample_info &info) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(info.gpu, &queue_family_count, NULL);
    assert(queue_family_count >= 1);

    std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(info.gpu, &queue_family_count, queue_props.data());
    assert(queue_family_count >= 1);

    /* This routine simply finds a compute queue for a later vkCreateDevice.
     */
    bool found = false;
    for (unsigned int i = 0; i < queue_props.size(); i++) {
        if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            info.graphics_queue_family_index = i;
            found = true;
            break;
        }
    }
    assert(found);
}

void my_init_descriptor_pool(struct sample_info &info) {
    vk::Device device(info.device);

    const vk::DescriptorPoolSize type_count[] = {
        { vk::DescriptorType::eStorageBuffer,   16 },
        { vk::DescriptorType::eSampler,         16 },
        { vk::DescriptorType::eSampledImage,    16 },
        { vk::DescriptorType::eStorageImage,    16 }
    };

    vk::DescriptorPoolCreateInfo createInfo;
    createInfo.setMaxSets(64)
            .setPoolSizeCount(sizeof(type_count) / sizeof(type_count[0]))
            .setPPoolSizes(type_count)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

    info.desc_pool = device.createDescriptorPoolUnique(createInfo);
}

VkSampler create_compatible_sampler(VkDevice device, int opencl_flags) {
    typedef std::pair<int,VkSamplerAddressMode> address_mode_map;
    const address_mode_map address_mode_translator[] = {
            { CLK_ADDRESS_NONE, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
            { CLK_ADDRESS_CLAMP, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER },
            { CLK_ADDRESS_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT },
            { CLK_ADDRESS_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT }
    };

    const VkFilter filter = ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ?
                             VK_FILTER_LINEAR :
                             VK_FILTER_NEAREST);
    const VkBool32 unnormalizedCoordinates = ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_FALSE : VK_TRUE);

    const auto found_map = std::find_if(std::begin(address_mode_translator), std::end(address_mode_translator), [&opencl_flags](const address_mode_map& am) {
        return (am.first == (opencl_flags & CLK_ADDRESS_MASK));
    });
    const VkSamplerAddressMode addressMode = (found_map == std::end(address_mode_translator) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : found_map->second);

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter ;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = addressMode;
    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.unnormalizedCoordinates = unnormalizedCoordinates;

    VkSampler result = VK_NULL_HANDLE;
    vulkan_utils::throwIfNotSuccess(vkCreateSampler(device, &samplerCreateInfo, NULL, &result),
                                    "vkCreateSampler");

    return result;
}

/* ============================================================================================== */

void invoke_copybuffertoimage_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkBuffer  src_buffer,
                                  VkImageView   dst_image,
                                  int src_offset,
                                  int src_pitch,
                                     cl_channel_order src_channel_order,
                                     cl_channel_type src_channel_type,
                                  bool swap_components,
                                  bool premultiply,
                                  int width,
                                  int height) {
    struct scalar_args {
        int inSrcOffset;        // offset 0
        int inSrcPitch;         // offset 4
        int inSrcChannelOrder;  // offset 8 -- cl_channel_order
        int inSrcChannelType;   // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inPremultiply;      // offset 20 -- bool
        int inWidth;            // offset 24
        int inHeight;           // offset 28
    };
    static_assert(0 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inSrcChannelOrder), "inSrcChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inSrcChannelType), "inSrcChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inPremultiply), "inPremultiply offset incorrect");
    static_assert(24 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(28 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            src_offset,
            src_pitch,
            src_channel_order,
            src_channel_type,
            (swap_components ? 1 : 0),
            (premultiply ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addBufferArgument(src_buffer);
    invocation.addWriteOnlyImageArgument(dst_image);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, kernel, num_workgroups);
}

void invoke_copyimagetobuffer_kernel(const clspv_utils::kernel_module&   module,
                                     const clspv_utils::kernel&          kernel,
                                     const sample_info& info,
                                  const std::vector<VkSampler>& samplers,
                                  VkImageView src_image,
                                  VkBuffer dst_buffer,
                                  int dst_offset,
                                  int dst_pitch,
                                     cl_channel_order dst_channel_order,
                                     cl_channel_type dst_channel_type,
                                  bool swap_components,
                                  int width,
                                  int height) {
    struct scalar_args {
        int inDestOffset;       // offset 0
        int inDestPitch;        // offset 4
        int inDestChannelOrder; // offset 8 -- cl_channel_order
        int inDestChannelType;  // offset 12 -- cl_channel_type
        int inSwapComponents;   // offset 16 -- bool
        int inWidth;            // offset 20
        int inHeight;           // offset 24
    };
    static_assert(0 == offsetof(scalar_args, inDestOffset), "inDestOffset offset incorrect");
    static_assert(4 == offsetof(scalar_args, inDestPitch), "inDestPitch offset incorrect");
    static_assert(8 == offsetof(scalar_args, inDestChannelOrder), "inDestChannelOrder offset incorrect");
    static_assert(12 == offsetof(scalar_args, inDestChannelType), "inDestChannelType offset incorrect");
    static_assert(16 == offsetof(scalar_args, inSwapComponents), "inSwapComponents offset incorrect");
    static_assert(20 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
    static_assert(24 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

    const scalar_args scalars = {
            dst_offset,
            width,
            dst_channel_order,
            dst_channel_type,
            (swap_components ? 1 : 0),
            width,
            height
    };

    const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
    const clspv_utils::WorkgroupDimensions num_workgroups(
            (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
            (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addLiteralSamplers(samplers.begin(), samplers.end());
    invocation.addReadOnlyImageArgument(src_image);
    invocation.addBufferArgument(dst_buffer);
    invocation.addPodArgument(scalars);

    invocation.run(info.graphics_queue, kernel, num_workgroups);
}

std::tuple<int,int,int> invoke_localsize_kernel(const clspv_utils::kernel_module&   module,
                                                const clspv_utils::kernel&          kernel,
                                                const sample_info&                  info,
                                                const std::vector<VkSampler>&       samplers) {
    struct scalar_args {
        int outWorkgroupX;  // offset 0
        int outWorkgroupY;  // offset 4
        int outWorkgroupZ;  // offset 8
    };
    static_assert(0 == offsetof(scalar_args, outWorkgroupX), "outWorkgroupX offset incorrect");
    static_assert(4 == offsetof(scalar_args, outWorkgroupY), "outWorkgroupY offset incorrect");
    static_assert(8 == offsetof(scalar_args, outWorkgroupZ), "outWorkgroupZ offset incorrect");

    vulkan_utils::buffer outArgs(info, sizeof(scalar_args));

    // The localsize kernel needs only a single workgroup with a single workitem
    const clspv_utils::WorkgroupDimensions num_workgroups(1, 1);

    clspv_utils::kernel_invocation invocation(info.device, info.cmd_pool, info.memory_properties);

    invocation.addBufferArgument(outArgs.buf);

    invocation.run(info.graphics_queue, kernel, num_workgroups);

    vulkan_utils::memory_map argMap(outArgs);
    auto outScalars = static_cast<const scalar_args*>(argMap.data);

    const auto result = std::make_tuple(outScalars->outWorkgroupX,
                                        outScalars->outWorkgroupY,
                                        outScalars->outWorkgroupZ);
    return result;
}


/* ============================================================================================== */

test_utils::Results test_readlocalsize(const clspv_utils::kernel_module& module,
                                       const clspv_utils::kernel&        kernel,
                                       const sample_info&                info,
                                       const std::vector<VkSampler>&     samplers,
                                       const test_utils::options&        opts) {
    const clspv_utils::WorkgroupDimensions expected = kernel.getWorkgroupSize();

    const auto observed = invoke_localsize_kernel(module, kernel, info, samplers);

    const bool success = (expected.x == std::get<0>(observed) &&
                          expected.y == std::get<1>(observed) &&
                          1 == std::get<2>(observed));

    if (opts.logVerbose && ((success && opts.logCorrect) || (!success && opts.logIncorrect))) {
        const std::string label = module.getName() + "/" + kernel.getEntryPoint();
        LOGE("%s: %s workgroup_size expected{x=%d, y=%d, z=1} observed{x=%d, y=%d, z=%d}",
             success ? "CORRECT" : "INCORRECT",
             label.c_str(),
             expected.x, expected.y,
             std::get<0>(observed), std::get<1>(observed), std::get<2>(observed));
    }

    return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
};

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copytoimage(const clspv_utils::kernel_module&  module,
                                     const clspv_utils::kernel&         kernel,
                                     const sample_info&                 info,
                                     const std::vector<VkSampler>&      samplers,
                                     const test_utils::options&         opts) {
    std::string typeLabel = pixels::traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixels::traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyBufferToImageKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  src_buffer(info, buffer_size);
    vulkan_utils::image   dstImage(info, buffer_width, buffer_height, pixels::traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixels::traits<BufferPixelType>::translate((gpu_types::float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(src_buffer);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixels::traits<ImagePixelType>::translate((gpu_types::float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dstImage);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    invoke_copybuffertoimage_kernel(module, kernel,
                                    info,
                                    samplers,
                                    src_buffer.buf,
                                    dstImage.view,
                                    0,
                                    buffer_width,
                                    pixels::traits<BufferPixelType>::cl_pixel_order,
                                    pixels::traits<BufferPixelType>::cl_pixel_type,
                                    false,
                                    false,
                                    buffer_width,
                                    buffer_height);

    const bool success = test_utils::check_results<BufferPixelType, ImagePixelType>(src_buffer.mem, dstImage.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        opts);

    dstImage.reset();
    src_buffer.reset();

    return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
}

template <typename ImagePixelType>
test_utils::Results test_copytoimage_series(const clspv_utils::kernel_module&   module,
                                            const clspv_utils::kernel&          kernel,
                                            const sample_info&                  info,
                                            const std::vector<VkSampler>&       samplers,
                                            const test_utils::options&          opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copytoimage<gpu_types::uchar, ImagePixelType>,
            test_copytoimage<gpu_types::uchar4, ImagePixelType>,
            test_copytoimage<gpu_types::half, ImagePixelType>,
            test_copytoimage<gpu_types::half4, ImagePixelType>,
            test_copytoimage<float, ImagePixelType>,
            test_copytoimage<gpu_types::float2, ImagePixelType>,
            test_copytoimage<gpu_types::float4, ImagePixelType>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

test_utils::Results test_copytoimage_matrix(const clspv_utils::kernel_module&   module,
                                            const clspv_utils::kernel&          kernel,
                                            const sample_info&                  info,
                                            const std::vector<VkSampler>&       samplers,
                                            const test_utils::options&          opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copytoimage_series<gpu_types::float4>,
            test_copytoimage_series<gpu_types::half4>,
            test_copytoimage_series<gpu_types::uchar4>,
            test_copytoimage_series<gpu_types::float2>,
            test_copytoimage_series<gpu_types::half2>,
            test_copytoimage_series<gpu_types::uchar2>,
            test_copytoimage_series<float>,
            test_copytoimage_series<gpu_types::half>,
            test_copytoimage_series<gpu_types::uchar>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

/* ============================================================================================== */

template <typename BufferPixelType, typename ImagePixelType>
test_utils::Results test_copyfromimage(const clspv_utils::kernel_module&    module,
                                       const clspv_utils::kernel&           kernel,
                                       const sample_info&                   info,
                                       const std::vector<VkSampler>&        samplers,
                                       const test_utils::options&           opts) {
    std::string typeLabel = pixels::traits<BufferPixelType>::type_name;
    typeLabel += '-';
    typeLabel += pixels::traits<ImagePixelType>::type_name;

    std::string testLabel = "memory.spv/CopyImageToBufferKernel/";
    testLabel += typeLabel;

    const int buffer_height = 64;
    const int buffer_width = 64;

    const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

    // allocate buffers and images
    vulkan_utils::buffer  dst_buffer(info, buffer_size);
    vulkan_utils::image   srcImage(info, buffer_width, buffer_height, pixels::traits<ImagePixelType>::vk_pixel_type);

    // initialize source and destination buffers
    {
        auto src_value = pixels::traits<ImagePixelType>::translate((gpu_types::float4){ 0.2f, 0.4f, 0.8f, 1.0f });
        vulkan_utils::memory_map src_map(srcImage);
        auto src_data = static_cast<decltype(src_value)*>(src_map.data);
        std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
    }

    {
        auto dst_value = pixels::traits<BufferPixelType>::translate((gpu_types::float4){ 0.1f, 0.3f, 0.5f, 0.7f });
        vulkan_utils::memory_map dst_map(dst_buffer);
        auto dst_data = static_cast<decltype(dst_value)*>(dst_map.data);
        std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
    }

    invoke_copyimagetobuffer_kernel(module, kernel,
                                    info,
                                    samplers,
                                    srcImage.view,
                                    dst_buffer.buf,
                                    0,
                                    buffer_width,
                                    pixels::traits<BufferPixelType>::cl_pixel_order,
                                    pixels::traits<BufferPixelType>::cl_pixel_type,
                                    false,
                                    buffer_width,
                                    buffer_height);

    const bool success = test_utils::check_results<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem,
                                                                        buffer_width, buffer_height,
                                                                        buffer_height,
                                                                        testLabel.c_str(),
                                                                        opts);

    srcImage.reset();
    dst_buffer.reset();

    return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
}

template <typename ImagePixelType>
test_utils::Results test_copyfromimage_series(const clspv_utils::kernel_module&     module,
                                              const clspv_utils::kernel&            kernel,
                                              const sample_info&                    info,
                                              const std::vector<VkSampler>&         samplers,
                                              const test_utils::options&            opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copyfromimage<gpu_types::uchar, ImagePixelType>,
            test_copyfromimage<gpu_types::uchar4, ImagePixelType>,
            test_copyfromimage<gpu_types::half, ImagePixelType>,
            test_copyfromimage<gpu_types::half4, ImagePixelType>,
            test_copyfromimage<float, ImagePixelType>,
            test_copyfromimage<gpu_types::float2, ImagePixelType>,
            test_copyfromimage<gpu_types::float4, ImagePixelType>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}

test_utils::Results test_copyfromimage_matrix(const clspv_utils::kernel_module&     module,
                                              const clspv_utils::kernel&            kernel,
                                              const sample_info&                    info,
                                              const std::vector<VkSampler>&         samplers,
                                              const test_utils::options&            opts) {
    const test_utils::test_kernel_fn tests[] = {
            test_copyfromimage_series<gpu_types::float4>,
            test_copyfromimage_series<gpu_types::half4>,
            test_copyfromimage_series<gpu_types::uchar4>,
            test_copyfromimage_series<gpu_types::float2>,
            test_copyfromimage_series<gpu_types::half2>,
            test_copyfromimage_series<gpu_types::uchar2>,
            test_copyfromimage_series<float>,
            test_copyfromimage_series<gpu_types::half>,
            test_copyfromimage_series<gpu_types::uchar>,
    };

    return test_utils::test_kernel_invocation(module, kernel,
                                              std::begin(tests), std::end(tests),
                                              info,
                                              samplers,
                                              opts);
}


/* ============================================================================================== */

const test_utils::module_test_bundle module_tests[] = {
        {
                "clspv_tests/localsize",
                {
                        {"ReadLocalSize", test_readlocalsize}
                },
        },
        {
                "clspv_tests/Fills",
                {
                        { "FillWithColorKernel", fill_kernel::test_series, { 32, 32 } }
                }
        },
        {
                "clspv_tests/Memory",
                {
                        { "CopyBufferToImageKernel", test_copytoimage_matrix, { 32, 32 } },
                        { "CopyImageToBufferKernel", test_copyfromimage_matrix, { 32, 32 } }
                }
        },
};

std::vector<test_utils::module_test_bundle> read_loadmodule_file() {
    std::vector<test_utils::module_test_bundle> result;

    std::unique_ptr<std::FILE, decltype(&std::fclose)> spvmap_file(AndroidFopen("loadmodules.txt", "r"),
                                                                   &std::fclose);
    if (spvmap_file) {
        std::fseek(spvmap_file.get(), 0, SEEK_END);
        std::string buffer(std::ftell(spvmap_file.get()), ' ');
        std::fseek(spvmap_file.get(), 0, SEEK_SET);
        std::fread(&buffer.front(), 1, buffer.length(), spvmap_file.get());
        spvmap_file.reset();

        std::istringstream in(buffer);
        while (!in.eof()) {
            std::string line;
            std::getline(in, line);

            std::istringstream in_line(line);

            std::string op;
            in_line >> op;
            if (op.empty() || op[0] == '#') {
                // line is either blank or a comment, skip it
            }
            else if (op == "+") {
                // add module to list of modules to load
                std::string moduleName;
                in_line >> moduleName;
                result.push_back({ moduleName });
            }
            else if (op == "-") {
                // skip kernel in module
                std::string moduleName;
                in_line >> moduleName;

                auto mod_map = std::find_if(result.begin(),
                                            result.end(),
                                            [&moduleName](const test_utils::module_test_bundle& mb) {
                                                return mb.name == moduleName;
                                            });
                if (mod_map != result.end()) {
                    std::string entryPoint;
                    in_line >> entryPoint;

                    mod_map->kernelTests.push_back({ entryPoint, nullptr, clspv_utils::WorkgroupDimensions(0, 0) });
                }
                else {
                    LOGE("read_loadmodule_file: cannot find module '%s' from command '%s'",
                         moduleName.c_str(),
                         line.c_str());
                }
            }
            else {
                LOGE("read_loadmodule_file: ignoring ill-formed line '%s'", line.c_str());
            }
        }
    }

    return result;
}

test_utils::Results run_all_tests(const sample_info& info, const std::vector<VkSampler>& samplers) {
    const test_utils::options opts = {
            false,  // logVerbose
            true,   // logIncorrect
            false   // logCorrect
    };

    test_utils::Results test_results;

    for (auto m : module_tests) {
        test_results += test_utils::test_module(m.name, m.kernelTests, info, samplers, opts);
    }

    auto loadmodule_tests = read_loadmodule_file();
    for (auto m : loadmodule_tests) {
        test_results += test_utils::test_module(m.name, m.kernelTests, info, samplers, opts);
    }

    return test_results;
}

/* ============================================================================================== */

int sample_main(int argc, char *argv[]) {
    struct sample_info info = {};
    init_global_layer_properties(info);

    /* Use standard_validation meta layer that enables all
     * recommended validation layers
     */
    info.instance_layer_names.push_back("VK_LAYER_LUNARG_standard_validation");
    if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
        /* If standard validation is not present, search instead for the
         * individual layers that make it up, in the correct order.
         */
        info.instance_layer_names.clear();
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_threading");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_parameter_validation");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_object_tracker");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_core_validation");
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_unique_objects");

        if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
            LOGE("Cannot find validation layers! :(");
            info.instance_layer_names.clear();
        }
    }

    info.instance_extension_names.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    init_instance(info, "vulkansamples_device");
    init_debug_report_callback(info, dbgFunc);

    init_enumerate_device(info);
    init_compute_queue_family_index(info);

    // The clspv solution we're using requires two Vulkan extensions to be enabled.
    info.device_extension_names.push_back("VK_KHR_storage_buffer_storage_class");
    info.device_extension_names.push_back("VK_KHR_variable_pointers");
    init_device(info);
    init_device_queue(info);

    init_command_pool(info);
    my_init_descriptor_pool(info);

    // This sample presumes that all OpenCL C kernels were compiled with the same samplermap file,
    // whose contents and order are statically known to the application. Thus, the app can create
    // a set of compatible samplers thusly.
    const int sampler_flags[] = {
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_NONE            | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_TRUE
    };
    std::vector<VkSampler> samplers;
    std::transform(std::begin(sampler_flags), std::end(sampler_flags),
                   std::back_inserter(samplers),
                   std::bind(create_compatible_sampler, info.device, std::placeholders::_1));


    const auto test_results = run_all_tests(info, samplers);

    //
    // Clean up
    //

    std::for_each(samplers.begin(), samplers.end(), [&info] (VkSampler s) { vkDestroySampler(info.device, s, nullptr); });

    // Cannot use the shader module desctruction built into the sampel framework because it is too
    // tightly tied to the graphics pipeline (e.g. hard-coding the number and type of shaders).

    info.desc_pool.reset();
    destroy_command_pool(info);
    destroy_device(info);

    LOGI("Complete! %d tests passed. %d tests failed. %d kernels loaded, %d kernels skipped, %d kernels failed",
         test_results.mNumTestSuccess,
         test_results.mNumTestFail,
         test_results.mNumKernelLoadSuccess,
         test_results.mNumKernelLoadSkip,
         test_results.mNumKernelLoadFail);

    return 0;
}


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

#include "memmove_test.hpp"
#include "test_manifest.hpp"
#include "test_result_logging.hpp"
#include "test_utils.hpp"
#include "util_init.hpp"
#include "vulkan_utils/vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

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


void init_validation_layers(struct sample_info& info) {
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
}

void init_compute_queue_family_index(struct sample_info &info) {
    /* This routine simply finds a compute queue for a later vkCreateDevice.
     */

    auto queue_props = info.gpu.getQueueFamilyProperties();

    auto found = std::find_if(queue_props.begin(), queue_props.end(), [](vk::QueueFamilyProperties p) {
        return (p.queueFlags & vk::QueueFlagBits::eCompute);
    });
    assert(found != queue_props.end());

    info.graphics_queue_family_index = std::distance(queue_props.begin(), found);
    info.graphics_queue_family_properties = queue_props[info.graphics_queue_family_index];
}

void my_init_descriptor_pool(struct sample_info &info) {
    const vk::DescriptorPoolSize type_count[] = {
        { vk::DescriptorType::eStorageBuffer,   16 },
        { vk::DescriptorType::eUniformBuffer,   16 },
        { vk::DescriptorType::eSampler,         16 },
        { vk::DescriptorType::eSampledImage,    16 },
        { vk::DescriptorType::eStorageImage,    16 }
    };

    vk::DescriptorPoolCreateInfo createInfo;
    createInfo.setMaxSets(64)
            .setPoolSizeCount(sizeof(type_count) / sizeof(type_count[0]))
            .setPPoolSizes(type_count)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

    info.desc_pool = info.device->createDescriptorPoolUnique(createInfo);
}

void dumpInstanceExtensions()
{
    auto properties = vk::enumerateInstanceExtensionProperties();

    LOGI("instanceExtensionProperties {");
    for (const auto& p : properties)
    {
        LOGI("   %s", p.extensionName);
    }
    LOGI("}");
}

void dumpDeviceExtensions(vk::PhysicalDevice device)
{
    auto properties = device.enumerateDeviceExtensionProperties();

    LOGI("deviceExtensionProperties {");
    for (const auto& p : properties)
    {
        LOGI("   %s", p.extensionName);
    }
    LOGI("}");
}

void dumpDeviceMemoryProperties(vk::PhysicalDevice device)
{
    const auto memProps = device.getMemoryProperties();

    LOGI("physcialDeviceMemoryProperties{");

    LOGI("   memoryTypes {");
    for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        std::ostringstream os;
        os << memProps.memoryTypes[i];
        LOGI("      %s", os.str().c_str());
    }
    LOGI("   }");

    LOGI("   memoryHeaps {");
    for (std::uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        std::ostringstream os;
        os << memProps.memoryHeaps[i];
        LOGI("      %s", os.str().c_str());
    }
    LOGI("   }");

    LOGI("}");
}

/* ============================================================================================== */

int sample_main(int argc, char *argv[]) {
    android_utils::iassetstream is("test_manifest.txt");
    const auto manifest = test_manifest::read(is);
    is.close();

    struct sample_info info = {};
    init_global_layer_properties(info);
    if (manifest.use_validation_layer) {
        init_validation_layers(info);
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

    dumpInstanceExtensions();
    dumpDeviceExtensions(info.gpu);
    dumpDeviceMemoryProperties(info.gpu);

    clspv_utils::device device(info.gpu,
                               *info.device,
                               *info.desc_pool,
                               *info.cmd_pool,
                               info.graphics_queue);

    const auto results = test_manifest::run(manifest, device);
    test_result_logging::logResults(info, results);

    memmove_test::runAllTests(info);

    //
    // Clean up
    //
    device = clspv_utils::device();
    info.desc_pool.reset();
    info.cmd_pool.reset();
    info.device->waitIdle();
    info.device.reset();

    LOGI("ClspvTest complete!!");

    return 0;
}

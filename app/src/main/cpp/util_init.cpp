/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google, Inc.
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

/*
VULKAN_SAMPLE_DESCRIPTION
samples "init" utility functions
*/

#include <cstdlib>
#include <assert.h>
#include <string.h>
#include "util_init.hpp"
#include "util.hpp"

using namespace std;

/*
 * TODO: function description here
 */
void init_global_layer_properties(struct sample_info &info) {
    auto vk_props = vk::enumerateInstanceLayerProperties();

    /*
     * Now gather the extension list for each instance layer.
     */
    for (auto l : vk_props) {
        layer_properties layer_props;

        layer_props.properties = l;
        layer_props.extensions = vk::enumerateInstanceExtensionProperties(std::string(l.layerName));

        info.instance_layer_properties.push_back(layer_props);
    }
}

/*
 * Return 1 (true) if all layer names specified in check_names
 * can be found in given layer properties.
 */
bool demo_check_layers(const std::vector<layer_properties> &layer_props, const std::vector<const char *> &layer_names) {
    for (auto ln : layer_names) {
        const bool found = layer_props.end() != std::find_if(layer_props.begin(), layer_props.end(),
                                                             [ln](const layer_properties& lp) {
                                                                 return 0 == strcmp(ln, lp.properties.layerName);
                                                             });

        if (!found) {
            std::cout << "Cannot find layer: " << ln << std::endl;
            return false;
        }
    }
    return true;
}

VkResult init_instance(struct sample_info &info, char const *const app_short_name) {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = app_short_name;
    app_info.applicationVersion = 1;
    app_info.pEngineName = app_short_name;
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = NULL;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = info.instance_layer_names.size();
    inst_info.ppEnabledLayerNames = info.instance_layer_names.size() ? info.instance_layer_names.data() : NULL;
    inst_info.enabledExtensionCount = info.instance_extension_names.size();
    inst_info.ppEnabledExtensionNames = info.instance_extension_names.data();

    VkResult res = vkCreateInstance(&inst_info, NULL, &info.inst);
    assert(res == VK_SUCCESS);

    return res;
}

VkResult init_device(struct sample_info &info) {
    VkResult res;
    VkDeviceQueueCreateInfo queue_info = {};

    float queue_priorities[1] = {0.0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.pNext = NULL;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priorities;
    queue_info.queueFamilyIndex = info.graphics_queue_family_index;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = NULL;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = info.device_extension_names.size();
    device_info.ppEnabledExtensionNames = device_info.enabledExtensionCount ? info.device_extension_names.data() : NULL;
    device_info.pEnabledFeatures = NULL;

    res = vkCreateDevice(info.gpu, &device_info, NULL, &info.device);
    assert(res == VK_SUCCESS);

    return res;
}

VkResult init_enumerate_device(struct sample_info &info, uint32_t gpu_count) {
    uint32_t const U_ASSERT_ONLY req_count = gpu_count;
    VkResult res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, NULL);
    assert(gpu_count);
    std::vector<VkPhysicalDevice> gpus(gpu_count, VK_NULL_HANDLE);
    assert(gpus.size() == gpu_count);

    res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, gpus.data());
    assert(!res && gpu_count >= req_count);
    info.gpu = gpus[0];

    /* This is as good a place as any to do this */
    vkGetPhysicalDeviceMemoryProperties(info.gpu, &info.memory_properties);

    return res;
}

VkResult init_debug_report_callback(struct sample_info &info, PFN_vkDebugReportCallbackEXT dbgFunc) {
    VkResult res;
    VkDebugReportCallbackEXT debug_report_callback;

    info.dbgCreateDebugReportCallback =
        (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(info.inst, "vkCreateDebugReportCallbackEXT");
    if (!info.dbgCreateDebugReportCallback) {
        std::cout << "GetInstanceProcAddr: Unable to find "
                     "vkCreateDebugReportCallbackEXT function."
                  << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::cout << "Got dbgCreateDebugReportCallback function\n";

    info.dbgDestroyDebugReportCallback =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(info.inst, "vkDestroyDebugReportCallbackEXT");
    if (!info.dbgDestroyDebugReportCallback) {
        std::cout << "GetInstanceProcAddr: Unable to find "
                     "vkDestroyDebugReportCallbackEXT function."
                  << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::cout << "Got dbgDestroyDebugReportCallback function\n";

    VkDebugReportCallbackCreateInfoEXT create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    create_info.pNext = NULL;
    create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    create_info.pfnCallback = dbgFunc;
    create_info.pUserData = NULL;

    res = info.dbgCreateDebugReportCallback(info.inst, &create_info, NULL, &debug_report_callback);
    switch (res) {
        case VK_SUCCESS:
            std::cout << "Successfully created debug report callback object\n";
            info.debug_report_callbacks.push_back(debug_report_callback);
            break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            std::cout << "dbgCreateDebugReportCallback: out of host memory pointer\n" << std::endl;
            return VK_ERROR_INITIALIZATION_FAILED;
            break;
        default:
            std::cout << "dbgCreateDebugReportCallback: unknown failure\n" << std::endl;
            return VK_ERROR_INITIALIZATION_FAILED;
            break;
    }
    return res;
}

void destroy_debug_report_callback(struct sample_info &info) {
    while (info.debug_report_callbacks.size() > 0) {
        info.dbgDestroyDebugReportCallback(info.inst, info.debug_report_callbacks.back(), NULL);
        info.debug_report_callbacks.pop_back();
    }
}

void init_command_pool(struct sample_info &info) {
    /* DEPENDS on init_swapchain_extension() */
    VkResult U_ASSERT_ONLY res;

    VkCommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.pNext = NULL;
    cmd_pool_info.queueFamilyIndex = info.graphics_queue_family_index;
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    res = vkCreateCommandPool(info.device, &cmd_pool_info, NULL, &info.cmd_pool);
    assert(res == VK_SUCCESS);
}

void init_device_queue(struct sample_info &info) {
    /* DEPENDS on init_swapchain_extension() */

    vkGetDeviceQueue(info.device, info.graphics_queue_family_index, 0, &info.graphics_queue);
}

void destroy_descriptor_pool(struct sample_info &info) { vkDestroyDescriptorPool(info.device, info.desc_pool, NULL); }

void destroy_command_pool(struct sample_info &info) { vkDestroyCommandPool(info.device, info.cmd_pool, NULL); }

void destroy_device(struct sample_info &info) {
    vkDeviceWaitIdle(info.device);
    vkDestroyDevice(info.device, NULL);
}

void destroy_instance(struct sample_info &info) { vkDestroyInstance(info.inst, NULL); }



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

void init_instance(struct sample_info &info, char const *const app_short_name) {
    info.instance_extension_names.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    vk::ApplicationInfo app_info;
    app_info.setPApplicationName(app_short_name)
            .setApplicationVersion(1)
            .setPEngineName(app_short_name)
            .setEngineVersion(1)
            .setApiVersion(VK_API_VERSION_1_0);

    vk::InstanceCreateInfo inst_info;
    inst_info.setPApplicationInfo(&app_info)
            .setEnabledLayerCount(info.instance_layer_names.size())
            .setPpEnabledLayerNames(info.instance_layer_names.size() ? info.instance_layer_names.data() : NULL)
            .setEnabledExtensionCount(info.instance_extension_names.size())
            .setPpEnabledExtensionNames(info.instance_extension_names.size() ? info.instance_extension_names.data() : NULL);

    info.inst = vk::createInstanceUnique(inst_info);

    info.getPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR) info.inst->getProcAddr("vkGetPhysicalDeviceFeatures2KHR");
}

void init_device(struct sample_info &info) {
    const float queue_priorities[1] = { 0.0f };

    vk::DeviceQueueCreateInfo queue_info;
    queue_info.setQueueCount(1)
            .setPQueuePriorities(queue_priorities)
            .setQueueFamilyIndex(info.graphics_queue_family_index);

    vk::DeviceCreateInfo device_info;
    device_info.setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&queue_info)
            .setEnabledExtensionCount(info.device_extension_names.size())
            .setPpEnabledExtensionNames(info.device_extension_names.size() ? info.device_extension_names.data() : NULL);

    info.device = info.gpu.createDeviceUnique(device_info);
}

void init_enumerate_device(struct sample_info &info) {
    auto gpus = info.inst->enumeratePhysicalDevices();
    // find the first physical device in the list which supports the variablePointers feature
    auto found_gpu = find_if(gpus.begin(), gpus.end(), [&info](vk::PhysicalDevice g) {
        vk::StructureChain<vk::PhysicalDeviceFeatures2KHR, vk::PhysicalDeviceVariablePointerFeaturesKHR> structureChain;
        vk::PhysicalDeviceFeatures2KHR& features = structureChain.get<vk::PhysicalDeviceFeatures2KHR>();
        info.getPhysicalDeviceFeatures2KHR((VkPhysicalDevice)g, reinterpret_cast<VkPhysicalDeviceFeatures2KHR*>(&features));

        return structureChain.get<vk::PhysicalDeviceVariablePointerFeaturesKHR>().variablePointers;
    });
    if (found_gpu == gpus.end()) {
        throw std::runtime_error("no compatible GPU found");
    }
    info.gpu = *found_gpu;

    /* This is as good a place as any to do this */
    info.memory_properties = info.gpu.getMemoryProperties();
    info.physical_device_properties = info.gpu.getProperties();
}

void init_debug_report_callback(struct sample_info &info, PFN_vkDebugReportCallbackEXT dbgFunc) {
    vk::DebugReportCallbackCreateInfoEXT create_info;
    create_info.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
            .setPfnCallback(dbgFunc)
            .setPUserData(NULL);

    info.debug_report_callbacks.push_back(info.inst->createDebugReportCallbackEXTUnique(create_info));
}

void init_command_pool(struct sample_info &info) {
    vk::CommandPoolCreateInfo cmd_pool_info;
    cmd_pool_info.setQueueFamilyIndex(info.graphics_queue_family_index)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    info.cmd_pool = info.device->createCommandPoolUnique(cmd_pool_info);
}

void init_device_queue(struct sample_info &info) {
    info.graphics_queue = info.device->getQueue(info.graphics_queue_family_index, 0);
}

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

#ifndef CLSPVTEST_UTIL_HPP
#define CLSPVTEST_UTIL_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <unistd.h>
#include <android/log.h>

#include <vulkan/vulkan.hpp>

#if defined(NDEBUG) && defined(__GNUC__)
#define U_ASSERT_ONLY __attribute__((unused))
#else
#define U_ASSERT_ONLY
#endif

/*
 * A layer can expose extensions, keep track of those
 * extensions here.
 */
struct layer_properties {
    vk::LayerProperties properties;
    std::vector<vk::ExtensionProperties> extensions;
};

/*
 * Structure for tracking information used / created / modified
 * by utility functions.
 */
struct sample_info {
    std::vector<const char *> instance_layer_names;
    std::vector<const char *> instance_extension_names;
    std::vector<layer_properties> instance_layer_properties;
    VkInstance inst;

    std::vector<const char *> device_extension_names;
    std::vector<VkPhysicalDevice> gpus;
    VkDevice device;
    VkQueue graphics_queue;
    VkQueue present_queue;
    uint32_t graphics_queue_family_index;
    uint32_t present_queue_family_index;
    VkPhysicalDeviceProperties gpu_props;
    std::vector<VkQueueFamilyProperties> queue_props;
    VkPhysicalDeviceMemoryProperties memory_properties;

    VkCommandPool cmd_pool;

    VkDescriptorPool desc_pool;

    PFN_vkCreateDebugReportCallbackEXT dbgCreateDebugReportCallback;
    PFN_vkDestroyDebugReportCallbackEXT dbgDestroyDebugReportCallback;
    std::vector<VkDebugReportCallbackEXT> debug_report_callbacks;
};

// Main entry point of samples
int sample_main(int argc, char *argv[]);

// Android specific definitions & helpers.
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "VK-SAMPLE", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "VK-SAMPLE", __VA_ARGS__))
// Replace printf to logcat output.
#define printf(...) __android_log_print(ANDROID_LOG_DEBUG, "VK-SAMPLE", __VA_ARGS__);

bool Android_process_command();
ANativeWindow* AndroidGetApplicationWindow();
FILE* AndroidFopen(const char* fname, const char* mode);
void AndroidGetWindowSize(int32_t *width, int32_t *height);
bool AndroidLoadFile(const char* filePath, std::string *data);

#endif // CLSPVTEST_UTIL_HPP

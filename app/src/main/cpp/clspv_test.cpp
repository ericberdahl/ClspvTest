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
#include "copybuffertoimage_kernel.hpp"
#include "copyimagetobuffer_kernel.hpp"
#include "fill_kernel.hpp"
#include "fp_utils.hpp"
#include "gpu_types.hpp"
#include "half.hpp"
#include "opencl_types.hpp"
#include "pixels.hpp"
#include "readconstantdata_kernel.hpp"
#include "readlocalsize_kernel.hpp"
#include "testgreaterthanorequalto_kernel.hpp"
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
    /* This routine simply finds a compute queue for a later vkCreateDevice.
     */

    auto queue_props = info.gpu.getQueueFamilyProperties();

    auto found = std::find_if(queue_props.begin(), queue_props.end(), [](vk::QueueFamilyProperties p) {
        return (p.queueFlags & vk::QueueFlagBits::eCompute);
    });

    info.graphics_queue_family_index = std::distance(queue_props.begin(), found);
    assert(found != queue_props.end());
}

void my_init_descriptor_pool(struct sample_info &info) {
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

    info.desc_pool = info.device->createDescriptorPoolUnique(createInfo);
}

vk::UniqueSampler create_compatible_sampler(vk::Device device, int opencl_flags) {
    typedef std::pair<int,vk::SamplerAddressMode> address_mode_map;
    const address_mode_map address_mode_translator[] = {
            { CLK_ADDRESS_NONE, vk::SamplerAddressMode::eRepeat },
            { CLK_ADDRESS_CLAMP_TO_EDGE, vk::SamplerAddressMode::eClampToEdge },
            { CLK_ADDRESS_CLAMP, vk::SamplerAddressMode::eClampToBorder },
            { CLK_ADDRESS_REPEAT, vk::SamplerAddressMode::eRepeat },
            { CLK_ADDRESS_MIRRORED_REPEAT, vk::SamplerAddressMode::eMirroredRepeat }
    };

    const vk::Filter filter = ((opencl_flags & CLK_FILTER_MASK) == CLK_FILTER_LINEAR ?
                             vk::Filter::eLinear :
                             vk::Filter::eNearest);
    const vk::Bool32 unnormalizedCoordinates = ((opencl_flags & CLK_NORMALIZED_COORDS_MASK) == CLK_NORMALIZED_COORDS_FALSE ? VK_TRUE : VK_FALSE);

    const auto found_map = std::find_if(std::begin(address_mode_translator), std::end(address_mode_translator), [&opencl_flags](const address_mode_map& am) {
        return (am.first == (opencl_flags & CLK_ADDRESS_MASK));
    });
    const vk::SamplerAddressMode addressMode = (found_map == std::end(address_mode_translator) ? vk::SamplerAddressMode::eRepeat : found_map->second);

    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.setMagFilter(filter)
            .setMinFilter(filter)
            .setMipmapMode(vk::SamplerMipmapMode::eNearest)
            .setAddressModeU(addressMode)
            .setAddressModeV(addressMode)
            .setAddressModeW(addressMode)
            .setAnisotropyEnable(VK_FALSE)
            .setCompareEnable(VK_FALSE)
            .setUnnormalizedCoordinates(unnormalizedCoordinates);

    return device.createSamplerUnique(samplerCreateInfo);
}

/* ============================================================================================== */

const test_utils::module_test_bundle module_tests[] = {
        {
                "shaders_cl/localsize",
                {
                        {"ReadLocalSize", readlocalsize_kernel::test }
                },
        },
        {
                "shaders_cl/Fills",
                {
                        { "FillWithColorKernel", fill_kernel::test_series, { 32, 32 } }
                }
        },
        {
                "shaders_cl/Memory",
                {
                        { "CopyBufferToImageKernel", copybuffertoimage_kernel::test_matrix, { 32, 32 } },
                        { "CopyImageToBufferKernel", copyimagetobuffer_kernel::test_matrix, { 32, 32 } }
                }
        },
        {
                "shaders_cl/ReadConstantData",
                {
                        {"ReadConstantData", readconstantdata_kernel::test_all, { 32, 1 } }
                },
        },
        {
                "shaders_cl/TestComparisons",
                {
                        {"TestGreaterThanOrEqualTo", testgreaterthanorequalto_kernel::test_all, { 32, 32 } }
                },
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

test_utils::Results run_all_tests(const sample_info& info, vk::ArrayProxy<const vk::Sampler> samplers) {
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
    std::vector<vk::UniqueSampler> samplers;
    std::transform(std::begin(sampler_flags), std::end(sampler_flags),
                   std::back_inserter(samplers),
                   std::bind(create_compatible_sampler, *info.device, std::placeholders::_1));

    const auto rawSamplers = vulkan_utils::extractUniques(samplers);
    const auto test_results = run_all_tests(info, rawSamplers);

    //
    // Clean up
    //

    samplers.clear();
    info.desc_pool.reset();
    info.cmd_pool.reset();
    info.device->waitIdle();
    info.device.reset();

    LOGI("Complete! %d tests passed. %d tests failed. %d kernels loaded, %d kernels skipped, %d kernels failed",
         test_results.mNumTestSuccess,
         test_results.mNumTestFail,
         test_results.mNumKernelLoadSuccess,
         test_results.mNumKernelLoadSkip,
         test_results.mNumKernelLoadFail);

    return 0;
}

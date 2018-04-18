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

std::string read_asset_file(const std::string& fileName) {
    std::string result;

    std::unique_ptr<std::FILE, decltype(&std::fclose)> assetFile(AndroidFopen(fileName.c_str(), "r"),
                                                                 &std::fclose);
    if (assetFile) {
        std::fseek(assetFile.get(), 0, SEEK_END);
        result.resize(std::ftell(assetFile.get()), ' ');

        std::fseek(assetFile.get(), 0, SEEK_SET);
        std::fread(&result.front(), 1, result.length(), assetFile.get());
    }
    else {
        LOGI("Asset file '%s' not found", fileName.c_str());
    }

    return result;
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

struct test_map_t {
    const char*                 name;
    test_utils::test_kernel_fn  test_fn;
};

const test_map_t test_map[] = {
        { "readLocalSize",      readlocalsize_kernel::test },
        { "fill",               fill_kernel::test_series },
        { "fill<float4>",       fill_kernel::test<gpu_types::float4> },
        { "fill<half4>",        fill_kernel::test<gpu_types::half4> },
        { "copyImageToBuffer",  copyimagetobuffer_kernel::test_matrix },
        { "copyBufferToImage",  copybuffertoimage_kernel::test_matrix },
        { "readConstantData",   readconstantdata_kernel::test_all },
        { "testGtEq",           testgreaterthanorequalto_kernel::test_all },
};

test_utils::test_kernel_fn lookup_test_fn(const std::string& testName) {
    test_utils::test_kernel_fn result = nullptr;

    auto found = std::find_if(std::begin(test_map), std::end(test_map), [&testName](const test_map_t& entry){ return testName == entry.name; });
    if (found != std::end(test_map)) {
        result = found->test_fn;
    }

    return result;
}

struct manifest_t {
    bool                                        use_validation_layer = true;
    std::vector<test_utils::module_test_bundle> tests;
};

manifest_t read_manifest(std::istream& in) {
    manifest_t      result;
    bool            verbose     = false;

    test_utils::module_test_bundle* currentModule = NULL;
    while (!in.eof()) {
        std::string line;
        std::getline(in, line);

        std::istringstream in_line(line);

        std::string op;
        in_line >> op;
        if (op.empty() || op[0] == '#') {
            // line is either blank or a comment, skip it
        }
        else if (op == "module") {
            // add module to list of modules to load
            test_utils::module_test_bundle moduleEntry;
            in_line >> moduleEntry.name;

            result.tests.push_back(moduleEntry);
            currentModule = &result.tests.back();
        }
        else if (op == "test") {
            // test kernel in module
            if (currentModule) {
                test_utils::kernel_test_map testEntry;
                testEntry.verbose = verbose;

                std::string testName;
                in_line >> testEntry.entry
                        >> testName
                        >> testEntry.workgroupSize.x
                        >> testEntry.workgroupSize.y;

                while (!in_line.eof()) {
                    std::string arg;
                    in_line >> arg;

                    // comment delimiter halts collection of test arguments
                    if (arg[0] == '#') break;

                    testEntry.args.push_back(arg);
                }

                testEntry.test = lookup_test_fn(testName);

                bool lineIsGood = true;

                if (!testEntry.test) {
                    LOGE("%s: cannot find test '%s' from command '%s'",
                         __func__,
                         testName.c_str(),
                         line.c_str());
                    lineIsGood = false;
                }
                if (1 > testEntry.workgroupSize.x || 1 > testEntry.workgroupSize.y) {
                    LOGE("%s: bad workgroup dimensions {%d,%d} from command '%s'",
                         __func__,
                         testEntry.workgroupSize.x,
                         testEntry.workgroupSize.y,
                         line.c_str());
                    lineIsGood = false;
                }

                if (lineIsGood) {
                    currentModule->kernelTests.push_back(testEntry);
                }
            }
            else {
                LOGE("%s: no module for test '%s'", __func__, line.c_str());
            }
        }
        else if (op == "skip") {
            // skip kernel in module
            if (currentModule) {
                test_utils::kernel_test_map skipEntry;
                skipEntry.workgroupSize = clspv_utils::WorkgroupDimensions(0, 0);

                in_line >> skipEntry.entry;

                currentModule->kernelTests.push_back(skipEntry);
            }
            else {
                LOGE("%s: no module for skip '%s'", __func__, line.c_str());
            }
        }
        else if (op == "vkValidation") {
            // turn vulkan validation layers on/off
            std::string on_off;
            in_line >> on_off;

            if (on_off == "all") {
                result.use_validation_layer = true;
            }
            else if (on_off == "none") {
                result.use_validation_layer = false;
            }
            else {
                LOGE("%s: unrecognized vkValidation token '%s'", __func__, on_off.c_str());
            }
        }
        else if (op == "verbosity") {
            // set verbosity of tests
            std::string verbose_level;
            in_line >> verbose_level;

            if (verbose_level == "full") {
                verbose = true;
            }
            else if (verbose_level == "silent") {
                verbose = false;
            }
            else {
                LOGE("%s: unrecognized verbose level '%s'", __func__, verbose_level.c_str());
            }
        }
        else if (op == "end") {
            // terminate reading the manifest
            break;
        }
        else {
            LOGE("%s: ignoring ill-formed line '%s'", __func__, line.c_str());
        }
    }

    return result;
}

manifest_t read_manifest(const std::string& inManifest) {
    std::istringstream is(inManifest);
    return read_manifest(is);
}

void run_manifest(const manifest_t&                     manifest,
                  const sample_info&                    info,
                  vk::ArrayProxy<const vk::Sampler>     samplers,
                  test_utils::ModuleResultSet&          resultSet) {
    for (auto m : manifest.tests) {
        resultSet.push_back(test_utils::test_module(m.name, m.kernelTests, info, samplers));
    }
}

std::pair<unsigned int, unsigned int> countResults(const test_utils::InvocationResult& ir) {
    // an invocation passes if it generates at least one correct value and no incorrect values
    return (ir.mNumCorrect > 0 && ir.mNumErrors == 0 ? std::make_pair(1, 0) : std::make_pair(0, 1));
};

std::pair<unsigned int, unsigned int> countResults(const test_utils::KernelResult& kr) {
    // a kernel's results are the aggregate sum of its invocations
    return std::accumulate(kr.mInvocations.begin(), kr.mInvocations.end(),
                           std::make_pair(0, 0),
                           [](std::pair<unsigned int, unsigned int> r, const test_utils::InvocationResult& ir) {
                               auto addend = countResults(ir);
                               r.first += addend.first;
                               r.second += addend.second;
                               return r;
                           });
};

std::pair<unsigned int, unsigned int> countResults(const test_utils::ModuleResult& mr) {
    // a module's results are the aggregate sum of its kernels, combined with the result of its own
    // loading (i.e. whether it loaded correctly or not)
    return std::accumulate(mr.mKernels.begin(), mr.mKernels.end(),
                           mr.mLoadedCorrectly ? std::make_pair(1, 0) : std::make_pair(0, 1),
                           [](std::pair<unsigned int, unsigned int> r, const test_utils::KernelResult& kr) {
                               auto addend = countResults(kr);
                               r.first += addend.first;
                               r.second += addend.second;
                               return r;
                           });
};

std::pair<unsigned int, unsigned int> countResults(const test_utils::ModuleResultSet& moduleResultSet) {
    return std::accumulate(moduleResultSet.begin(), moduleResultSet.end(),
                           std::make_pair(0, 0),
                           [](std::pair<unsigned int, unsigned int> r, const test_utils::ModuleResult& mr) {
                               auto addend = countResults(mr);
                               r.first += addend.first;
                               r.second += addend.second;
                               return r;
                           });
};

void logPhysicalDeviceInfo(sample_info& info) {
    const vk::PhysicalDeviceProperties props = info.gpu.getProperties();
    std::ostringstream os;
    os << "PhysicalDevice {" << std::endl
       << "   apiVersion:" << props.apiVersion << std::endl
       << "   driverVersion:" << props.driverVersion << std::endl
       << "   vendorID:" << props.vendorID << std::endl
       << "   deviceID:" << props.deviceID << std::endl
       << "   deviceName:" << props.deviceName << std::endl
       << "}";
    LOGI("%s", os.str().c_str());
}

void logResults(sample_info& info, const test_utils::InvocationResult& ir) {
    std::ostringstream os;

    const clspv_utils::execution_time_t totalTime = ir.mExecutionTime;
    os << (ir.mNumCorrect > 0 && ir.mNumErrors == 0 ? "PASS" : "FAIL");

    if (!ir.mVariation.empty()) {
        os << " variation:" << ir.mVariation << "";
    }

    os << " correctValues:" << ir.mNumCorrect
       << " incorrectValues:" << ir.mNumErrors
       << " wallClockTime:" << totalTime.cpu_duration.count() * 1000.0 << "ms"
       << " executionTime:" << vulkan_utils::timestamp_delta_ns(totalTime.timestamps.host_barrier, totalTime.timestamps.execution, info.physical_device_properties, info.graphics_queue_family_properties)/1000.0f << "µs"
       << " hostBarrierTime:" << vulkan_utils::timestamp_delta_ns(totalTime.timestamps.start, totalTime.timestamps.host_barrier, info.physical_device_properties, info.graphics_queue_family_properties)/1000.0f << "µs"
       << " gpuBarrierTime:" << vulkan_utils::timestamp_delta_ns(totalTime.timestamps.execution, totalTime.timestamps.gpu_barrier, info.physical_device_properties, info.graphics_queue_family_properties)/1000.0f << "µs";

    LOGI("      %s", os.str().c_str());

    for (auto err : ir.mMessages) {
        LOGD("         %s", err.c_str());
    }
}

void logResults(sample_info& info, const test_utils::KernelResult& kr) {
    std::ostringstream os;

    os << "Kernel:" << kr.mEntryName;
    if (kr.mSkipped) {
        os << " SKIPPED";
    }
    else if (!kr.mCompiledCorrectly) {
        os << " COMPILE-FAILURE";
    }
    if (!kr.mExceptionString.empty()) {
    	os << " " << kr.mExceptionString;
    }
    LOGI("   %s", os.str().c_str());

    for (auto ir : kr.mInvocations) {
        logResults(info, ir);
    }
}

void logResults(sample_info& info, const test_utils::ModuleResult& mr) {
    std::ostringstream os;
    os << "Module:" << mr.mModuleName;
    if (!mr.mExceptionString.empty()) {
        os << " loadException:" << mr.mExceptionString;
    }
    LOGI("%s", os.str().c_str());

    for (auto kr : mr.mKernels) {
        logResults(info, kr);
    }
}

void logResults(sample_info& info, const test_utils::ModuleResultSet& moduleResultSet) {
    logPhysicalDeviceInfo(info);

    for (auto mr : moduleResultSet) {
        logResults(info, mr);
    }

    auto results = countResults(moduleResultSet);

    std::ostringstream os;
    os << "Overall Summary"
       << " pass:" << results.first << " fail:" << results.second;
    LOGI("%s", os.str().c_str());
}

/* ============================================================================================== */

int sample_main(int argc, char *argv[]) {
    const auto manifest = read_manifest(read_asset_file("test_manifest.txt"));

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

    // This sample presumes that all OpenCL C kernels were compiled with the same samplermap file,
    // whose contents and order are statically known to the application. Thus, the app can create
    // a set of compatible samplers thusly.
    const int sampler_flags[] = {
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_NONE            | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_FALSE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_LINEAR     | CLK_NORMALIZED_COORDS_TRUE,
            CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_TRUE,
            CLK_ADDRESS_NONE            | CLK_FILTER_NEAREST    | CLK_NORMALIZED_COORDS_TRUE
    };
    std::vector<vk::UniqueSampler> samplers;
    std::transform(std::begin(sampler_flags), std::end(sampler_flags),
                   std::back_inserter(samplers),
                   std::bind(create_compatible_sampler, *info.device, std::placeholders::_1));

    const auto rawSamplers = vulkan_utils::extractUniques(samplers);

    test_utils::ModuleResultSet moduleResultSet;
    run_manifest(manifest, info, rawSamplers, moduleResultSet);

    logResults(info, moduleResultSet);

    //
    // Clean up
    //

    samplers.clear();
    info.desc_pool.reset();
    info.cmd_pool.reset();
    info.device->waitIdle();
    info.device.reset();

    LOGI("ClspvTest complete!!");

    return 0;
}

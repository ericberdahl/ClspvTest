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

#include <vulkan/vulkan.hpp>

#include <android/asset_manager.h>
#include <android/log.h>

#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/iostreams/stream.hpp>

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
    std::vector<const char *>           instance_layer_names;
    std::vector<const char *>           instance_extension_names;
    std::vector<layer_properties>       instance_layer_properties;
    vk::UniqueInstance                  inst;
    PFN_vkGetPhysicalDeviceFeatures2KHR getPhysicalDeviceFeatures2KHR   = nullptr;

    std::vector<const char *>           device_extension_names;
    vk::PhysicalDevice                  gpu;
    vk::UniqueDevice                    device;
    vk::Queue                           graphics_queue;

    uint32_t                            graphics_queue_family_index     = 0;
    vk::QueueFamilyProperties           graphics_queue_family_properties;

    vk::PhysicalDeviceProperties        physical_device_properties;
    vk::UniqueCommandPool               cmd_pool;
    vk::UniqueDescriptorPool            desc_pool;

    std::vector<vk::UniqueDebugReportCallbackEXT> debug_report_callbacks;
};

// Main entry point of samples
int sample_main(int argc, char *argv[]);

// Android specific definitions & helpers.
#define LOG(LEVEL, ...) ((void)__android_log_print(ANDROID_LOG_##LEVEL, "VK-SAMPLE", __VA_ARGS__))
#define LOGD(...) LOG(DEBUG, __VA_ARGS__)
#define LOGI(...) LOG(INFO, __VA_ARGS__)
#define LOGW(...) LOG(WARN, __VA_ARGS__)
#define LOGE(...) LOG(ERROR, __VA_ARGS__)
// Replace printf to logcat output.
#define printf(...) LOGD(__VA_ARGS__)

bool Android_process_command();
ANativeWindow* AndroidGetApplicationWindow();

namespace android_utils {
    FILE* asset_fopen(const char* fname, const char* mode);

    // Helpder class to forward the cout/cerr output to logcat derived from:
    // http://stackoverflow.com/questions/8870174/is-stdcout-usable-in-android-ndk
    class LogBuffer : public std::streambuf {
    public:
        LogBuffer(android_LogPriority priority);

    private:
        static const std::int32_t    kBufferSize = 128;

        virtual int_type overflow(int_type c) override;

        virtual int_type sync() override;

    private:
        android_LogPriority priority_ = ANDROID_LOG_INFO;
        char                buffer_[kBufferSize];
    };

    class AssetSource {
    public:
        typedef char char_type;
        struct category
                : boost::iostreams::input_seekable,
                  boost::iostreams::device_tag,
                  boost::iostreams::closable_tag {
        };

        // Default constructor
        AssetSource() {}

        // Constructor taking a std:: string
        explicit AssetSource(const std::string &path,
                             std::ios::openmode mode = std::ios::in) {
            open(path, mode);
        }

        // Constructor taking a C-style string
        explicit AssetSource(const char *path,
                             std::ios::openmode mode = std::ios::in) {
            open(path, mode);
        }

        AssetSource(const AssetSource &other);

        ~AssetSource() {}

        bool is_open() const { return (nullptr != mAsset.get()); }

        void open(const std::string &path, std::ios::openmode mode = std::ios::in) {
            open(path.c_str(), mode);
        }

        void open(const char *path, std::ios::openmode mode = std::ios::in);

        std::streamsize read(char_type *s, std::streamsize n);

        std::streampos seek(boost::iostreams::stream_offset off, std::ios_base::seekdir way);

        void close();

    private:
        std::shared_ptr<AAsset> mAsset;
    };

    typedef boost::iostreams::stream<AssetSource> iassetstream;
}

#endif // CLSPVTEST_UTIL_HPP

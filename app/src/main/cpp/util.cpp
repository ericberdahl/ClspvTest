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
samples utility functions
*/

#include <assert.h>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "util.hpp"

#include <android_native_app_glue.h>

// Static variable that keeps ANativeWindow and asset manager instances.
static android_app *Android_application = nullptr;

namespace {
    // Android fopen stub described at
    // http://www.50ply.com/blog/2013/01/19/loading-compressed-android-assets-with-file-pointer/#comment-1850768990
    int android_read(void *cookie, char *buf, int size) {
        return AAsset_read(static_cast<AAsset *>(cookie), buf, size);
    }

    int android_write(void *cookie, const char *buf, int size) {
        return EACCES;  // can't provide write access to the apk
    }

    fpos_t android_seek(void *cookie, fpos_t offset, int whence) {
        return AAsset_seek(static_cast<AAsset *>(cookie), offset, whence);
    }

    int android_close(void *cookie) {
        AAsset_close(static_cast<AAsset *>(cookie));
        return 0;
    }
}
//
// Android specific helper functions.
//

void Android_handle_cmd(android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            sample_main(0, nullptr);
            LOGI("\n");
            LOGI("=================================================");
            LOGI("          The sample ran successfully!!");
            LOGI("=================================================");
            LOGI("\n");
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            break;
        default:
            LOGI("event not handled: %d", cmd);
    }
}

bool Android_process_command() {
    assert(Android_application != nullptr);
    int events;
    android_poll_source *source;
    // Poll all pending events.
    if (ALooper_pollAll(0, NULL, &events, (void **)&source) >= 0) {
        // Process each polled events
        if (source != NULL) source->process(Android_application, source);
    }
    return Android_application->destroyRequested;
}

void android_main(struct android_app *app) {
    assert(nullptr == Android_application);

    // Set static variables.
    Android_application = app;
    // Set the callback to process system events
    app->onAppCmd = Android_handle_cmd;

    // Forward cout/cerr to logcat.
    std::cout.rdbuf(new android_utils::LogBuffer(ANDROID_LOG_INFO));
    std::cerr.rdbuf(new android_utils::LogBuffer(ANDROID_LOG_ERROR));

    // Main loop
    do {
        Android_process_command();
    }  // Check if system requested to quit the application
    while (app->destroyRequested == 0);

    return;
}

namespace android_utils {

    FILE* asset_fopen(const char *fname, const char *mode) {
        if (mode[0] == 'w') {
            return NULL;
        }

        assert(Android_application != nullptr);
        AAsset *asset = AAssetManager_open(Android_application->activity->assetManager, fname, 0);
        if (!asset) {
            return NULL;
        }

        return funopen(asset, android_read, android_write, android_seek, android_close);
    }

    LogBuffer::LogBuffer(android_LogPriority priority) {
        priority_ = priority;
        this->setp(buffer_, buffer_ + kBufferSize - 1);
    }

    LogBuffer::int_type LogBuffer::overflow(int_type c) {
        if (c == traits_type::eof()) {
            *this->pptr() = traits_type::to_char_type(c);
            this->sbumpc();
        }
        return this->sync() ? traits_type::eof() : traits_type::not_eof(c);
    }

    LogBuffer::int_type LogBuffer::sync() {
        int32_t rc = 0;
        if (this->pbase() != this->pptr()) {
            char writebuf[kBufferSize + 1];
            std::memcpy(writebuf, this->pbase(), this->pptr() - this->pbase());
            writebuf[this->pptr() - this->pbase()] = '\0';

            rc = __android_log_write(priority_, "std", writebuf) > 0;
            this->setp(buffer_, buffer_ + kBufferSize - 1);
        }
        return rc;
    }

    AssetSource::AssetSource(const AssetSource &other) :
            mAsset(other.mAsset) {

    }

    void AssetSource::open(const char *path, std::ios::openmode mode) {
        if (mode & (std::ios::out | std::ios::trunc))
            throw std::runtime_error("invalid mode");

        assert(Android_application != nullptr);
        AAsset *asset = AAssetManager_open(Android_application->activity->assetManager, path,
                                           AASSET_MODE_RANDOM);
        if (!asset) {
            throw std::runtime_error("asset not found");
        }

        mAsset.reset(asset, &AAsset_close);
    }

    std::streamsize AssetSource::read(char_type *s, std::streamsize n) {
        return AAsset_read(mAsset.get(), s, n);
    }

    std::streampos AssetSource::seek(boost::iostreams::stream_offset off,
                                     std::ios_base::seekdir way) {
        return AAsset_seek(mAsset.get(), off, way);
    }

    void AssetSource::close() {
        mAsset.reset();
    }

}
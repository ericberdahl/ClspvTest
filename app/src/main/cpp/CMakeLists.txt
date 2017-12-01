#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.4.1)

# build native_app_glue as a static lib
set(${CMAKE_C_FLAGS}, "${CMAKE_C_FLAGS}")
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

get_filename_component(PROJECT_SOURCE_DIR
                       "${CMAKE_SOURCE_DIR}/.."
                       ABSOLUTE)

# now build app's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror -DVK_USE_PLATFORM_ANDROID_KHR")

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native-activity SHARED
    copyimagetobuffer_kernel.cpp
    copybuffertoimage_kernel.cpp
    fill_kernel.cpp
    readlocalsize_kernel.cpp
    clspv_test.cpp
    clspv_utils.cpp
    gpu_types.cpp
    test_utils.cpp
    vulkan_utils.cpp
    util.cpp
    util_init.cpp)

target_include_directories(native-activity PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${ANDROID_NDK}/sources/android/native_app_glue)

# add lib dependencies
target_link_libraries(native-activity
    android
    native_app_glue
    vulkan
    log)

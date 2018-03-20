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
    readconstantdata_kernel.cpp
    readlocalsize_kernel.cpp
    testgreaterthanorequalto_kernel.cpp
    clspv_test.cpp
    clspv_utils.cpp
    gpu_types.cpp
    test_utils.cpp
    vulkan_utils.cpp
    util.cpp
    util_init.cpp
    )

target_include_directories(native-activity PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${ANDROID_NDK}/sources/android/native_app_glue)

# add lib dependencies
target_link_libraries(native-activity
    android
    native_app_glue
    vulkan
    log)

# build OpenCL C kernels
set(CLSHADER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/kernels)
set(CLSHADER_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/assets/shaders_cl)

set(CLSPV_COMMAND /usr/local/bin/clspv)
set(SPRIV_OPT_COMMAND /usr/local/bin/spirv-opt)

set(CLSPV_SAMPLERMAP ${CLSHADER_SOURCE_DIR}/sampler_map)

set(CLSPV_FLAGS)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -cl-single-precision-constant -cl-fast-relaxed-math -cl-denorms-are-zero -cl-mad-enable)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -samplermap=${CLSHADER_SOURCE_DIR}/sampler_map)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -cluster-pod-kernel-args)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-undef)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-inserts)

set(SPRIV_OPT_FLAGS)
set(SPRIV_OPT_FLAGS ${SPRIV_OPT_FLAGS} --set-spec-const-default-value "0:2 1:2 2:2")
set(SPRIV_OPT_FLAGS ${SPRIV_OPT_FLAGS} --inline-entry-points-exhaustive)

set(OPENCL_KERNELS
    Fills
    localsize
    Memory
    ReadConstantData
    TestComparisons
    )

set(kernel_binaries)
foreach (kernel ${OPENCL_KERNELS})
    add_custom_command(
        OUTPUT ${CLSHADER_OUTPUT_DIR}/${kernel}.spv ${CLSHADER_OUTPUT_DIR}/${kernel}.spvmap
        COMMAND ${CLSPV_COMMAND} ${CLSHADER_SOURCE_DIR}/${kernel}.cl -o=${CLSHADER_OUTPUT_DIR}/${kernel}.spvx -descriptormap=${CLSHADER_OUTPUT_DIR}/${kernel}.spvmap ${CLSPV_FLAGS}
        COMMAND ${SPRIV_OPT_COMMAND} ${CLSHADER_OUTPUT_DIR}/${kernel}.spvx -o ${CLSHADER_OUTPUT_DIR}/${kernel}.spv ${SPRIV_OPT_FLAGS}
        COMMAND ${CMAKE_COMMAND} -E remove ${CLSHADER_OUTPUT_DIR}/${kernel}.spvx
        DEPENDS ${CLSHADER_SOURCE_DIR}/${kernel}.cl ${CLSHADER_SOURCE_DIR}/sampler_map
        VERBATIM
        )
    list(APPEND kernel_binaries ${CLSHADER_OUTPUT_DIR}/${kernel}.spv ${CLSHADER_OUTPUT_DIR}/${kernel}.spvmap)
endforeach (kernel ${OPENCL_KERNELS})

add_custom_target(build-cl-shaders
                  DEPENDS ${CLSPV_SAMPLERMAP} ${kernel_binaries})
add_custom_command(TARGET build-cl-shaders
                   PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CLSHADER_OUTPUT_DIR})

add_dependencies(native-activity build-cl-shaders)

# build GLSL kernels
set(GLSL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/kernels)
set(GLSL_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/assets/shaders)

set(GLSLANG_COMMAND /usr/local/bin/glslangValidator)

set(GLSLANG_FLAGS)
set(GLSLANG_FLAGS ${GLSLANG_FLAGS} -V)

set(GLSL_KERNELS
    GL_Fills_reduced
    )

set(gl_kernel_binaries)
foreach (kernel ${GLSL_KERNELS})
    add_custom_command(
        OUTPUT ${GLSL_OUTPUT_DIR}/${kernel}.spv ${GLSL_OUTPUT_DIR}/${kernel}.spvmap
        COMMAND ${GLSLANG_COMMAND} ${GLSL_SOURCE_DIR}/${kernel}.comp -o ${GLSL_OUTPUT_DIR}/${kernel}.spv ${GLSLANG_FLAGS}
        DEPENDS ${GLSL_SOURCE_DIR}/${kernel}.comp ${GLSL_SOURCE_DIR}/${kernel}.spvmap
        VERBATIM
        )
    list(APPEND gl_kernel_binaries ${GLSL_OUTPUT_DIR}/${kernel}.spv)
endforeach (kernel ${GLSL_KERNELS})

add_custom_target(build-gl-shaders
                  DEPENDS ${gl_kernel_binaries})
add_custom_command(TARGET build-gl-shaders
                   PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${GLSL_OUTPUT_DIR})

add_dependencies(native-activity build-gl-shaders)

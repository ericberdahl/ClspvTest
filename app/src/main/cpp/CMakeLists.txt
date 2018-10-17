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

# locations of required external tools
set(CLSPV_COMMAND /usr/local/bin/clspv)
set(SPRIV_OPT_COMMAND /usr/local/bin/spirv-opt)
set(GLSLANG_COMMAND /usr/local/bin/glslangValidator)

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
        clspv_test.cpp
        gpu_types.cpp
        test_manifest.cpp
        test_result_logging.cpp
        test_utils.cpp
        util.cpp
        util_init.cpp
        clspv_utils/clspv_utils_interop.cpp
        clspv_utils/device.cpp
        file_utils.cpp
        crlf_savvy.cpp
        clspv_utils/interface.cpp
        clspv_utils/invocation.cpp
        clspv_utils/kernel.cpp
        clspv_utils/module.cpp
        kernel_tests/copyimagetobuffer_kernel.cpp
        kernel_tests/copybuffertoimage_kernel.cpp
        kernel_tests/fillarraystruct_kernel.cpp
        kernel_tests/fill_kernel.cpp
        kernel_tests/readconstantdata_kernel.cpp
        kernel_tests/readlocalsize_kernel.cpp
        kernel_tests/resample2dimage_kernel.cpp
        kernel_tests/resample3dimage_kernel.cpp
        kernel_tests/strangeshuffle_kernel.cpp
        kernel_tests/testgreaterthanorequalto_kernel.cpp
        vulkan_utils/vulkan_utils.cpp
        )

target_include_directories(native-activity PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp
    ${ANDROID_NDK}/sources/android/native_app_glue)

# add lib dependencies
target_link_libraries(native-activity
    android
    native_app_glue
    vulkan
    log)

# build OpenCL C kernels
set(CLSHADER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/kernels)
set(CLSHADER_INLINE_DIR ${PROJECT_SOURCE_DIR}/assets/shaders_inlined_cl)
set(CLSHADER_NOINLINE_DIR ${PROJECT_SOURCE_DIR}/assets/shaders_cl)

set(CLSPV_SAMPLERMAP ${CLSHADER_SOURCE_DIR}/sampler_map)

set(CLSPV_FLAGS)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -cl-single-precision-constant -cl-fast-relaxed-math -cl-denorms-are-zero -cl-mad-enable)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -samplermap=${CLSHADER_SOURCE_DIR}/sampler_map)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -cluster-pod-kernel-args)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-initializers)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-inserts)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-phis)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-scf)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -hack-undef)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -pod-ubo)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -enable-pre=0)
set(CLSPV_FLAGS ${CLSPV_FLAGS} -enable-load-pre=0)

set(SPIRV_OPT_FLAGS)
set(SPIRV_OPT_FLAGS ${SPIRV_OPT_FLAGS} --set-spec-const-default-value "0:2 1:2 2:2")

set(OPENCL_KERNELS
        Fills
        LocalMemory
        localsize
        Memory
        ReadConstantData
        StructArrays
        TestComparisons
        )

set(kernel_binaries)
foreach (kernel ${OPENCL_KERNELS})
    add_custom_command(
            OUTPUT ${CLSHADER_NOINLINE_DIR}/${kernel}.spv ${CLSHADER_NOINLINE_DIR}/${kernel}.spvmap
            COMMAND ${CLSPV_COMMAND} ${CLSHADER_SOURCE_DIR}/${kernel}.cl -o=${CLSHADER_NOINLINE_DIR}/${kernel}.spvx -descriptormap=${CLSHADER_NOINLINE_DIR}/${kernel}.spvmap ${CLSPV_FLAGS}
            COMMAND ${SPRIV_OPT_COMMAND} ${SPIRV_OPT_FLAGS} -Oconfig=${CLSHADER_SOURCE_DIR}/spirv-opt.config ${CLSHADER_NOINLINE_DIR}/${kernel}.spvx -o ${CLSHADER_NOINLINE_DIR}/${kernel}.spv
            COMMAND ${CMAKE_COMMAND} -E remove ${CLSHADER_NOINLINE_DIR}/${kernel}.spvx
            DEPENDS ${CLSHADER_SOURCE_DIR}/${kernel}.cl ${CLSHADER_SOURCE_DIR}/sampler_map ${CLSHADER_SOURCE_DIR}/spirv-opt.config
            VERBATIM
    )
    list(APPEND kernel_binaries ${CLSHADER_NOINLINE_DIR}/${kernel}.spv ${CLSHADER_NOINLINE_DIR}/${kernel}.spvmap)

    add_custom_command(
            OUTPUT ${CLSHADER_INLINE_DIR}/${kernel}.spv ${CLSHADER_INLINE_DIR}/${kernel}.spvmap
            COMMAND ${CLSPV_COMMAND} ${CLSHADER_SOURCE_DIR}/${kernel}.cl -o=${CLSHADER_INLINE_DIR}/${kernel}.spvx -descriptormap=${CLSHADER_INLINE_DIR}/${kernel}.spvmap ${CLSPV_FLAGS} -inline-entry-points
            COMMAND ${SPRIV_OPT_COMMAND} ${SPIRV_OPT_FLAGS} -Oconfig=${CLSHADER_SOURCE_DIR}/spirv-opt-inline.config -Oconfig=${CLSHADER_SOURCE_DIR}/spirv-opt.config ${CLSHADER_INLINE_DIR}/${kernel}.spvx -o ${CLSHADER_INLINE_DIR}/${kernel}.spv
            COMMAND ${CMAKE_COMMAND} -E remove ${CLSHADER_INLINE_DIR}/${kernel}.spvx
            DEPENDS ${CLSHADER_SOURCE_DIR}/${kernel}.cl ${CLSHADER_SOURCE_DIR}/sampler_map ${CLSHADER_SOURCE_DIR}/spirv-opt-inline.config ${CLSHADER_SOURCE_DIR}/spirv-opt.config
            VERBATIM
    )
    list(APPEND kernel_binaries ${CLSHADER_INLINE_DIR}/${kernel}.spv ${CLSHADER_INLINE_DIR}/${kernel}.spvmap)
endforeach (kernel ${OPENCL_KERNELS})

add_custom_target(build-cl-shaders
                  DEPENDS ${CLSPV_SAMPLERMAP} ${kernel_binaries})
add_custom_command(TARGET build-cl-shaders
                   PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CLSHADER_INLINE_DIR}
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CLSHADER_NOINLINE_DIR}
                   )

add_dependencies(native-activity build-cl-shaders)

# build GLSL kernels
set(GLSL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/kernels)
set(GLSL_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/assets/shaders)

set(GLSLANG_FLAGS)
set(GLSLANG_FLAGS ${GLSLANG_FLAGS} -V)

set(GLSL_KERNELS
    GL_Fills_reduced
    GL_LocalMemory
    )

set(gl_kernel_binaries)
foreach (kernel ${GLSL_KERNELS})
    set(kernel_outputs_i
        ${GLSL_OUTPUT_DIR}/${kernel}.spv
        ${GLSL_OUTPUT_DIR}/${kernel}.spvmap
        )
    add_custom_command(
        OUTPUT ${kernel_outputs_i}
        COMMAND ${GLSLANG_COMMAND} ${GLSL_SOURCE_DIR}/${kernel}.comp -o ${GLSL_OUTPUT_DIR}/${kernel}.spv ${GLSLANG_FLAGS}
        COMMAND ${CMAKE_COMMAND} -E copy ${GLSL_SOURCE_DIR}/${kernel}.spvmap ${GLSL_OUTPUT_DIR}/${kernel}.spvmap
        DEPENDS ${GLSL_SOURCE_DIR}/${kernel}.comp ${GLSL_SOURCE_DIR}/${kernel}.spvmap
        VERBATIM
        )
    list(APPEND gl_kernel_binaries ${kernel_outputs_i})
endforeach (kernel ${GLSL_KERNELS})

add_custom_target(build-gl-shaders
                  DEPENDS ${gl_kernel_binaries})
add_custom_command(TARGET build-gl-shaders
                   PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${GLSL_OUTPUT_DIR})

add_dependencies(native-activity build-gl-shaders)

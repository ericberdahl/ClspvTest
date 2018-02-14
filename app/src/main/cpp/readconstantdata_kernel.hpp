//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_READCONSTANTDATA_KERNEL_HPP
#define CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace readconstantdata_kernel {

    void invoke(const clspv_utils::kernel_module&   module,
                const clspv_utils::kernel&          kernel,
                const sample_info&                  info,
                vk::ArrayProxy<const vk::Sampler>   samplers,
                vk::Buffer                          dst_buffer,
                int                                 width);

    test_utils::Results test_all(const clspv_utils::kernel_module&    module,
                                 const clspv_utils::kernel&           kernel,
                                 const sample_info&                   info,
                                 vk::ArrayProxy<const vk::Sampler>    samplers,
                                 const test_utils::options&           opts);

}

#endif // CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

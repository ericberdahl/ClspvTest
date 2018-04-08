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

    clspv_utils::execution_time_t
    invoke(const clspv_utils::kernel_module&   module,
           const clspv_utils::kernel&          kernel,
           const sample_info&                  info,
           vk::ArrayProxy<const vk::Sampler>   samplers,
           vk::Buffer                          dst_buffer,
           int                                 width);

    void test_all(const clspv_utils::kernel_module&    module,
                  const clspv_utils::kernel&           kernel,
                  const sample_info&                   info,
                  vk::ArrayProxy<const vk::Sampler>    samplers,
                  test_utils::InvocationResultSet&     resultSet);

}

#endif // CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

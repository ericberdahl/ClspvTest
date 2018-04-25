//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP
#define CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.h>

namespace strangeshuffle_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&                 kernel,
           const sample_info&                   info,
           vk::ArrayProxy<const vk::Sampler>    samplers,
           vk::Buffer                           index_buffer,
           vk::Buffer                           source_buffer,
           vk::Buffer                           destination_buffer,
           std::size_t                          num_elements);

    void test(clspv_utils::kernel&              kernel,
              const sample_info&                info,
              vk::ArrayProxy<const vk::Sampler> samplers,
              const std::vector<std::string>&   args,
              bool                              verbose,
              test_utils::InvocationResultSet&  resultSet);
}

#endif //CLSPVTEST_STRANGESHUFFLE_KERNEL_HPP

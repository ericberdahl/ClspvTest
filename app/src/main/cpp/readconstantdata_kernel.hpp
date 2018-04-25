//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_READCONSTANTDATA_KERNEL_HPP
#define CLSPVTEST_READCONSTANTDATA_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace readconstantdata_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel& kernel,
           vk::Buffer           dst_buffer,
           int                  width);

    void test_all(clspv_utils::kernel&              kernel,
                  const std::vector<std::string>&   args,
                  bool                              verbose,
                  test_utils::InvocationResultSet&  resultSet);

}

#endif // CLSPVTEST_READCONSTANTDATA_KERNEL_HPP
